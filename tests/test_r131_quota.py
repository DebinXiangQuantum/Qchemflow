import json
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from chemflow.agent import llm_errors
from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy
from chemflow.eval import runner

RESET = "2026-10-14 18:20:12"


@pytest.mark.parametrize("code,message", [
    ("1308", "window exhausted"), ("1113", "balance exhausted"),
    ("", "使用上限"), ("", f"Usage limit; resets at {RESET}"),
    ("", f"Limit will reset at {RESET}"),
    ("insufficient_quota", "exhausted"),
])
def test_quota_classification(code, message):
    exc = SimpleNamespace(status_code=429, body={"error": {
        "code": code, "message": message}})
    assert llm_errors.status_error(exc) == (code or "quota_exhausted", False)


@pytest.mark.parametrize("code,message,reset", [
    ("1308", f"使用上限，限额将在 {RESET} 重置", RESET),
    ("1113", "insufficient balance", None),
    ("", f"usage limit will reset at {RESET}", RESET),
])
def test_quota_journal_stops_without_retry(tmp_path, code, message, reset):
    calls = []

    def complete(messages, tools):
        calls.append(True)
        return {"content": "", "tool_calls": [], "error": {
            "code": code, "message": message, "retryable": True}}

    journal = Journal(tmp_path / "journal.jsonl")
    policy = LLMPolicy("designer", SimpleNamespace(complete=complete),
                       SimpleNamespace(), journal=journal, retry_backoff_s=0)
    policy._complete([], [], 1, "")
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len(calls) == 1
    assert entries[-1]["state"] == "LLM_FAILED"
    assert entries[-1]["decision"]["quota"] is True
    assert entries[-1]["decision"].get("reset_at") == reset


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("quota", [True, False])
@pytest.mark.parametrize("quota_index", [0, 1])
def test_scheduler_drains_running_tasks_and_preserves_order(
        tmp_path, monkeypatch, jobs, quota, quota_index):
    tasks = runner.load_suite("core").tasks[:4]
    started = []
    other_started = Event()
    quota_scored = Event()
    original_score = runner.score_task

    def score(task, summary, journal, **kwargs):
        row = original_score(task, summary, journal, **kwargs)
        if row.get("quota"):
            quota_scored.set()
        return row

    monkeypatch.setattr(runner, "score_task", score)

    def execute(task, directory, *args, **kwargs):
        started.append(task.id)
        if task.id == tasks[quota_index].id:
            if jobs > 1:
                assert other_started.wait(5)
            journal = [{"state": "LLM_FAILED", "decision": {
                "code": "1308" if quota else "authentication", "reset_at": RESET}}]
            (directory / "journal.jsonl").write_text(json.dumps(journal[0]) + "\n")
        elif task.id == tasks[1 - quota_index].id:
            other_started.set()
            if jobs > 1 and quota:
                assert quota_scored.wait(5)
        return runner._score_artifacts(task, directory)

    monkeypatch.setattr(runner, "_run_task", execute)
    monkeypatch.setattr(runner, "_run_process", lambda work, **kwargs: execute(*work, **kwargs))
    out = tmp_path / "eval"
    results = runner.run(Namespace(suite="core", suite_file=None,
                                  tasks=",".join(task.id for task in tasks), policy="scripted",
                                  model=None, roles="designer", jobs=jobs, out=out))
    assert [row["id"] for row in results["rows"]] == [task.id for task in tasks]
    count = max(jobs, quota_index + 1) if quota else len(tasks)
    assert set(started) == {task.id for task in tasks[:count]}
    assert results["aggregate"]["not_run"] == len(tasks) - count
    assert json.loads((out / "results.json").read_text()) == results
    if quota:
        for row in results["rows"][count:]:
            assert row["decision"] == "not_run"
            assert not any(row[key] for key in ("valid", "completed", "correct", "llm_failed"))
            assert "quota" in row["error"] and RESET in row["error"]
        assert "provider quota" in (out / "report.md").read_text()
        assert RESET in (out / "report.md").read_text()


@pytest.mark.parametrize("message,expected", [
    (f"Limit will reset at {RESET}", RESET),
    ("usage limit; reset at 2026-10-14T18:20:12+08:00", "2026-10-14T18:20:12+08:00"),
    ("insufficient balance", None), ("rate limited, try later", None),
])
def test_reset_timestamp(message, expected):
    assert llm_errors.quota_reset(message) == expected


def test_worker_failure_preserves_quota_journal(tmp_path, monkeypatch):
    task = runner.load_suite("core").tasks[0]
    (tmp_path / "journal.jsonl").write_text(json.dumps({
        "state": "LLM_FAILED", "decision": {"code": "1113", "quota": True}}) + "\n")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr="worker failed"))
    row = runner._run_process((task, tmp_path, tmp_path / "public.pem",
                               "scripted", None, "designer"))
    assert row["quota"] and row["quota_code"] == "1113"
    assert row["decision"] == "error" and row["reset_at"] is None
    assert "reset time unknown" in runner._not_run(task, row)["error"]


def test_quota_cli_exits_zero_and_writes_artifacts(tmp_path, monkeypatch):
    from chemflow.agent import cli as agent_cli
    from chemflow.cli import main

    def complete(messages, tools):
        return {"content": "", "tool_calls": [], "error": {
            "code": "1308", "message": f"usage limit will reset at {RESET}",
            "retryable": False}}

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    monkeypatch.setattr(agent_cli, "client_factory", lambda *args:
                        SimpleNamespace(model="offline", complete=complete))
    out = tmp_path / "cli-eval"
    monkeypatch.setattr(sys, "argv", ["chemflow", "eval", "run", "--suite", "core",
                                     "--policy", "glm", "--tasks",
                                     "h2-ground,h2-budget-refusal", "--out", str(out)])
    assert main() in (None, 0)
    results = json.loads((out / "results.json").read_text())
    assert results["aggregate"]["not_run"] == 1
    assert (out / "report.md").exists()
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    failure = next(entry for entry in entries if entry["state"] == "LLM_FAILED")
    assert failure["decision"]["quota"] is True
    assert failure["decision"]["reset_at"] == RESET
    assert not (out / "h2-budget-refusal" / "journal.jsonl").exists()


def test_quota_during_initial_pool_dispatch_starts_no_more_tasks(tmp_path, monkeypatch):
    tasks = runner.load_suite("core").tasks[:4]
    started = []

    class ImmediateExecutor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            future = super().submit(function, *args, **kwargs)
            future.result(timeout=5)
            return future

    def execute(work, **kwargs):
        task = work[0]
        started.append(task.id)
        return runner.score_task(task, None, [{"state": "LLM_FAILED", "decision": {
            "quota": True, "code": "1308", "reset_at": RESET}}])

    monkeypatch.setattr(runner, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(runner, "_run_process", execute)
    results = runner.run(Namespace(suite="core", suite_file=None,
                                  tasks=",".join(task.id for task in tasks), policy="scripted",
                                  model=None, roles="designer", jobs=3, out=tmp_path / "pool"))
    assert started == [tasks[0].id]
    assert results["aggregate"]["not_run"] == 3
