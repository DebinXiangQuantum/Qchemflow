import hashlib
import json
import subprocess
import sys

import pytest

from chemflow import cli
from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.eval import runner
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


@pytest.fixture
def task():
    return runner.load_suite("core").tasks[0]


def save_artifacts(directory, task):
    summary = {"decision": "stop", "candidate": "answer", "spent": {
        resource: 7 for resource in runner.RESOURCES}, "results": [{
            "node": "answer", "value": f"{task.expected.value_hartree} hartree",
            "gate_verdict": "PASS", "relation": "equivalent"}]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "journal.jsonl").write_text('\n'.join(json.dumps(entry) for entry in [
        {"state": "LLM_CALL", "tool_calls": [{}, {}]}, {"state": "LLM_ERROR"}]))


@pytest.mark.parametrize("worker", [False, True])
def test_crash_preserves_artifacts_and_diagnostics(tmp_path, monkeypatch, task, worker):
    save_artifacts(tmp_path, task)
    originals = {path: hashlib.sha256(path.read_bytes()).digest()
                 for path in tmp_path.iterdir()}

    def crash(*args, **kwargs):
        if worker:
            return subprocess.CompletedProcess(args[0], 1, "", "Traceback\nIndexError: boom")
        raise IndexError("boom")

    work = (task, tmp_path, tmp_path / "public.pem", "scripted", None, "designer")
    if worker:
        monkeypatch.setattr(runner.subprocess, "run", crash)
        row = runner._run_process(work)
    else:
        monkeypatch.setattr(runner.agent_cli, "run", crash)
        row = runner._run_task(*work)
    assert row["decision"] == "error"
    assert not row["completed"] and not row["correct"]
    assert row["decision_ok"] is not True
    assert "IndexError: boom" in row["error"]
    assert row["llm_calls"] == 2 and row["tool_calls"] == 2
    assert all(row[resource] == 7 for resource in runner.RESOURCES)
    assert all(hashlib.sha256(path.read_bytes()).digest() == digest
               for path, digest in originals.items())
    assert runner.aggregate_rows([row])["errors"] == 1


@pytest.mark.parametrize("worker", [False, True])
def test_error_text_is_bounded(tmp_path, monkeypatch, task, worker):
    def crash(*args, **kwargs):
        if worker:
            return subprocess.CompletedProcess(args[0], 1, "", "x" * 5000 + "\nIndexError: end")
        raise RuntimeError("start" + "x" * 5000)

    work = (task, tmp_path, tmp_path / "public.pem", "scripted", None, "designer")
    if worker:
        monkeypatch.setattr(runner.subprocess, "run", crash)
        row = runner._run_process(work)
        assert row["error"].endswith("IndexError: end")
    else:
        monkeypatch.setattr(runner.agent_cli, "run", crash)
        row = runner._run_task(*work)
        assert row["error"].startswith("RuntimeError: start")
    assert len(row["error"]) <= runner.ERROR_LIMIT


@pytest.mark.parametrize("error", [SystemExit(2), SystemExit(1), KeyboardInterrupt()])
def test_control_flow_exceptions_keep_existing_behavior(tmp_path, monkeypatch, task, error):
    def crash(args):
        raise error

    monkeypatch.setattr(runner.agent_cli, "run", crash)
    work = (task, tmp_path, tmp_path / "public.pem", "scripted", None, "designer")
    if isinstance(error, SystemExit) and error.code == 2:
        assert runner._run_task(*work)["decision"] == "failed"
    else:
        with pytest.raises(type(error)) as caught:
            runner._run_task(*work)
        assert caught.value is error


def test_failed_worker_without_stderr(tmp_path, monkeypatch, task):
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, -9, "", ""))
    row = runner._run_process((task, tmp_path, tmp_path / "public.pem",
                               "scripted", None, "designer"))
    assert row["decision"] == "error" and "-9" in row["error"]


def test_cli_completes_after_first_task_crashes(tmp_path, monkeypatch):
    original = runner.agent_cli.run

    def crash_first(args):
        if args.task.parent.name == "h2-ground":
            raise IndexError("first task crashed")
        return original(args)

    monkeypatch.setattr(runner.agent_cli, "run", crash_first)
    monkeypatch.setattr(sys, "argv", ["chemflow", "eval", "run", "--suite", "core",
                                     "--policy", "scripted", "--tasks",
                                     "h2-ground,h2-atomization", "--out", str(tmp_path / "out")])
    assert cli.main() is None
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert results["aggregate"]["errors"] == 1
    assert results["rows"][1]["correct"]
    report = (tmp_path / "out" / "report.md").read_text()
    assert "### h2-ground" in report and "IndexError: first task crashed" in report


def test_candidates_without_reference_is_structured(task):
    store = TaskStore("test-secret")
    record = store.freeze(task.spec, "human")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    plan = tools.call("designer", "plan.new", {})["data"]["plan_id"]
    with pytest.raises(ValueError, match="plan has no reference node"):
        tools.call("designer", "plan.candidates", {
            "plan_id": plan, "candidates": [{"template": "sci", "select_cutoff": 0.01}]})
