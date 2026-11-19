"""Reviewer regression tests for R113 (written by Claude; implementers must not edit).

Found in the full-suite GLM benchmark (2026-10-13). glm-5.3 called `plan.candidates` on a plan
without a reference node, and the tool raised a raw IndexError (R110 has since made tool
exceptions structured). The exception escaped the task, and the whole evaluation exited with
code 2 after 32 of 55 tasks, writing no results.json.

Contract: a task that raises, in process or in a worker process, becomes a row with
`decision: "error"`, `completed: false`, `correct: false` and `error` (exception type and
message, or the worker's stderr tail). The run continues, results.json and report.md are
written, and the exit status is 0. `aggregate` counts `errors`.
"""
from __future__ import annotations

import subprocess
from argparse import Namespace


def _args(out, jobs):
    return Namespace(suite="core", suite_file=None, tasks="h2-ground,h2-atomization",
                     policy="scripted", model=None, roles="designer", jobs=jobs, out=out)


def test_r113_a_task_that_raises_in_process_becomes_an_error_row(tmp_path, monkeypatch) -> None:
    from chemflow.agent import cli as agent_cli
    from chemflow.eval import runner

    original = agent_cli.run

    def flaky(args):
        if "h2-atomization" in str(args.task):
            raise IndexError("list index out of range")
        return original(args)

    monkeypatch.setattr(agent_cli, "run", flaky)
    results = runner.run(_args(tmp_path / "out", 1))
    rows = {row["id"]: row for row in results["rows"]}
    assert rows["h2-ground"]["correct"]
    failed = rows["h2-atomization"]
    assert failed["decision"] == "error" and not failed["completed"] and not failed["correct"]
    assert "IndexError" in failed["error"]
    assert results["aggregate"]["errors"] == 1
    assert (tmp_path / "out" / "results.json").exists() and (tmp_path / "out" / "report.md").exists()


def test_r113_a_crashed_worker_process_becomes_an_error_row(tmp_path, monkeypatch) -> None:
    from chemflow.eval import runner

    original = subprocess.run

    def crash_one(command, **kwargs):
        if "chemflow.eval._worker" in command and "h2-atomization" in kwargs.get("input", ""):
            return subprocess.CompletedProcess(command, 1, stdout="",
                                               stderr="Traceback ...\nIndexError: boom")
        return original(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", crash_one)
    results = runner.run(_args(tmp_path / "out", 2))
    rows = {row["id"]: row for row in results["rows"]}
    assert rows["h2-ground"]["correct"]
    assert rows["h2-atomization"]["decision"] == "error"
    assert "IndexError" in rows["h2-atomization"]["error"]
