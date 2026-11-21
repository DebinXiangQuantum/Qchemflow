import io
import json
import subprocess
from argparse import Namespace

import pytest

from chemflow.agent import cli
from chemflow.agent.llm import DEFAULT_MAX_TURNS, FakeLLMClient, LLMPolicy
from chemflow.eval import _worker, runner


@pytest.mark.parametrize("max_turns", [0, -1, True, 1.5, None])
def test_invalid_turn_budget_is_rejected_before_output(tmp_path, max_turns):
    with pytest.raises(ValueError, match="--max-turns must be a positive integer"):
        runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="scripted",
                             model=None, roles="designer", out=tmp_path / "out", jobs=1,
                             max_turns=max_turns))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_budget_reaches_every_llm_policy_and_metadata(tmp_path, monkeypatch, explicit):
    constructed = {}

    def create_policy(role, *args, **kwargs):
        agent = LLMPolicy(role, *args, **kwargs)
        constructed[role] = agent.max_turns
        return agent

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test")
    monkeypatch.setattr(cli, "LLMPolicy", create_policy)
    monkeypatch.setattr(cli, "client_factory", lambda *args: FakeLLMClient([
        {"content": "Prose is fine", "tool_calls": []}] * 4))
    args = Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="glm", model=None,
                     roles="planner,modeler,designer,critic,analyst", jobs=1, out=tmp_path / "out")
    if explicit:
        args.max_turns = 7
    results = runner.run(args)
    expected = 7 if explicit else DEFAULT_MAX_TURNS
    assert constructed == dict.fromkeys(args.roles.split(","), expected)
    assert results["metadata"]["max_turns"] == expected
    assert json.loads((args.out / "results.json").read_text())["metadata"]["max_turns"] == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_process_payload_and_worker_forward_budget(tmp_path, monkeypatch, explicit):
    captured = {}

    def run_process(command, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(command, 0, '{"ok":true}', "")

    task = runner.load_suite("core").tasks[0]
    monkeypatch.setattr(runner.subprocess, "run", run_process)
    work = (task, tmp_path, tmp_path / "public.pem", "scripted", None, "designer")
    options = {"max_turns": 7} if explicit else {}
    assert runner._run_process(work, **options) == {"ok": True}
    expected = 7 if explicit else DEFAULT_MAX_TURNS
    assert captured["max_turns"] == expected
    if not explicit:
        captured.pop("max_turns")
    forwarded = {}

    def run_task(*args, **kwargs):
        forwarded.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(_worker, "_run_task", run_task)
    monkeypatch.setattr(_worker.sys, "stdin", io.StringIO(json.dumps(captured)))
    output = io.StringIO()
    monkeypatch.setattr(_worker.sys, "stdout", output)
    _worker.main()
    assert forwarded["max_turns"] == expected
    assert json.loads(output.getvalue()) == {"ok": True}


def test_parallel_dispatch_forwards_budget(tmp_path, monkeypatch):
    seen = []

    def run_process(work, max_turns):
        seen.append(max_turns)
        return runner.score_task(work[0], {}, [])

    monkeypatch.setattr(runner, "_run_process", run_process)
    results = runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground,h2-atomization",
                                   policy="scripted", model=None, roles="designer", jobs=2,
                                   out=tmp_path / "out", max_turns=7))
    assert seen == [7, 7]
    assert results["metadata"]["max_turns"] == 7
