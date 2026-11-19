import io
import json
import subprocess
from argparse import Namespace

import pytest
from test_m3b_llm import policy

from chemflow.agent import cli
from chemflow.agent.llm import DEFAULT_MAX_TURNS, FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.eval import _worker, runner
from chemflow.ir.schema import AcceptanceGate, Program


@pytest.mark.parametrize("mutation", [
    "scratch", "prefix", "references", "targets", "no_candidate", "gates", "unknown", "unbound",
])
def test_submit_reports_base_contract_error_and_remains_recoverable(tmp_path, mutation):
    agent, client, _ = policy(tmp_path, "designer", [])
    base, _ = Orchestrator._base_plan(agent.record, agent.record.spec.target.active)
    base_id = agent.tools.plans.put(base)
    agent.tools.design_base_plan_id = base_id
    trial, _ = Orchestrator._candidate_plan(base, [{"template": "sci"}])
    good_id = agent.tools.plans.put(trial)
    if mutation == "scratch":
        trial = Program(name="scratch")
    elif mutation == "prefix":
        trial.nodes[0].params["changed"] = True
    elif mutation == "references":
        trial.references = []
    elif mutation == "targets":
        trial.targets = trial.targets[len(base.targets):]
    elif mutation == "no_candidate":
        trial = base
    bad_id = agent.tools.plans.put(trial)
    if mutation == "gates":
        agent.tools.plans._plans[bad_id].gates = [AcceptanceGate(
            node="candidate0", kind="abs_error_vs_exact", threshold=0.1, unit="hartree")]
    if mutation == "unknown":
        bad_id = "unknown"
    if mutation == "unbound":
        agent.tools.design_base_plan_id = None
    with pytest.raises(ValueError, match="observation.base_plan_id"):
        agent.tools.call("designer", "design.submit", {"plan_id": bad_id})
    agent.tools.design_base_plan_id = base_id
    if mutation != "unbound":
        client.responses = iter([
            {"content": "", "tool_calls": [{"name": "design.submit",
                                            "arguments": {"plan_id": bad_id}}]},
            {"content": "", "tool_calls": [{"name": "design.submit",
                                            "arguments": {"plan_id": good_id}}]},
        ])
        result = agent.decide({"base_plan_id": base_id})
        assert len(client.messages) == 2
        assert "error" in result["outputs"][0]
        assert result["end_reason"] == "design.submit"
        assert result["outputs"][-1]["result"]["data"]["plan_id"] == good_id
    assert agent.tools.plans.get(base_id) == base


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
