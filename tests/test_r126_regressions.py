from __future__ import annotations

import json
from argparse import Namespace

import pytest

from chemflow.agent.journal import Journal
from chemflow.agent.llm import TERMINAL_TOOLS, FakeLLMClient, LLMPolicy
from chemflow.agent.task import TaskStore
from chemflow.agent.tools import ToolRegistry, make_tools
from chemflow.diagnostics import AdapterError
from chemflow.eval.runner import EvaluationTask, load_suite, score_task
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor
from chemflow.runtime.budget import BudgetLedger
from chemflow.types import CostRecord


def setup_plan(directory=None, shots=3072):
    task = next(task for task in load_suite("full").tasks if task.id == "refuse-sim-shots")
    store = TaskStore("regression")
    record = store.freeze(task.spec, "regression")
    registry = molecular_registry()
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store, plan_directory=directory)
    seed = tools.call("modeler", "plan.new", {"identifier": "base"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {
        "plan_id": seed, "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]
    tools.call("designer", "run", {"plan_id": base, "scale": "screen"})
    built = tools.call("designer", "plan.candidates", {
        "plan_id": base, "candidates": [{"template": "sqd", "shots": shots}]
    })["data"]["plan_id"]
    return tools, executor, record, store, built


@pytest.mark.parametrize("persisted", [False, True])
def test_verified_decline_preserves_cache_and_ledger(tmp_path, persisted):
    tools, executor, record, store, plan_id = setup_plan(tmp_path if persisted else None)
    cache = dict(executor.cache)
    cache_hits = executor.cache_hits
    ledger_path = tmp_path / f"{record.content_hash}.ledger.json"
    ledger_before = ledger_path.read_bytes() if persisted else None
    result = tools.call("designer", "design.decline", {
        "reason": "Batch exceeds shots", "plan_id": plan_id})["data"]
    assert result["plan_hash"] == tools.plans.get(plan_id).hash
    assert result["refusal"]["code"] == "CF-E070"
    assert result["refusal"]["nothing_consumed"] is True
    assert executor.cache == cache
    assert executor.cache_hits == cache_hits
    if persisted:
        assert ledger_path.read_bytes() == ledger_before
        assert result["refusal"]["spent_at_refusal"] == json.loads(ledger_before)
    with pytest.raises(AdapterError) as caught:
        executor.run_plan(tools.plans.get(plan_id), record, store,
                          ledger=BudgetLedger(record.spec.budget, ledger_path if persisted else None))
    assert caught.value.diagnostics[0].node == result["refusal"]["node"]
    assert caught.value.nothing_consumed


def test_verified_decline_uses_reloaded_remaining_budget(tmp_path):
    tools, executor, record, _store, plan_id = setup_plan(tmp_path, shots=1024)
    ledger_path = tmp_path / f"{record.content_hash}.ledger.json"
    before = ledger_path.read_bytes()
    hits = executor.cache_hits
    with pytest.raises(ValueError, match="fits the remaining task budget"):
        tools.call("designer", "design.decline", {"reason": "Claim", "plan_id": plan_id})
    assert ledger_path.read_bytes() == before
    assert executor.cache_hits == hits
    ledger = BudgetLedger(record.spec.budget, ledger_path)
    ledger.record(CostRecord(sim_shots=1500))
    result = tools.call("designer", "design.decline", {
        "reason": "Remaining shots are insufficient", "plan_id": plan_id})["data"]
    assert result["refusal"]["spent_at_refusal"]["sim_shots"] == 1500
    assert ledger.path.read_text() == json.dumps(ledger.spent)


def test_unknown_plan_is_not_a_verified_decline(tmp_path):
    tools = setup_plan(tmp_path)[0]
    with pytest.raises((KeyError, ValueError)):
        tools.call("designer", "design.decline", {"reason": "Claim", "plan_id": "missing"})
    assert tools.call("designer", "design.decline", {"reason": "Controls suffice"}) == {
        "data": {"reason": "Controls suffice"}}


def _run_batch_designer(tmp_path, monkeypatch, terminal):
    from chemflow.agent import cli
    from chemflow.eval.runner import run

    class Designer:
        def complete(self, messages, tools):
            built = [json.loads(message["content"])["result"]["data"]
                     for message in messages if message["role"] == "tool"]
            if built:
                name = terminal
                arguments = {"plan_id": built[-1]["plan_id"]}
                if terminal == "design.decline":
                    arguments["reason"] = "Batch exceeds remaining shots"
            else:
                name = "plan.candidates"
                base = json.loads(messages[1]["content"])["observation"]["base_plan_id"]
                arguments = {"plan_id": base, "candidates": [
                    {"template": "sqd", "shots": shots} for shots in (1024, 2048)]}
            return {"content": "", "tool_calls": [{"name": name, "arguments": arguments}]}

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Designer())
    result = run(Namespace(suite="full", suite_file=None, tasks="refuse-sim-shots",
                           policy="glm", model=None, roles="designer", jobs=1, out=tmp_path))
    return result


def test_verified_decline_budget_refusal_reaches_cli(tmp_path, monkeypatch):
    result = _run_batch_designer(tmp_path, monkeypatch, "design.decline")
    assert result["rows"][0]["correct"]
    summary = json.loads((tmp_path / "refuse-sim-shots" / "summary.json").read_text())
    assert summary["decision"] == "refused"
    assert summary["refusal"]["code"] == "CF-E070"
    assert summary["refusal"]["nothing_consumed"] is True
    assert summary["refusal"]["spent_at_refusal"] == summary["spent"]
    assert summary["spent"]["sim_shots"] == 0
    entries = [json.loads(line) for line in
               (tmp_path / "refuse-sim-shots" / "journal.jsonl").read_text().splitlines()]
    assert not any(entry["state"] == "SCREEN" for entry in entries)
    declined = next(entry for entry in entries if entry["state"] == "DESIGN_REFUSED")
    assert len(declined["observation"]["plan_hash"]) == 64


def test_submitting_an_unaffordable_plan_refuses_the_design_not_the_task(tmp_path, monkeypatch):
    # Reviewer decision: R115 holds for submissions. A runtime budget refusal of a submitted
    # plan is a design refusal and the controls answer; only a verified decline makes the
    # run a budget refusal.
    result = _run_batch_designer(tmp_path, monkeypatch, "design.submit")
    row = result["rows"][0]
    assert row["decision"] != "refused" and not row["correct"]
    entries = [json.loads(line) for line in
               (tmp_path / "refuse-sim-shots" / "journal.jsonl").read_text().splitlines()]
    refused = [entry for entry in entries if entry["state"] == "DESIGN_REFUSED"]
    assert refused and "CF-E070" in refused[0]["reason"]



@pytest.mark.parametrize("relation,expected", [
    ("equivalent", True), ("variational_upper_bound", False),
    ("stochastic_estimate", False), ("approximate", False), ("unknown", False),
])
def test_relation_scoring_requires_every_signed_relation_gate(relation, expected):
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"].append({"kind": "relation_at_least", "relation": "stochastic_estimate"})
    task = EvaluationTask.model_validate(data)
    summary = {"candidate": "answer", "results": [{"node": "answer", "relation": relation}]}
    assert score_task(task, summary, [])["relation_ok"] is expected


def test_relation_scoring_without_answer_or_minimum():
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"] = [gate for gate in data["spec"]["gates"]
                             if gate["kind"] != "relation_at_least"]
    task = EvaluationTask.model_validate(data)
    summary = {"candidate": "missing", "results": []}
    assert score_task(task, summary, [])["relation_ok"] is None
    summary["results"] = [{"node": "missing", "relation": "model_change"}]
    assert score_task(task, summary, [])["relation_ok"] is True


@pytest.mark.parametrize("role", ["designer", "modeler", "critic", "planner", "analyst"])
@pytest.mark.parametrize("max_turns", [1, 2, 5])
def test_terminal_reminder_counts_include_current_call(tmp_path, role, max_turns):
    tools = ToolRegistry()
    tools.register("cards.search", {role}, lambda data: {})
    client = FakeLLMClient([{"content": "", "tool_calls": [
        {"name": "cards.search", "arguments": {}}]} for _ in range(max_turns)])
    LLMPolicy(role, client, tools, journal=Journal(tmp_path / "journal.jsonl"),
              max_turns=max_turns).decide({})
    for turn, messages in enumerate(client.messages, start=1):
        remaining = max_turns - turn + 1
        reminders = [message["content"] for message in messages
                     if message["role"] == "user" and "turn" in message["content"].lower()]
        if role not in TERMINAL_TOOLS or remaining > 2:
            assert reminders == []
        else:
            assert f"{remaining} turn" in reminders[-1]
            assert "including this one" in reminders[-1]
            assert all(tool in reminders[-1] for tool in TERMINAL_TOOLS[role])
            assert len(reminders) == min(2, max_turns) - remaining + 1
