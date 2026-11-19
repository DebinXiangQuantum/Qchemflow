import json
from itertools import permutations

import pytest
from test_m3a_task import spec
from test_m3b_llm import policy

from chemflow.agent import orchestrator
from chemflow.agent.critic import CriticDecision, review
from chemflow.agent.llm import LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore
from chemflow.diagnostics import CheckError, Diagnostic


def control(identifier, **updates):
    return {"id": identifier, "method": "ccsd", "relation": "approximate",
            "error": 1e-5, "gate_passed": True, "evidence": False,
            "cost": {"wall_s": 1}, **updates}


@pytest.mark.parametrize("controls,expected", [
    ([control("reference", relation="equivalent", error=0), control("classical")],
     "classical"),
    ([control("weaker", method="hf", error=1e-4), control("stronger")], "stronger"),
    ([control("first", method="ccsd_t"), control("last", method="hf")], "last"),
    ([control("last"), control("first")], "first"),
    ([control("reference", relation="equivalent")], "reference"),
    ([control("missing", error=None), control("known")], "known"),
])
def test_unsupported_fallback_uses_no_candidate_ordering(controls, expected):
    for ordering in permutations(controls):
        verdict = review([control("unsupported", error=0)], list(ordering))
        assert verdict == review([], list(ordering))
        assert verdict.candidate == expected
        assert verdict.reason.startswith("use classical: ")


def test_dominance_keeps_the_dominating_control_and_its_reason():
    candidate = control("candidate", evidence=True)
    affordable = control("cheap", evidence=True, error=1e-4)
    strongest = control("strongest", cost={"wall_s": 10})
    verdict = review([candidate], [affordable, strongest])
    assert verdict.candidate == "cheap"
    assert verdict.reason == "use classical: cheap meets gates at no greater decision cost"


def test_no_passing_control_and_escalation_remain_unchanged():
    failed = control("failed", gate_passed=False)
    assert review([control("unsupported")], [failed]) == CriticDecision(
        "stop", "No supported candidate within budget")
    assert review([failed], [control("passing")]).decision == "revise"
    supported = control("supported", evidence=True)
    verdict = review([supported], [control("passing", cost={"wall_s": 10})])
    assert verdict.decision == "escalate" and verdict.candidate == "supported"


class Designer:
    def complete(self, messages, tools):
        results = [json.loads(message["content"]) for message in messages
                   if message["role"] == "tool"]
        if results:
            name = "design.submit"
            arguments = {"plan_id": results[-1]["result"]["data"]["plan_id"]}
        else:
            name = "plan.candidates"
            base = json.loads(messages[1]["content"])["observation"]["base_plan_id"]
            arguments = {"plan_id": base, "candidates": [{"template": "sci"}]}
        return {"content": "", "tool_calls": [{"name": name, "arguments": arguments}]}


@pytest.mark.parametrize("llm", [False, True])
@pytest.mark.parametrize("passing", [False, True])
def test_final_revise_preserves_headlines_and_no_control_behavior(tmp_path, monkeypatch,
                                                                llm, passing):
    original_review = review

    def revise(candidates, controls):
        if candidates:
            return CriticDecision("revise", "try another candidate")
        return original_review(candidates, controls)

    monkeypatch.setattr(orchestrator, "review", revise)
    monkeypatch.setattr("chemflow.agent.critic.review", revise)
    if not passing:
        original_item = Orchestrator._item
        monkeypatch.setattr(Orchestrator, "_item", staticmethod(
            lambda *args: original_item(*args) | {"gate_passed": False}))
    store = TaskStore("offline")
    policies = {"designer": LLMPolicy("designer", Designer(), None)} if llm else None
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=1, policies=policies)
    result = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    assert result.decision.decision == "stop"
    if passing:
        assert result.decision.reason.startswith("use classical: ")
        assert any(item["node"] == result.decision.candidate for item in result.results)
        assert len({item["node"] for item in result.results}) == len(result.results)
    else:
        assert result.decision == CriticDecision("stop", "Maximum screening rounds reached")


def test_zero_screening_rounds_still_reports_control(tmp_path):
    store = TaskStore("offline")
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=0)
    result = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    assert result.decision.reason.startswith("use classical: ")
    assert any(item["node"] == result.decision.candidate for item in result.results)


@pytest.mark.parametrize("severity", ["warning", "error"])
def test_submit_static_check_filters_severity(tmp_path, monkeypatch, severity):
    agent, _, _ = policy(tmp_path, "designer", [])
    base, _ = Orchestrator._base_plan(agent.record, agent.record.spec.target.active)
    base_id = agent.tools.plans.put(base)
    agent.tools.design_base_plan_id = base_id
    built = agent.tools.call("designer", "plan.candidates", {
        "plan_id": base_id, "candidates": [{"template": "sci"}]})["data"]
    issue = Diagnostic(code="CF-E999", severity=severity, message="test static diagnostic")
    checked = []

    def check(plan, registry, purpose="screen"):
        assert purpose == "deploy"
        checked.append(plan.hash)
        return [Diagnostic(code="CF-W999", severity="warning", message="advisory"), issue]

    monkeypatch.setattr("chemflow.agent.tools.check", check)
    if severity == "error":
        with pytest.raises(CheckError) as caught:
            agent.tools.call("designer", "design.submit", {"plan_id": built["plan_id"]})
        assert caught.value.diagnostics == [issue]
    else:
        result = agent.tools.call("designer", "design.submit", {"plan_id": built["plan_id"]})
        assert result["data"] == {"plan_id": built["plan_id"]}
    assert checked == [agent.tools.plans.get(built["plan_id"]).hash]
