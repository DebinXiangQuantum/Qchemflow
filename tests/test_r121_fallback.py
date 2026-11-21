import json
from itertools import permutations

import pytest

from chemflow.agent.critic import CriticDecision, review


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
