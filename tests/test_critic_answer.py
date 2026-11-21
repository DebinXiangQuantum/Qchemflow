from __future__ import annotations

import json

import pytest
from test_g5_evaluation import arguments

from chemflow.agent.critic import review
from chemflow.agent.policies import ScriptedPolicy
from chemflow.eval.runner import run


def control(identifier, **updates):
    return {"id": identifier, "gate_passed": True, "evidence": True,
            "relation": "approximate", "error": 0.0001, "cost": {"wall_s": 1}, **updates}


def test_no_passing_control_means_revise():
    decision = review([], [control("invalid", gate_passed=False)])
    assert decision.decision == "revise" and decision.candidate is None
    assert review([], []).decision == "revise"


def test_classical_gate_pass_does_not_require_candidate_relation_evidence():
    decision = review([], [control("classical", evidence=False)])
    assert decision.decision == "stop" and decision.candidate == "classical"


def test_error_precedes_id_and_missing_error_loses():
    best = control("best", error=0.00001, cost={"wall_s": 10})
    cheap = control("cheap", cost={"wall_s": 0.001})
    unknown = control("unknown", error=0.00001, cost={"wall_s": None})
    reference = control("reference", error=0, relation="equivalent")
    assert review([], [reference, cheap, unknown, best]).candidate == "best"
    assert review([], [reference]).candidate == "reference"
    assert review([], [control("missing-error", error=None), best]).candidate == "best"


def test_candidate_present_rules_are_unchanged():
    classical = control("classical")
    assert review([control("candidate", gate_passed=False)], [classical]).decision == "revise"
    assert review([control("candidate")], [classical]).candidate == "classical"
    assert review([control("candidate")], []).decision == "escalate"


@pytest.mark.parametrize("task_id", [
    "h2-ground", "h2-atomization", "hubbard-ring6", "impurity-spectral",
    "h2-counterpoise", "hcn-hnc-enthalpy",
])
def test_no_candidate_answers_are_reported_for_every_core_kind(tmp_path, monkeypatch, task_id):
    original = ScriptedPolicy.decide

    def decide(self, observation):
        if self.role == "designer":
            return {"candidates": []}
        return original(self, observation)

    monkeypatch.setattr(ScriptedPolicy, "decide", decide)
    result = run(arguments(tmp_path / "out", tasks=task_id))
    row = result["rows"][0]
    directory = tmp_path / "out" / task_id
    summary = json.loads((directory / "summary.json").read_text())
    journal = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    critique = next(entry for entry in journal if entry["state"] == "CRITIQUE")
    expected = review([], critique["observation"]["controls"])
    assert row["completed"] and row["gate_passed"] and row["decision"] == "stop"
    assert summary["candidate"] == row["answer_node"] == expected.candidate
    assert f"use classical: {expected.candidate}" in (directory / "report.md").read_text()
    assert any(item["node"] == expected.candidate for item in summary["results"])
