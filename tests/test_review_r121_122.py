"""Reviewer regression tests for R121-R122 (written by Claude; implementers must not edit).

Found by the post-fix glm-5.3 designer benchmark (2026-10-14, commit 7b9fb46, full suite).
GLM's candidates were screened in 45 of 55 tasks, against 6 before. Two new failure modes
appeared:
- R121: h2-dimer-collinear and h2-dimer-parallel ended with no answer: decision "stop",
  "No supported candidate within budget", candidate None. GLM's composed interaction-energy
  candidate passed the gate but had no evidence, so `review` rejected it. Its last branch then
  returned no recommendation, although every classical control passed the 0.5 mHa gate. The same
  gap exists at the end of the screening rounds: "Maximum screening rounds reached" is a stop
  with no candidate, in both the scripted loop and the LLM path. In both cases the run answered
  nothing when a gate-passing control existed.
- R122: impurity-asymmetric was refused with "Candidate plan failed static check". The
  designer submitted a plan that `check` rejects, but nothing told the designer before its turn
  ended.

Contract:
- R121: a run whose final decision is not escalate, and that has a gate-passing control,
  answers with the strongest gate-passing control. It uses the same rule as the no-candidate
  path: "use classical: ...", lowest error, ladder then id ties, and non-equivalent controls
  first. This covers `review`'s unsupported-candidate branch and the end of the screening
  rounds after a revise, in both the scripted loop and the LLM design path. With no
  gate-passing control, the outcome stays as before.
- R122: `design.submit` runs the static check on the submitted plan. Error diagnostics are
  returned to the model as a structured error that includes their codes, and the turn
  continues. The orchestrator's own check stays in place.
"""
from __future__ import annotations

import sys

from chemflow.agent.critic import CriticDecision, review

PASSING_CONTROL = {"id": "n6", "method": "ccsd_t", "relation": "approximate", "error": 1e-5,
                   "gate_passed": True, "evidence": False,
                   "cost": {"wall_s": 0.1, "cpu_core_s": 0.1}}


def test_r121_an_unsupported_candidate_leaves_the_best_control_as_the_answer() -> None:
    unsupported = {"id": "c3", "method": "ccsd_t", "relation": "approximate", "error": 1e-6,
                   "gate_passed": True, "evidence": False,
                   "cost": {"wall_s": 0.0, "cpu_core_s": 0.0}}
    weaker = PASSING_CONTROL | {"id": "n3", "method": "hf", "error": 1e-4}
    verdict = review([unsupported], [weaker, PASSING_CONTROL])
    assert verdict.decision == "stop" and verdict.candidate == "n6"
    assert verdict.reason.startswith("use classical")


def _revise_candidates(monkeypatch):
    from chemflow.agent import critic

    original = critic.review

    def revise(candidates, controls):
        if candidates:
            return CriticDecision("revise", "No candidate passed all gates")
        return original(candidates, controls)

    for module in list(sys.modules.values()):
        if (getattr(module, "__name__", "").startswith("chemflow") and
                getattr(module, "review", None) is original):
            monkeypatch.setattr(module, "review", revise)


def _answered_by_a_control(row):
    assert row["decision"] == "stop" and row["completed"] and row["correct"], row
    assert row["answer_node"] and not row["answer_from_candidate"], row
