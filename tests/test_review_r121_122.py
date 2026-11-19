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

import json
import sys

from test_review_r114_118 import (
    _base,
    _designer_calls_before_screening,
    _eval,
    _results,
    _turn,
)

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


def test_r121_the_scripted_loop_answers_after_its_last_round(tmp_path, monkeypatch) -> None:
    from test_m3a_task import spec

    from chemflow.agent.orchestrator import Orchestrator
    from chemflow.agent.task import TaskStore

    _revise_candidates(monkeypatch)
    store = TaskStore("reviewer")
    agent = Orchestrator(store, lambda task: store.freeze(task, "reviewer"), tmp_path,
                         max_rounds=1)  # the designer still has candidates when rounds run out
    run = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    assert run.decision.decision == "stop" and run.decision.candidate, vars(run.decision)
    assert not run.decision.candidate.startswith("candidate")
    assert run.decision.reason.startswith("use classical")


def test_r121_the_llm_design_path_answers_after_a_revise(tmp_path, monkeypatch) -> None:
    _revise_candidates(monkeypatch)

    def script(call, messages):
        if call == 1:
            return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        built = next(data for name, data in _results(messages) if name == "plan.candidates")
        return _turn([("design.submit", {"plan_id": built["plan_id"]})])

    _, rows, _, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    assert rows["h2-ground"]["design"] == "screened"
    _answered_by_a_control(rows["h2-ground"])


def test_r122_submit_returns_static_check_errors_and_the_turn_continues(tmp_path,
                                                                        monkeypatch) -> None:
    broken = {"id": "broken0", "op": "chemflow.ground_energy", "inputs": [], "params": [],
              "effects": "pure", "refinement": None, "binding": None}

    def script(call, messages):
        base = _base(messages)
        if call == 1:
            return _turn([("plan.add_node", {"plan_id": base, "node": broken, "target": True})])
        if call == 2:
            added = [data for name, data in _results(messages) if name == "plan.add_node"][-1]
            assert added["diagnostics"], "the broken node must fail the static check"
            return _turn([("design.submit", {"plan_id": added["plan_id"]})])
        if call == 3:
            return _turn([("plan.candidates", {"plan_id": base, "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        if call == 4:
            built = [data for name, data in _results(messages) if name == "plan.candidates"][-1]
            return _turn([("design.submit", {"plan_id": built["plan_id"]})])
        return _turn(content="should never be asked")

    _, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    entries = journal("h2-ground")
    assert _designer_calls_before_screening(entries) == 4
    submits = [entry["decision"] for entry in entries if entry["state"] == "LLM_TOOL" and
               entry["decision"].get("name") == "design.submit"]
    assert "error" in submits[0] and "CF-E" in json.dumps(submits[0]["error"])
    assert "result" in submits[1]
    assert rows["h2-ground"]["design"] == "screened"
