"""Reviewer regression tests for the critic's answer (written by Claude; implementers must not
edit).

Found by the G5 harness (2026-10-13): for reaction, interaction, thermochemistry and spectral
tasks the scripted agent stopped with "no new candidates" and recommended nothing, although
gate-passing classical controls had been computed. ChemFlow's rule is that a classical control
meeting the gate is a valid answer. With no candidates, `chemflow.agent.critic.review` returns
`stop` with `candidate` set to the gate-passing control of smallest error. Errors equal within
1e-9 Ha are broken by the method ladder hf < mp2 < ccsd < ccsd_t < fci (each control summary
carries `method`), then by id. Measured run time is never used: it is noise at these sizes and
made the choice nondeterministic. The exact screening reference is chosen only if no other
control passes. With nothing passing there
is no answer. The orchestrator must use this instead of its own "no new candidates" stop.
"""
from __future__ import annotations

from chemflow.agent.critic import review


def _control(name, passed, error, wall, relation="approximate"):
    return {"id": f"node-{name}", "method": name, "gate_passed": passed, "evidence": True,
            "relation": relation, "error": error, "cost": {"wall_s": wall}}


def test_without_candidates_the_best_passing_control_is_the_answer() -> None:
    controls = [_control("hf", False, 0.02, 0.001, "variational_upper_bound"),
                _control("mp2", True, 5e-4, 0.01), _control("ccsd_t", True, 1e-7, 0.01),
                _control("ccsd", True, 1e-7 + 1e-12, 0.03)]
    decision = review([], controls)
    # equal errors: the ladder decides, not the (noisy) measured time
    assert decision.decision == "stop" and decision.candidate == "node-ccsd"


def test_the_exact_reference_is_the_answer_only_when_nothing_else_passes() -> None:
    controls = [_control("hf", False, 0.02, 0.001, "variational_upper_bound"),
                _control("fci", True, 0.0, 0.05, "equivalent")]
    assert review([], controls).candidate == "node-fci"
    controls.append(_control("ccsd", True, 1e-6, 0.02))
    assert review([], controls).candidate == "node-ccsd"


def test_with_nothing_passing_there_is_no_answer() -> None:
    decision = review([], [_control("hf", False, 0.02, 0.001, "variational_upper_bound")])
    assert decision.candidate is None and decision.decision != "stop"
