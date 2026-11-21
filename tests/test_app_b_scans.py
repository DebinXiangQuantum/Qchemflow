from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemflow.agent.task import PESScanTarget, TaskSpec
from chemflow.eval.runner import EvaluationTask, score_task


def target():
    return PESScanTarget(kind="pes_scan", molecule={"atoms": "H 0 0 0; H 0 0 .74"},
                         electronic_model={"basis": "sto-3g"},
                         coordinate={"type": "bond", "atoms": [0, 1], "values": [.7, 1.2]})


def record(scan=None):
    return SimpleNamespace(spec=TaskSpec(question="Scan", target=scan or target(),
        gates=[{"kind": "abs_error_vs_exact", "threshold": .0016, "unit": "hartree"},
               {"kind": "relation_at_least", "relation": "approximate"}],
        budget={"wall_s": 1800}, allowed_effects=["pure"], decision="Choose a control"))


def test_curve_scoring_requires_every_point_and_matching_length():
    task = EvaluationTask(id="curve", category="path", spec=record().spec,
        expected={"values_hartree": [-1., -2.]}, reference_source="Independent fixture")

    def scored(values):
        return score_task(task, {"candidate": "answer", "decision": "stop", "results": [
            {"node": "answer", "values_hartree": values, "relation": "approximate",
             "gate_verdict": "PASS"}]}, [])

    assert scored([-1., -2.])["correct"]
    assert not scored([-1., -1.9])["correct"]
    assert not scored([-1.])["correct"]
    assert not scored([-1., float("nan")])["completed"]
    with pytest.raises(ValidationError):
        EvaluationTask(id="curve", category="path", spec=record().spec,
            expected={"values_hartree": [float("inf")]}, reference_source="Fixture")
