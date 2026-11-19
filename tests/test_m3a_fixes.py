import pytest

from chemflow.agent.critic import review
from chemflow.agent.policies import ScriptedPolicy
from chemflow.agent.report import ReportLintError, lint_report, write_report
from chemflow.agent.task import TaskSpec


def test_exact_reference_cannot_dominate_candidate():
    candidate = {"id": "sqd", "relation": "variational_upper_bound", "evidence": True,
                 "gate_passed": True, "cost": {"wall_s": 10}}
    reference = {"id": "fci", "relation": "equivalent", "evidence": True,
                 "gate_passed": True, "cost": {"wall_s": 1}, "error": 0}
    assert review([candidate], [reference]).decision == "escalate"


def test_auto_sector_cannot_pin_orbitals():
    with pytest.raises(ValueError, match="cannot fix sector"):
        TaskSpec.model_validate({"question": "auto", "target": {
            "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"}, "active": "auto",
            "sector": {"n_orb": 2}}, "gates": [], "decision": "screen"})


def test_revision_changes_hyperparameters_or_exhausts():
    cards = [{"template": "sci", "target": "ground_energy"},
             {"template": "sqd", "target": "ground_energy"}]
    designer = ScriptedPolicy("designer")
    first = designer.decide({"cards": cards})["candidates"]
    second = designer.decide({"cards": cards, "attempted": first})["candidates"]
    assert second and all(candidate not in first for candidate in second)
    assert designer.decide({"cards": cards, "attempted": first + second}) == {
        "candidates": []}


def test_report_claims_are_scoped_to_generated_headline():
    headline = {"node": "ref", "value": "-1 hartree", "relation": "equivalent",
                "evidence": "computed", "gate_verdict": "PASS", "cost": {},
                "backend_kind": "cpu_numeric", "screening_reference": True}
    report = write_report("screen", "stop", "use classical", [headline])
    assert "exact reference available only at screening scale" in report
    with pytest.raises(ReportLintError, match="exact"):
        lint_report(report + "The SQD result is exact.\n", [headline])
    with pytest.raises(ReportLintError, match="exact"):
        write_report("screen", "stop", "exact", [{**headline, "relation": "approximate"}])
