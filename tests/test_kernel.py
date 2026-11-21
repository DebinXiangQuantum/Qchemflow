import pytest

from chemflow import Q
from chemflow.relations import RELATION_TABLE, TargetRelation, compose
from chemflow.types import ConstantLedger, Evidence


def test_units() -> None:
    assert Q(1, "hartree").to("eV").value == pytest.approx(27.211386, rel=1e-6)
    assert Q(1, "angstrom").to("bohr").to("angstrom").value == pytest.approx(1)
    with pytest.raises(ValueError):
        _ = Q(1, "hartree") + Q(1, "angstrom")


def test_relation_table_exhaustive() -> None:
    assert len(RELATION_TABLE) == 25
    for upstream in TargetRelation:
        for downstream in TargetRelation:
            actual = compose(upstream, downstream)
            if TargetRelation.model_change in (upstream, downstream):
                assert actual == TargetRelation.model_change
            elif upstream == TargetRelation.equivalent:
                assert actual == downstream
            elif downstream == TargetRelation.equivalent:
                assert actual == upstream
            elif upstream == downstream == TargetRelation.variational_upper_bound:
                assert actual == TargetRelation.variational_upper_bound
            elif upstream == downstream == TargetRelation.stochastic_estimate:
                assert actual == TargetRelation.stochastic_estimate
            else:
                assert actual == TargetRelation.approximate


def test_evidence_and_constants() -> None:
    with pytest.raises(ValueError, match="CF-E030"):
        ConstantLedger().add("nuclear_repulsion", Q(1, "hartree")).add(
            "nuclear_repulsion", Q(2, "hartree"))
    with pytest.raises(ValueError, match="CF-E040"):
        Evidence(status="certified_bound")
    with pytest.raises(ValueError, match="CF-E040"):
        Evidence._runtime(status="certified_bound", artifact_ref="invented")
