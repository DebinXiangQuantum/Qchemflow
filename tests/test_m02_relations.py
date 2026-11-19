import pytest

from chemflow import build
from chemflow.check.static import infer_relations
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation as R
from chemflow.relations import combine_difference


@pytest.mark.parametrize("left,right,expected", [
    (left, right, expected)
    for left, row in zip(R, (
        (R.equivalent, R.approximate, R.stochastic_estimate, R.approximate, R.model_change),
        (R.variational_upper_bound, R.approximate, R.approximate, R.approximate,
         R.model_change),
        (R.stochastic_estimate, R.approximate, R.stochastic_estimate, R.approximate,
         R.model_change),
        (R.approximate, R.approximate, R.approximate, R.approximate, R.model_change),
        (R.model_change, R.model_change, R.model_change, R.model_change, R.model_change),
    ))
    for right, expected in zip(R, row)
])
def test_difference_relation_table(left: R, right: R, expected: R) -> None:
    assert combine_difference(left, right) == expected


def test_static_difference_matches_runtime_relation() -> None:
    with build.program("static_difference") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"), model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        upper = builder.ground_energy(hamiltonian, sector)
        builder.refine(upper, "hf")
        builder.bind(upper, "hf")
        exact = builder.ground_energy(hamiltonian, sector)
        difference = builder.node("energy_difference", left=exact, right=upper)
    assert infer_relations(builder.to_ir(), molecular_registry())[difference.id] == R.approximate
