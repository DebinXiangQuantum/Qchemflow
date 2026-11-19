import pytest

from chemflow import build
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation
from chemflow.runtime import Executor, freeze_task
from chemflow.runtime.executor import _check_relation_consistency
from chemflow.units import Q


def _estimates():
    with build.program("consistency") as builder:
        ham = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                  builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        exact = builder.ground_energy(ham, sector)
        builder.refine(exact, "fci")
        builder.bind(exact, "fci_pyscf")
        other = builder.ground_energy(ham, sector)
        builder.refine(other, "hf")
        builder.bind(other, "hf")
    program = builder.to_ir()
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    return exact.id, other.id, results[exact.id], results[other.id], {
        node.id: node for node in program.nodes}


@pytest.mark.parametrize("offset,violates", [(1e-7 - 1e-9, False),
                                            (1e-7 + 1e-9, True)])
def test_upper_bound_tolerance(offset, violates):
    exact_id, upper_id, exact, upper, nodes = _estimates()
    upper = upper.model_copy(update={"value": Q(exact.value.value - offset, "hartree")})
    if violates:
        with pytest.raises(CheckError, match="CF-E130") as error:
            _check_relation_consistency({exact_id: exact, upper_id: upper}, nodes)
        assert exact_id in str(error.value) and upper_id in str(error.value)
        assert "Ha" in str(error.value)
    else:
        _check_relation_consistency({exact_id: exact, upper_id: upper}, nodes)


@pytest.mark.parametrize("offset,violates", [(1e-7 - 1e-9, False),
                                            (1e-7 + 1e-9, True)])
def test_equivalent_tolerance(offset, violates):
    exact_id, other_id, exact, other, nodes = _estimates()
    other = other.model_copy(update={"relation": TargetRelation.equivalent,
                                     "value": Q(exact.value.value + offset, "hartree")})
    if violates:
        with pytest.raises(CheckError, match="CF-E130"):
            _check_relation_consistency({exact_id: exact, other_id: other}, nodes)
    else:
        _check_relation_consistency({exact_id: exact, other_id: other}, nodes)


@pytest.mark.parametrize("relation", [TargetRelation.approximate,
                                      TargetRelation.stochastic_estimate,
                                      TargetRelation.model_change])
def test_noncomparable_relations_are_ignored(relation):
    exact_id, other_id, exact, other, nodes = _estimates()
    other = other.model_copy(update={"relation": relation,
                                     "value": Q(exact.value.value - 10, "hartree")})
    _check_relation_consistency({exact_id: exact, other_id: other}, nodes)
