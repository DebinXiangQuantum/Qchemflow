import pytest

from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task

O2 = "O 0 0 0; O 0 0 1.21"


@pytest.mark.parametrize(("sector_spin", "refinement_spin", "expected"), [
    (1, None, "approximate"),
    (None, 1, "approximate"),
    (0, None, "variational_upper_bound"),
])
def test_adapt_spin_target_relation(sector_spin, refinement_spin, expected):
    with build.program("adapt-spin") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule(O2),
                                          builder.electronic_model(basis="sto-3g"),
                                          active={"n_elec": 2, "n_orb": 2})
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2, s=sector_spin)
        energy = builder.ground_energy(hamiltonian, sector, s=sector_spin)
        options = {"s": refinement_spin} if refinement_spin is not None else {}
        builder.refine(energy, "adapt_vqe", **options)
        builder.bind(energy, "adapt_statevector")
    program = builder.to_ir()
    result = Executor(molecular_registry()).run(program, task=freeze_task(program))[energy.id]
    assert result.relation == expected
    assert abs(result.diagnostics["s2"]) < 1e-8
    assert bool(result.provenance.relation_weakening_reason) == (expected == "approximate")
