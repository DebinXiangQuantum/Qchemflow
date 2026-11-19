import pytest

from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task


@pytest.mark.parametrize("method", ["hf", "mp2"])
def test_only_computed_spin_diagnostics_are_reported(method: str) -> None:
    with build.program("spin_diagnostics") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"), model)
        energy = builder.ground_energy(hamiltonian,
                                       builder.sector(n_alpha=1, n_beta=1, n_orb=2))
        builder.refine(energy, method)
        builder.bind(energy, method)
    program = builder.to_ir()
    diagnostics = Executor(molecular_registry()).run(program, task=freeze_task(program))[
        energy.id].diagnostics
    if method == "hf":
        assert diagnostics["s2"] == pytest.approx(0)
    else:
        assert "s2" not in diagnostics
