import pytest

from chemflow import build
from chemflow.check import check
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task


@pytest.mark.parametrize("operation", ["energy_difference", "compare"])
def test_active_space_identity_static_and_runtime(operation: str) -> None:
    with build.program("active_identity") as builder:
        molecule = builder.molecule("Li 0 0 0; H 0 0 1.6")
        model = builder.electronic_model(basis="sto-3g")
        small = builder.ground_energy(
            builder.hamiltonian(molecule, model, active={"n_elec": 2, "n_orb": 2}),
            builder.sector(n_alpha=1, n_beta=1, n_orb=2))
        large = builder.ground_energy(
            builder.hamiltonian(molecule, model, active={"n_elec": 2, "n_orb": 3}),
            builder.sector(n_alpha=1, n_beta=1, n_orb=3))
        builder.node(operation, left=small, right=large)
    program = builder.to_ir()
    assert "CF-E011" in {issue.code for issue in check(program, molecular_registry(),
                                                        task=freeze_task(program))}
    with pytest.raises(CheckError):
        Executor(molecular_registry()).run(program, task=freeze_task(program))
    program.nodes[-1].params["model_change"] = True
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert results[program.nodes[-1].id].relation == "model_change"
