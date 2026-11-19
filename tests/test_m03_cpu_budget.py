import pytest

from chemflow import build
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task


def test_cpu_spend_is_recorded_and_enforced() -> None:
    with build.program("cpu") as builder:
        hamiltonian = builder.hamiltonian(
            builder.molecule("H 0 0 0; H 0 0 0.74"),
            builder.electronic_model(basis="sto-3g"),
        )
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        energy = builder.ground_energy(hamiltonian, sector)
    program = builder.to_ir()
    result = Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert result[energy.id].cost.cpu_core_s > 0
    program.budget["cpu_core_s"] = 0
    with pytest.raises(CheckError) as error:
        Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert "CF-E070" in {issue.code for issue in error.value.diagnostics}
