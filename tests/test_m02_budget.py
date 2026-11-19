import pytest

from chemflow import build
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task


def test_unknown_wall_estimate_uses_measured_spend() -> None:
    registry = molecular_registry()
    with build.program("measured_budget") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        builder.budget(wall_s=0)
    program = builder.to_ir()
    with pytest.raises(CheckError) as error:
        Executor(registry).run(program, task=freeze_task(program))
    assert "CF-E070" in {item.code for item in error.value.diagnostics}
    builder.budget(wall_s=600)
    assert molecule.id in Executor(registry).run(builder.to_ir(), task=freeze_task(builder.to_ir()))


def test_known_cost_refused_before_adapter() -> None:
    registry = molecular_registry()
    calls = []
    registry.implementations["molecule"].resource_model = lambda **kwargs: {"wall_s": 2}
    registry.implementations["molecule"].adapter = lambda store, **kwargs: calls.append(1)
    with build.program("preflight_budget") as builder:
        builder.molecule("H 0 0 0")
        builder.budget(wall_s=1)
    program = builder.to_ir()
    with pytest.raises(CheckError):
        Executor(registry).run(program, task=freeze_task(program))
    assert calls == []
