import pytest

from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.units import Q


def test_stochastic_gate_uses_confidence_and_freezes_it() -> None:
    with build.program("stochastic") as builder:
        hamiltonian = builder.hamiltonian(
            builder.molecule("H 0 0 0; H 0 0 0.74"),
            builder.electronic_model(basis="sto-3g"),
        )
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        candidate = builder.node("sample", operator=encoding, prep=prep, shots=20, seed=1)
        reference = builder.node("expectation", operator=encoding, prep=prep)
        gate = builder.accept(candidate, abs_error_le=Q(0.04, "hartree"),
                              reference=reference.id, confidence=0.90)
    program = builder.to_ir()
    verdict = Executor(molecular_registry()).run(program, task=freeze_task(program))["gates"][0]
    assert verdict.stat_err is not None
    assert verdict.z == pytest.approx(1.64485362695)
    assert verdict.passed == (verdict.measured.value + verdict.z * verdict.stat_err <= 0.04)
    assert gate.intact()
    gate.confidence = 0.95
    assert not gate.intact()
