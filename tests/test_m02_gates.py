import pytest

from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.types import GateVerdict
from chemflow.units import Q


def test_gate_verdicts_include_relation_and_converted_error() -> None:
    with build.program("gate_verdicts") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"), model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        exact = builder.ground_energy(hamiltonian, sector)
        upper = builder.ground_energy(hamiltonian, sector)
        builder.refine(upper, "hf")
        builder.bind(upper, "hf")
        builder.accept(exact, relation="approximate")
        builder.accept(exact, abs_error_le=Q(1, "eV"), reference=exact.id)
    program = builder.to_ir()
    verdicts = Executor(molecular_registry()).run(program, task=freeze_task(program))["gates"]
    assert all(isinstance(verdict, GateVerdict) for verdict in verdicts)
    assert verdicts[0].passed
    assert verdicts[0].relation == "equivalent"
    assert verdicts[1].passed
    assert verdicts[1].measured.unit == "eV"
    assert verdicts[1].reference == exact.id


def test_gate_error_is_converted_into_the_gate_unit() -> None:
    with build.program("gate_units") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"), model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        exact = builder.ground_energy(hamiltonian, sector)
        upper = builder.ground_energy(hamiltonian, sector)
        builder.refine(upper, "hf")
        builder.bind(upper, "hf")
        builder.accept(upper, abs_error_le=Q(1, "eV"), reference=exact.id)
        builder.accept(upper, abs_error_le=Q(0.1, "eV"), reference=exact.id)
    program = builder.to_ir()
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    gap_ha = results[upper.id].value.value - results[exact.id].value.value
    loose, tight = results["gates"]
    assert gap_ha > 0.01
    assert loose.measured.unit == "eV"
    assert loose.measured.value == pytest.approx(Q(gap_ha, "hartree").to("eV").value, rel=1e-12)
    assert loose.measured.value > 0.1
    assert loose.passed and not tight.passed
