"""Reviewer regression tests for M0/M1 (written by Claude; implementers must not edit).

Every program here is built only through the public builder, with no hand-planted
context flags: the checker/runtime has to infer context itself.
"""
import pytest
from pyscf import gto, mcscf, scf

from chemflow import build
from chemflow.check import check
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.registry import ImplementationSpec
from chemflow.runtime import Executor, freeze_task
from chemflow.types import BackendKind, Evidence


def codes(program, registry=None, **kwargs) -> list[str]:
    return [d.code for d in check(program, registry or molecular_registry(), **kwargs)]


def h2_program(name: str, r: float = 0.74, basis: str = "sto-3g"):
    builder = build.program(name).__enter__()
    mol = builder.molecule(f"H 0 0 0; H 0 0 {r}")
    model = builder.electronic_model(basis=basis)
    ham = builder.hamiltonian(mol, model)
    sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
    return builder, ham, sector


def test_active_space_preserves_charge_and_spin() -> None:
    with build.program("cation") as b:
        mol = b.molecule("Li 0 0 0; H 0 0 1.6", charge=1, spin=1)
        model = b.electronic_model(basis="sto-3g", reference="ROHF")
        scf_node = b.node("scf", molecule=mol, model=model)
        active = b.node("active_space", scf=scf_node, n_elec=1, n_orb=3)
    results = Executor(molecular_registry()).run(b.to_ir(), task=freeze_task(b.to_ir()))
    assert results[active.id].mol.charge == 1
    assert results[active.id].mol.nelectron == 3


def test_binding_must_match_refinement() -> None:
    b, ham, sector = h2_program("mismatch")
    energy = b.ground_energy(ham, sector)
    b.refine(energy, "hf")
    b.bind(energy, "fci_pyscf")
    found = codes(b.to_ir())
    assert "CF-E050" in found or "CF-E080" in found


def test_cross_geometry_expectation_rejected_statically() -> None:
    with build.program("xgeom") as b:
        model = b.electronic_model(basis="sto-3g")
        h_08 = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.8"), model)
        h_10 = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 1.0"), model)
        enc_08 = b.node("encode", hamiltonian=h_08)
        enc_10 = b.node("encode", hamiltonian=h_10)
        prep_10 = b.node("prepare", hamiltonian=h_10, encoding=enc_10)
        b.node("expectation", operator=enc_08, prep=prep_10)
    assert "CF-E010" in codes(b.to_ir())


def test_energy_difference_across_basis_rejected_statically() -> None:
    with build.program("xbasis") as b:
        mol = b.molecule("H 0 0 0; H 0 0 0.74")
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        e_small = b.ground_energy(b.hamiltonian(mol, b.electronic_model(basis="sto-3g")), sector)
        sector_big = b.sector(n_alpha=1, n_beta=1, n_orb=4)
        e_big = b.ground_energy(b.hamiltonian(mol, b.electronic_model(basis="6-31g")), sector_big)
        b.node("energy_difference", left=e_small, right=e_big)
    assert "CF-E011" in codes(b.to_ir())


def test_certified_evidence_cannot_be_constructed_outside_runtime() -> None:
    with pytest.raises(Exception):
        Evidence(status="certified_bound", artifact_ref="anything")
    with pytest.raises(Exception):
        Evidence(status="certified_bound", artifact_ref="anything", source="implementation")


def test_hardware_node_predicted_beyond_budget_never_runs() -> None:
    """Revised for R1: devices (CPU, GPU, QPU) are limited by the budget, not by approvals."""
    registry = molecular_registry()
    calls: list[int] = []
    registry.register(ImplementationSpec(
        id="fake_qpu", version="1", target="sector", backend_kind=BackendKind.qpu,
        adapter=lambda store, **kwargs: calls.append(1), effects="random",
        resource_model=lambda **kwargs: {"qpu_shots": 5000, "qpu_s": 3.0}))
    with build.program("hardware") as b:
        node = b.node("sector", n_alpha=1, n_beta=1, n_orb=2)  # valid, input-free: only the budget refuses it
        b.bind(node, "fake_qpu")
        b.budget(qpu_shots=1000)
    program = b.to_ir()
    with pytest.raises(CheckError) as info:
        Executor(registry).run(program, task=freeze_task(program))
    assert "CF-E070" in {d.code for d in info.value.diagnostics}
    assert not calls


def test_gate_tamper_detected_even_after_refreeze() -> None:
    b, ham, sector = h2_program("gate")
    energy = b.ground_energy(ham, sector)
    b.accept(energy, relation="equivalent")
    program = b.to_ir()
    task = freeze_task(program)
    program.gates[0].relation = "approximate"
    program.gates[0].freeze()
    with pytest.raises(CheckError) as info:
        Executor(molecular_registry()).run(program, task=task)
    assert info.value.diagnostics, "tampering must produce structured diagnostics"


def test_singlet_request_never_returns_triplet() -> None:
    geometry = "O 0 0 0; O 0 0 1.21"
    with build.program("o2") as b:
        mol = b.molecule(geometry)
        ham = b.hamiltonian(mol, b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 2, "n_orb": 2})
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2, s=0)
        energy = b.ground_energy(ham, sector, s=0)
    program = b.to_ir()
    mf = scf.RHF(gto.M(atom=geometry, basis="sto-3g", verbose=0)).run()
    casci = mcscf.CASCI(mf, 2, 2)
    casci.fix_spin_(ss=0)
    singlet = casci.kernel()[0]
    try:
        result = Executor(molecular_registry()).run(program, task=freeze_task(program))
    except CheckError as error:
        assert "CF-E020" in {d.code for d in error.diagnostics}
        return
    assert result[energy.id].value.value == pytest.approx(singlet, abs=1e-6)


def test_fci_binding_supported_for_active_space() -> None:
    geometry = "Li 0 0 0; H 0 0 1.6"
    with build.program("lih") as b:
        ham = b.hamiltonian(b.molecule(geometry), b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 2, "n_orb": 3})
        energy = b.ground_energy(ham, b.sector(n_alpha=1, n_beta=1, n_orb=3))
        b.refine(energy, "fci_pyscf")
        b.bind(energy, "fci_pyscf")
    program = b.to_ir()
    mf = scf.RHF(gto.M(atom=geometry, basis="sto-3g", verbose=0)).run()
    reference = mcscf.CASCI(mf, 3, 2).kernel()[0]
    result = Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert result[energy.id].value.value == pytest.approx(reference, abs=1e-8)


def test_runtime_failures_are_structured() -> None:
    registry = molecular_registry()
    registry.implementations["compare"].capability_predicate = lambda **kwargs: False
    with build.program("cap") as b:
        b.node("compare")
    program = b.to_ir()
    with pytest.raises(CheckError) as info:
        Executor(registry).run(program, task=freeze_task(program))
    assert info.value.diagnostics and all(d.message for d in info.value.diagnostics)
