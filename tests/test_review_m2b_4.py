"""Reviewer regression tests, M2b round 4 (written by Claude; implementers must not edit).

Ground truth (reviewer, dense diagonalization of all 36 H4 square CAS(4,4) states with an
independent S2 operator): singlet ground state -1.949191 Ha. fci_pyscf with s=0 returned
-1.621757 Ha (an excited singlet, above HF) labelled `equivalent`, because PySCF's
fix_spin_ with shift=2.0 converged to the wrong root while reporting converged=True.
"""
import re

import numpy as np
import pytest
from test_review_m0_2 import _s2_operator

from chemflow import build
from chemflow.diagnostics import CheckError
from chemflow.impls.fci import matrix, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.ops.molecular import ground_energy, molecular_registry
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ArtifactStore, Executor, freeze_task
from chemflow.types import ActiveSpaceSpec, BackendKind, ElectronicModel

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"


def _dense_lowest(geometry: str, n_elec: int, n_orb: int, total_spin: int) -> float:
    store = ArtifactStore()
    data = build_molecule(geometry, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=n_elec, n_orb=n_orb))
    states = sector_states(data.sector)
    energies, vectors = np.linalg.eigh(matrix(data.hamiltonian, store, states).toarray())
    s2 = _s2_operator(states, n_orb)
    target = total_spin * (total_spin + 1)
    return next(e for e, v in zip(energies, vectors.T) if abs(v @ s2 @ v - target) < 1e-8)


@pytest.mark.parametrize("total_spin", [0, 1])
def test_spin_targeted_fci_pyscf_finds_the_lowest_state_of_that_spin(total_spin: int) -> None:
    with build.program("fci_spin") as b:
        ham = b.hamiltonian(b.molecule(H4), b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 4, "n_orb": 4})
        energy = b.ground_energy(ham, b.sector(n_alpha=2, n_beta=2, n_orb=4, s=total_spin),
                                 s=total_spin)
        b.refine(energy, "fci")
        b.bind(energy, "fci_pyscf")
    program = b.to_ir()
    result = Executor(molecular_registry()).run(program, task=freeze_task(program))[energy.id]
    assert result.value.value == pytest.approx(_dense_lowest(H4, 4, 4, total_spin), abs=1e-8)


def _registry_with_broken_exact_solver():
    registry = molecular_registry()

    def broken(store, hamiltonian, sector, **kwargs):
        energy, diagnostics = ground_energy(hamiltonian, sector, store, "sector_diag_numpy")
        return energy + 0.5, diagnostics

    registry.register(ImplementationSpec(
        id="broken_fci", version="1", target="ground_energy", implements="fci",
        backend_kind=BackendKind.cpu_numeric, adapter=broken,
        resource_model=lambda **kwargs: {"qpu_shots": 0, "qpu_s": 0, "gpu_s": 0}))
    return registry


def test_relation_contradictions_are_detected_at_runtime() -> None:
    with build.program("contradiction") as b:
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"))
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        upper = b.ground_energy(ham, sector)
        b.refine(upper, "hf")
        b.bind(upper, "hf")
        exact = b.ground_energy(ham, sector)
        b.refine(exact, "fci")
        b.bind(exact, "broken_fci")
    program = b.to_ir()
    with pytest.raises(CheckError) as info:
        Executor(_registry_with_broken_exact_solver()).run(program, task=freeze_task(program))
    found = [d for d in info.value.diagnostics if d.code == "CF-E130"]
    text = " ".join(f"{d.node or ''} {d.message}" for d in found)
    assert found
    for node_id in (upper.id, exact.id):
        assert re.search(rf"\b{node_id}\b", text), node_id


def test_consistent_relations_raise_nothing() -> None:
    with build.program("consistent") as b:
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"))
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        for template, binding in (("hf", "hf"), ("fci", "fci_pyscf"), ("fci", "sector_diag_numpy"),
                                  ("sci", "sci_pyscf")):
            node = b.ground_energy(ham, sector)
            b.refine(node, template)
            b.bind(node, binding)
    program = b.to_ir()
    Executor(molecular_registry()).run(program, task=freeze_task(program))
