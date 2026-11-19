"""Reviewer regression tests for M2a (written by Claude; implementers must not edit)."""
import numpy as np
import pytest
from test_review_m0_2 import _s2_operator

from chemflow import build
from chemflow.impls.ansatz import uccsd_pool
from chemflow.impls.fci import sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"
O2 = "O 0 0 0; O 0 0 1.21"


def _run(program):
    return Executor(molecular_registry()).run(program, task=freeze_task(program))


@pytest.mark.parametrize("geometry,n_elec,n_orb", [(H4, 4, 4), ("Li 0 0 0; H 0 0 1.6", 4, 5)])
def test_spin_adapted_generators_commute_with_s2(geometry: str, n_elec: int, n_orb: int) -> None:
    store = ArtifactStore()
    data = build_molecule(geometry, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=n_elec, n_orb=n_orb))
    s2 = _s2_operator(sector_states(data.sector), n_orb)
    names, generators = uccsd_pool(data.sector, spin_adapted=True)
    assert generators
    for name, generator in zip(names, generators):
        dense = generator.toarray()
        assert np.abs(dense @ s2 - s2 @ dense).max() < 1e-12, name


def test_spin_adapted_vqe_stays_singlet_and_keeps_upper_bound() -> None:
    with build.program("h4_sa") as b:
        ham = b.hamiltonian(b.molecule(H4), b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 4, "n_orb": 4})
        energy = b.ground_energy(ham, b.sector(n_alpha=2, n_beta=2, n_orb=4, s=0),
                                 ansatz="uccsd", spin_adapted=True, s=0)
        b.refine(energy, "vqe")
        b.bind(energy, "vqe_statevector")
    result = _run(b.to_ir())[energy.id]
    assert abs(result.diagnostics["s2"]) < 1e-8
    assert result.relation == "variational_upper_bound"


def test_adapt_respects_spin_target() -> None:
    # Closed-shell ADAPT stays in S=0; a triplet request in the Ms=0 sector must not be
    # labelled an upper bound for the S=1 target.
    with build.program("o2_adapt_triplet") as b:
        ham = b.hamiltonian(b.molecule(O2), b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 2, "n_orb": 2})
        energy = b.ground_energy(ham, b.sector(n_alpha=1, n_beta=1, n_orb=2, s=1), s=1)
        b.refine(energy, "adapt_vqe")
        b.bind(energy, "adapt_statevector")
    result = _run(b.to_ir())[energy.id]
    assert abs(result.diagnostics["s2"] - 2.0) > 1e-4
    assert result.relation == "approximate"
    assert result.provenance.relation_weakening_reason
