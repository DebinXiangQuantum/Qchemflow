"""Reviewer regression tests for M2a (written by Claude; implementers must not edit)."""
import numpy as np
import pytest
from test_review_m0_2 import _s2_operator

from chemflow.impls.ansatz import uccsd_pool
from chemflow.impls.fci import sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"
O2 = "O 0 0 0; O 0 0 1.21"


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
