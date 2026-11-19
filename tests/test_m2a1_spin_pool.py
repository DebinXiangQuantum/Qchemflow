import numpy as np
import pytest
from test_review_m0_2 import _s2_operator

from chemflow.impls.ansatz import uccsd_pool
from chemflow.impls.fci import sector_states
from chemflow.impls.pyscf_impls import build_molecule, fci_pyscf
from chemflow.impls.sector import SectorEngine
from chemflow.impls.vqe import vqe
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"
LIH = "Li 0 0 0; H 0 0 1.6"


@pytest.mark.parametrize(("geometry", "electrons", "orbitals"), [
    (H4, 4, 4), (LIH, 4, 5),
])
def test_spin_free_pool_commutes_and_is_independent(geometry, electrons, orbitals):
    store = ArtifactStore()
    data = build_molecule(geometry, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
    names, generators = uccsd_pool(data.sector, spin_adapted=True)
    assert len(names) == len(generators) == len(set(names))
    assert any("E3,0 E2,1" in name for name in names)
    spin = _s2_operator(sector_states(data.sector), orbitals)
    for generator in generators:
        dense = generator.toarray()
        np.testing.assert_allclose(dense @ spin, spin @ dense, atol=1e-12)
    assert np.linalg.matrix_rank(np.stack([generator.toarray().ravel()
                                           for generator in generators])) == len(generators)


def test_spin_adapted_h4_vqe_preserves_singlet_and_bound():
    store = ArtifactStore()
    data = build_molecule(H4, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=4))
    result = vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector,
                 spin_adapted=True)
    exact, _ = fci_pyscf(data, store)
    assert abs(result.s2) < 1e-8
    assert result.energy >= exact - 1e-9
