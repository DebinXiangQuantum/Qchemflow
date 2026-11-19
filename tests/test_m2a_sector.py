import numpy as np
import pytest

from chemflow.impls.fci import matrix, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import SectorEngine, hf_vector
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


@pytest.mark.parametrize(("atoms", "active"), [
    ("H 0 0 0; H 0 0 0.74", None),
    ("Li 0 0 0; H 0 0 1.6", (2, 3)),
    ("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0", (4, 4)),
    ("Li 0 0 0; H 0 0 1.6", (4, 5)),
])
def test_contract_matches_dense(atoms, active):
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=active[0], n_orb=active[1])
                          if active else None)
    engine = SectorEngine(data.hamiltonian, data.sector, store)
    vector = np.random.default_rng(17).normal(size=len(engine.states))
    expected = matrix(data.hamiltonian, store, sector_states(data.sector)) @ vector
    np.testing.assert_allclose(engine.apply(vector), expected, atol=1e-10)
    assert abs(engine.energy(vector) - np.vdot(vector, expected)) < 1e-10
    assert np.linalg.norm(hf_vector(data.sector)) == 1
