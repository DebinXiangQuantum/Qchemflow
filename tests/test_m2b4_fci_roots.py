import numpy as np
import pytest
from test_review_m0_2 import _s2_operator

from chemflow.impls.fci import matrix, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.ops.molecular import ground_energy
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


@pytest.mark.parametrize("geometry,electrons,orbitals", [
    ("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0", 4, 4),
    ("Li 0 0 0; H 0 0 1.6", 4, 5),
    ("O 0 0 0; O 0 0 1.2", 2, 2),
])
@pytest.mark.parametrize("spin", [0, 1])
def test_spin_targeted_fci_matches_dense_spectrum(geometry, electrons, orbitals, spin):
    store = ArtifactStore()
    data = build_molecule(geometry, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
    states = sector_states(data.sector)
    energies, vectors = np.linalg.eigh(matrix(data.hamiltonian, store, states).toarray())
    spin_operator = _s2_operator(states, orbitals)
    expected = next(energy for energy, vector in zip(energies, vectors.T)
                    if abs(vector @ spin_operator @ vector - spin * (spin + 1)) < 1e-8)
    actual, diagnostics = ground_energy(data, {"n_alpha": data.sector.n_alpha,
                                               "n_beta": data.sector.n_beta,
                                               "n_orb": orbitals, "s": spin},
                                        store, "fci_pyscf")
    assert actual == pytest.approx(expected, abs=1e-8)
    assert diagnostics["s2"] == pytest.approx(spin * (spin + 1), abs=1e-8)
    assert diagnostics["n_roots_examined"] >= 1
