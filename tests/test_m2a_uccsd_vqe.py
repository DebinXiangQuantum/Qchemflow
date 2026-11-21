import pytest

from chemflow.impls.pyscf_impls import build_molecule, fci_pyscf
from chemflow.impls.sector import SectorEngine
from chemflow.impls.vqe import vqe
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


@pytest.mark.parametrize(("atoms", "active"), [
    ("H 0 0 0; H 0 0 0.74", None),
    ("Li 0 0 0; H 0 0 1.6", (2, 3)),
])
def test_vqe_reaches_fci(atoms, active):
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=active[0], n_orb=active[1])
                          if active else None)
    result = vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector)
    exact, _ = fci_pyscf(data, store)
    assert exact - 1e-9 <= result.energy < exact + 1e-6
    assert result.n_energy_evals > 0
