import pytest

from chemflow.impls.fci import sector_diag_numpy
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sci import sci_pyscf
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


@pytest.mark.parametrize("atoms,active", [
    ("Li 0 0 0; H 0 0 1.6", ActiveSpaceSpec(n_elec=4, n_orb=5)),
    ("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0",
     ActiveSpaceSpec(n_elec=4, n_orb=4)),
])
def test_selected_ci_tight_and_loose(atoms, active):
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis="sto-3g"), store, active=active)
    reference = sector_diag_numpy(data.hamiltonian, data.sector, store)
    tight, diagnostics = sci_pyscf(data, store, select_cutoff=1e-10)
    assert tight == pytest.approx(reference, abs=1e-7)
    assert diagnostics["determinant_count"] > 0
    loose, _ = sci_pyscf(data, store, select_cutoff=0.01)
    assert loose >= reference - 1e-9
