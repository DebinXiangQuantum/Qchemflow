"""Multistart keeps the single-start result among its candidates."""

from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import SectorEngine
from chemflow.impls.vqe import vqe
from chemflow.runtime import ArtifactStore
from chemflow.types import ElectronicModel


def test_lucj_start_energies_include_initial_seed():
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 0.74", ElectronicModel(basis="sto-3g"), store)
    engine = SectorEngine(data.hamiltonian, data.sector, store)
    single = vqe(engine, data.sector, ansatz="lucj", maxiter=5, seed=7)
    multi = vqe(engine, data.sector, ansatz="lucj", maxiter=5, seed=7, n_starts=3)
    assert len(multi.start_energies) == 3
    assert multi.start_energies[0] == single.energy
    assert multi.energy == min(multi.start_energies)
