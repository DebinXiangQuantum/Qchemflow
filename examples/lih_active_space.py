from pyscf import mcscf

from chemflow.impls.fci import sector_diag_numpy
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def main() -> None:
    store = ArtifactStore()
    data = build_molecule("Li 0 0 0; H 0 0 1.6", ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=2, n_orb=3))
    energy = sector_diag_numpy(data.hamiltonian, data.sector, store)
    casci = mcscf.CASCI(data.mf, 3, 2)
    reference = casci.kernel()[0]
    print(f"LiH CAS(2e,3o) sector_diag_numpy={energy:.12f} CASCI={reference:.12f}")
    assert abs(energy - reference) < 1e-8


if __name__ == "__main__":
    main()
