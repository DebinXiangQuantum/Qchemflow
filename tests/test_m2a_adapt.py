from chemflow.impls.adapt import adapt_vqe
from chemflow.impls.pyscf_impls import build_molecule, fci_pyscf
from chemflow.impls.sector import SectorEngine
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def test_h4_adapt():
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0",
                          ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=4))
    result = adapt_vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector)
    exact, _ = fci_pyscf(data, store)
    assert exact - 1e-9 <= result.energy < exact + 1.6e-3
    assert result.n_operators <= 40
    assert result.n_gradient_evaluations > 0
    assert len(result.selected_operators) == result.n_operators
    assert molecular_registry().refinements["adapt_vqe"].relation == (
        TargetRelation.variational_upper_bound)
