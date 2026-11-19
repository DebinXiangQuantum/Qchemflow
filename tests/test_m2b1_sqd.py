"""SQD never invokes the full-sector dense Hamiltonian builder."""

from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sqd import sqd_sector
from chemflow.runtime import ArtifactStore
from chemflow.types import ElectronicModel, Samples


def test_selected_strings_do_not_require_full_matrix(monkeypatch):
    from chemflow.impls import fci as full_sector

    def forbidden(*args, **kwargs):
        raise AssertionError("full-sector matrix used")

    monkeypatch.setattr(full_sector, "matrix", forbidden)
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 0.74", ElectronicModel(basis="sto-3g"), store)
    samples = Samples(counts={"5": 20}, shots=20, encoding_id="test",
                      basis="computational", source="test", seed=1)
    energy, details = sqd_sector(data, data.sector, store, samples,
                                 n_batches=1, max_iter=1, with_ext=False)
    assert isinstance(energy, float)
    assert details["subspace_dimension"] <= 4
