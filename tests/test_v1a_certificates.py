
import numpy as np

from chemflow.runtime.artifacts import ArtifactStore

MODEL = {"U": 2., "eps_bath": [0.], "t_bath": [0.5], "unit": "hartree"}
SECTOR = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
SPEC = {"question": "H2", "target": {
    "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
    "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
    "sector": SECTOR, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": .0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 10000},
    "allowed_effects": ["pure", "simulator"], "decision": "choose"}


def test_artifact_store_persistence_round_trip(tmp_path):
    store = ArtifactStore()
    reference = store.put(np.array([5, 6], dtype=np.int64))
    store.persist(tmp_path)
    assert ArtifactStore(tmp_path).get(reference).tolist() == [5, 6]
