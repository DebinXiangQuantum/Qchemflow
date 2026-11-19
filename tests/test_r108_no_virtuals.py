import numpy as np
import pytest

from chemflow.impls.lucj import LucjParams, lucj_state
from chemflow.impls.pyscf_impls import build_molecule, classical_energy
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ElectronicModel, Sector


@pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd_t"])
def test_no_virtual_correlation_short_circuits(method, monkeypatch):
    store = ArtifactStore()
    data = build_molecule("He 0 0 0", ElectronicModel(basis="sto-3g"), store)

    def forbidden(*args, **kwargs):
        raise AssertionError("No correlated solver should be constructed")

    monkeypatch.setattr("chemflow.impls.pyscf_impls.mp.MP2", forbidden)
    monkeypatch.setattr("chemflow.impls.pyscf_impls.cc.CCSD", forbidden)
    diagnostics = {}
    assert classical_energy(data, method, store, diagnostics) == data.mf.e_tot
    assert "No virtual" in diagnostics["note"]


def test_single_orbital_lucj_has_no_rotation():
    state = lucj_state(Sector(n_orb=1, n_alpha=1, n_beta=1), [LucjParams.zeros(1)])
    np.testing.assert_allclose(state, [1])
