import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chemflow.impls.fci import sector_states
from chemflow.impls.lucj import LucjParams, lucj_state
from chemflow.impls.pyscf_impls import build_molecule, fci_pyscf
from chemflow.impls.sector import SectorEngine, hf_vector
from chemflow.impls.vqe import vqe
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel, Sector


@pytest.mark.parametrize(("n_orb", "layers"), [(4, 1), (4, 2), (6, 1), (6, 2)])
def test_reference_amplitudes(n_orb, layers):
    path = Path(__file__).resolve().parents[1] / "reference" / "sqd_lib.py"
    if not path.exists():
        pytest.skip("reference/sqd_lib.py reference implementation is absent")
    spec = importlib.util.spec_from_file_location("sqd_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sector = Sector(n_orb=n_orb, n_alpha=2, n_beta=2)
    rng = np.random.default_rng(33)
    parameters = [LucjParams.random(n_orb, rng) for _ in range(layers)]
    reference = module.reference_state_amplitudes(
        SimpleNamespace(norb=n_orb, n_alpha=2, n_beta=2), parameters)
    actual = lucj_state(sector, parameters)
    expected = np.array([reference.get(bits, 0) for bits in sector_states(sector)])
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, expected * phase / abs(phase), atol=1e-10)


def test_lucj_variational_h4():
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0",
                          ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=4))
    engine = SectorEngine(data.hamiltonian, data.sector, store)
    result = vqe(engine, data.sector, ansatz="lucj", maxiter=100)
    exact, _ = fci_pyscf(data, store)
    assert exact - 1e-9 <= result.energy < engine.energy(hf_vector(data.sector))
