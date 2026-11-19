import numpy as np
from pyscf import fci

from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sci import sci_pyscf
from chemflow.impls.spin_complete import complete_spin_strings
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def test_spin_completion_makes_product_projector_commute_with_s2():
    rng = np.random.default_rng(19)
    n_orb = 5
    n_elec = 2
    all_strings = list(map(int, fci.cistring.make_strings(range(n_orb), n_elec)))
    for _ in range(8):
        initial = set(map(int, rng.choice(all_strings, size=3, replace=False)))
        completed, reason = complete_spin_strings(initial, n_orb)
        assert reason is None
        addresses = [fci.cistring.str2addr(n_orb, n_elec, string)
                     for string in completed]
        vector = np.zeros((len(all_strings), len(all_strings)))
        vector[np.ix_(addresses, addresses)] = rng.normal(
            size=(len(addresses), len(addresses)))
        transformed = fci.spin_op.contract_ss(vector, n_orb, (n_elec, n_elec))
        outside = np.ones_like(vector, dtype=bool)
        outside[np.ix_(addresses, addresses)] = False
        assert np.all(np.abs(transformed[outside]) < 1e-10)


def test_spin_completion_cap_preserves_original_space():
    selected = {3, 12}
    completed, reason = complete_spin_strings(selected, 4, max_product_dimension=4)
    assert completed == selected
    assert reason is not None


def test_n2_selected_ci_is_an_exact_singlet():
    store = ArtifactStore()
    data = build_molecule("N 0 0 0; N 0 0 1.10", ElectronicModel(basis="6-31g"),
                          store, active=ActiveSpaceSpec(n_elec=8, n_orb=10))
    _, diagnostics = sci_pyscf(data, store, select_cutoff=1e-2, total_spin=0)
    assert abs(diagnostics["s2"]) < 1e-8
    assert diagnostics["dimension_after_completion"] >= diagnostics[
        "dimension_before_completion"]
    assert diagnostics["spin_completion_reason"] is None
