import numpy as np
import pytest
from pyscf import fci, mcscf

from chemflow.impls.fci import sector_diag_numpy
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.statevector import encode, expectation, prepare_hf
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


@pytest.mark.parametrize("atoms,active", [
    ("H 0 0 0; H 0 0 0.74", None),
    ("Li 0 0 0; H 0 0 1.6", ActiveSpaceSpec(n_elec=2, n_orb=3)),
])
def test_reference_physics(atoms: str, active: ActiveSpaceSpec | None) -> None:
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis="sto-3g"), store, active=active)
    if active:
        casci = mcscf.CASCI(data.mf, active.n_orb, active.n_elec)
        reference = casci.kernel()[0]
    else:
        reference = fci.FCI(data.mf).kernel()[0]
    assert sector_diag_numpy(data.hamiltonian, data.sector, store) == pytest.approx(
        reference, abs=1e-8)
    encoding, operator = encode(data.hamiltonian, store)
    prep = prepare_hf(data.hamiltonian, encoding, store, data.sector.n_alpha,
                      data.sector.n_beta)
    assert expectation(operator, prep, store) == pytest.approx(data.mf.e_tot, abs=1e-8)
    if not active:
        pauli = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
                 "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
        reconstructed = np.zeros_like(store.get(operator.matrix_ref), dtype=complex)
        for word, coefficient in operator.terms.items():
            term = np.array([[1]], dtype=complex)
            for letter in reversed(word):
                term = np.kron(term, pauli[letter])
            reconstructed += coefficient * term
        assert np.allclose(reconstructed, store.get(operator.matrix_ref), atol=1e-10)


def test_examples_smoke() -> None:
    from examples.h2_pes import main as h2_main
    from examples.lih_active_space import main as lih_main

    h2_main()
    lih_main()
