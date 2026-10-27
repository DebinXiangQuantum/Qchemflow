import numpy as np
import pytest
from pyscf import fci, mcscf

from chemflow import build
from chemflow.impls.fci import sector_diag_numpy
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.statevector import encode, expectation, prepare_hf, sample
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import ArtifactStore, Executor
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


def test_relations_and_provenance() -> None:
    with build.program("h2") as builder:
        mol = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(mol, model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        energies = {}
        for method in ("fci_pyscf", "hf", "ccsd"):
            energy = builder.ground_energy(hamiltonian, sector)
            builder.refine(energy, method)
            builder.bind(energy, method)
            energies[method] = energy.id
    results = Executor(molecular_registry()).run(builder.to_ir())
    fci_energy = results[energies["fci_pyscf"]]
    hf_energy = results[energies["hf"]]
    ccsd_energy = results[energies["ccsd"]]
    assert hf_energy.relation == "variational_upper_bound"
    assert hf_energy.value.value >= fci_energy.value.value
    assert ccsd_energy.relation == "approximate"
    for estimate in (hf_energy, fci_energy, ccsd_energy):
        assert estimate.provenance.input_hashes and estimate.cost.wall_s >= 0


def test_finite_shots() -> None:
    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 0.74", ElectronicModel(basis="sto-3g"), store)
    encoding, operator = encode(data.hamiltonian, store)
    prep = prepare_hf(data.hamiltonian, encoding, store, 1, 1)
    exact = expectation(operator, prep, store)
    _, mean, error = sample(operator, prep, store, 1000, seed=42)
    assert error > 0 and abs(mean - exact) < 5 * error
    _, _, larger_error = sample(operator, prep, store, 100_000, seed=42)
    assert larger_error < error
    with build.program("shot") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        encoding_node = builder.node("encode", hamiltonian=hamiltonian)
        prep_node = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding_node)
        shots_node = builder.node("sample", operator=encoding_node, prep=prep_node,
                                  shots=1000, seed=42)
        shots_node.effects = "random"
    estimate = Executor(molecular_registry()).run(builder.to_ir())[shots_node.id]
    assert estimate.relation == "stochastic_estimate"
    assert estimate.uncertainty.stat_err is not None
    assert abs(estimate.value.value - exact) < 5 * estimate.uncertainty.stat_err


def test_examples_smoke() -> None:
    from examples.h2_pes import main as h2_main
    from examples.lih_active_space import main as lih_main

    h2_main()
    lih_main()
