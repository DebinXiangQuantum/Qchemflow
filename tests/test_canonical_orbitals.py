"""Degenerate canonical orbitals are fixed deterministically, so bitstrings are reproducible."""

import numpy as np
import pytest
from pyscf import gto, scf

from chemflow.impls.pyscf_impls import _from_scf, canonical_orbitals
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types.model import ActiveSpaceSpec, ElectronicModel


@pytest.fixture(scope="module")
def nitrogen():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.098", basis="cc-pvdz", verbose=0)
    return scf.RHF(mol).run()


def scrambled(mf, rng):
    """The same SCF solution with each degenerate block rotated and every sign flipped."""
    coefficients = np.array(mf.mo_coeff, copy=True)
    energies = mf.mo_energy
    start = 0
    for index in range(1, len(energies) + 1):
        if index == len(energies) or abs(energies[index] - energies[index - 1]) > 1e-6:
            size = index - start
            if size > 1:
                rotation, _ = np.linalg.qr(rng.standard_normal((size, size)))
                coefficients[:, start:index] = coefficients[:, start:index] @ rotation
            start = index
    return coefficients * rng.choice([-1.0, 1.0], coefficients.shape[1])


def test_any_solver_choice_maps_to_the_same_orbitals(nitrogen):
    rng = np.random.default_rng(4)
    reference = canonical_orbitals(nitrogen.mol, nitrogen.mo_coeff, nitrogen.mo_energy,
                                   nitrogen.mo_occ)
    for _ in range(3):
        variant = canonical_orbitals(nitrogen.mol, scrambled(nitrogen, rng),
                                     nitrogen.mo_energy, nitrogen.mo_occ)
        assert np.allclose(variant, reference, atol=1e-8)


def test_canonical_orbitals_are_the_same_scf_solution(nitrogen):
    orbitals = canonical_orbitals(nitrogen.mol, nitrogen.mo_coeff, nitrogen.mo_energy,
                                  nitrogen.mo_occ)
    overlap = nitrogen.get_ovlp()
    assert np.allclose(orbitals.T @ overlap @ orbitals, np.eye(orbitals.shape[1]), atol=1e-8)
    fock = orbitals.T @ nitrogen.get_fock() @ orbitals
    assert np.allclose(np.diag(fock), nitrogen.mo_energy, atol=1e-7)
    occupied = nitrogen.mo_occ > 0
    density = 2 * orbitals[:, occupied] @ orbitals[:, occupied].T
    assert np.allclose(density, nitrogen.make_rdm1(), atol=1e-8)


def test_active_space_integrals_do_not_depend_on_the_solver_choice(nitrogen):
    rng = np.random.default_rng(9)
    model = ElectronicModel(basis="cc-pvdz")
    active = ActiveSpaceSpec(n_elec=10, n_orb=8)
    integrals = []
    for coefficients in (nitrogen.mo_coeff, scrambled(nitrogen, rng)):
        mf = scf.RHF(nitrogen.mol)
        mf.__dict__.update(mo_coeff=coefficients, mo_energy=nitrogen.mo_energy,
                           mo_occ=nitrogen.mo_occ, e_tot=nitrogen.e_tot, converged=True)
        store = ArtifactStore()
        data = _from_scf(mf, model, store, active, nitrogen.mol.atom, "angstrom", 0, 0)
        integrals.append(store.get(data.hamiltonian.h1_ref))
        assert np.allclose(mf.mo_coeff, store.get(data.space.coefficients_ref))
    assert np.allclose(integrals[0], integrals[1], atol=1e-8)
