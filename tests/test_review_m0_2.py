"""Reviewer regression tests, round 2 (written by Claude; implementers must not edit)."""
from math import comb

import numpy as np
import pytest

from chemflow.impls.fci import _annihilate, _create, matrix, sector_spin_eigenstate, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def _s2_operator(states: list[int], n_orb: int) -> np.ndarray:
    """Independent S^2 = Sz^2 + Sz + S-S+ in the chemflow determinant basis."""
    index = {bits: i for i, bits in enumerate(states)}
    s2 = np.zeros((len(states), len(states)))
    for column, bits in enumerate(states):
        n_alpha = (bits & ((1 << n_orb) - 1)).bit_count()
        n_beta = (bits >> n_orb).bit_count()
        sz = (n_alpha - n_beta) / 2
        s2[column, column] += sz * sz + sz
        for p in range(n_orb):
            for q in range(n_orb):
                current, sign = bits, 1
                for create, mode in ((False, n_orb + q), (True, q), (False, p), (True, n_orb + p)):
                    step = (_create if create else _annihilate)(current, mode)
                    if step is None:
                        break
                    current, factor = step
                    sign *= factor
                else:
                    if current in index:
                        s2[index[current], column] += sign
    return s2


def test_spin_selection_correct_beyond_symmetric_string_spaces() -> None:
    store = ArtifactStore()
    data = build_molecule("Li 0 0 0; H 0 0 1.6", ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=5))
    states = sector_states(data.sector)
    energies, vectors = np.linalg.eigh(matrix(data.hamiltonian, store, states).toarray())
    s2 = _s2_operator(states, data.sector.n_orb)
    for total_spin in (0, 1):
        expected = next(e for e, v in zip(energies, vectors.T)
                        if abs(v @ s2 @ v - total_spin * (total_spin + 1)) < 1e-6)
        energy, reported = sector_spin_eigenstate(data.hamiltonian, data.sector, store,
                                                  total_spin)
        assert energy == pytest.approx(expected, abs=1e-8)
        assert reported == pytest.approx(total_spin * (total_spin + 1), abs=1e-6)
    assert comb(data.sector.n_orb, data.sector.n_alpha) == 10


@pytest.mark.parametrize("n_orb,n_alpha,n_beta", [(4, 2, 2), (5, 2, 2), (5, 3, 1), (6, 3, 3),
                                                  (6, 2, 3)])
def test_spin_square_in_chemflow_basis_matches_independent_operator(
        n_orb: int, n_alpha: int, n_beta: int) -> None:
    from chemflow.impls.fci import spin_square
    from chemflow.types import Sector

    sector = Sector(n_alpha=n_alpha, n_beta=n_beta, n_orb=n_orb)
    states = sector_states(sector)
    s2 = _s2_operator(states, n_orb)
    rng = np.random.default_rng(7)
    for _ in range(5):
        vector = rng.normal(size=len(states))
        vector /= np.linalg.norm(vector)
        assert spin_square(vector, sector) == pytest.approx(vector @ s2 @ vector, abs=1e-10)
