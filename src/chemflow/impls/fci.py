from itertools import combinations
from math import comb

import numpy as np
from pyscf import fci
from scipy import sparse
from scipy.sparse.linalg import eigsh

from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian, Sector


def _annihilate(bits: int, index: int) -> tuple[int, int] | None:
    if not bits & (1 << index):
        return None
    return bits ^ (1 << index), (-1) ** ((bits & ((1 << index) - 1)).bit_count())


def _create(bits: int, index: int) -> tuple[int, int] | None:
    if bits & (1 << index):
        return None
    return bits | (1 << index), (-1) ** ((bits & ((1 << index) - 1)).bit_count())


def sector_states(sector: Sector) -> list[int]:
    return [sum(1 << orbital for orbital in alpha) |
            sum(1 << (orbital + sector.n_orb) for orbital in beta)
            for alpha in combinations(range(sector.n_orb), sector.n_alpha)
            for beta in combinations(range(sector.n_orb), sector.n_beta)]


def spin_square(vector: np.ndarray, sector: Sector) -> float:
    """Evaluate S² after permuting chemflow determinants to PySCF CI string order."""
    alpha_count = comb(sector.n_orb, sector.n_alpha)
    beta_count = comb(sector.n_orb, sector.n_beta)
    alpha_order, beta_order = ci_string_order(sector)
    ci = np.empty((alpha_count, beta_count), dtype=np.asarray(vector).dtype)
    ci[np.ix_(alpha_order, beta_order)] = np.asarray(vector).reshape(alpha_count, beta_count)
    return float(fci.spin_op.spin_square(ci, sector.n_orb,
                                        (sector.n_alpha, sector.n_beta))[0])


def ci_string_order(sector: Sector) -> tuple[list[int], list[int]]:
    return tuple([
        fci.cistring.str2addr(sector.n_orb, electrons,
                              sum(1 << orbital for orbital in occupied))
        for occupied in combinations(range(sector.n_orb), electrons)]
        for electrons in (sector.n_alpha, sector.n_beta))


def matrix(hamiltonian: Hamiltonian, store: ArtifactStore,
           states: list[int]) -> sparse.csr_matrix:
    """Chemist spatial (pq|rs): 1/2 (pq|rs) a†p a†r as aq; alpha block first."""
    h1 = store.get(hamiltonian.h1_ref)
    eri = store.get(hamiltonian.h2_ref)
    n_orb = hamiltonian.n_orb
    index = {bits: position for position, bits in enumerate(states)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def append(bits: int, target: int, coefficient: float) -> None:
        if bits in index and abs(coefficient) > 1e-14:
            rows.append(index[bits])
            columns.append(target)
            values.append(coefficient)

    for position, bits in enumerate(states):
        append(bits, position, hamiltonian.ledger.total().value)
        for annihilator in range(2 * n_orb):
            first = _annihilate(bits, annihilator)
            if first is None:
                continue
            after_first, sign_first = first
            for creator in range(2 * n_orb):
                if creator // n_orb != annihilator // n_orb:
                    continue
                created = _create(after_first, creator)
                if created is not None:
                    after_created, sign_created = created
                    append(after_created, position, sign_first * sign_created *
                           h1[creator % n_orb, annihilator % n_orb])
            for second_annihilator in range(2 * n_orb):
                second = _annihilate(after_first, second_annihilator)
                if second is None:
                    continue
                after_second, sign_second = second
                for first_creator in range(2 * n_orb):
                    if first_creator // n_orb != annihilator // n_orb:
                        continue
                    for second_creator in range(2 * n_orb):
                        if second_creator // n_orb != second_annihilator // n_orb:
                            continue
                        middle = _create(after_second, second_creator)
                        if middle is None:
                            continue
                        last = _create(middle[0], first_creator)
                        if last is None:
                            continue
                        coefficient = (0.5 * eri[first_creator % n_orb, annihilator % n_orb,
                                                 second_creator % n_orb,
                                                 second_annihilator % n_orb] * sign_first *
                                       sign_second * middle[1] * last[1])
                        append(last[0], position, coefficient)
    return sparse.coo_matrix((values, (rows, columns)),
                             shape=(len(states), len(states))).tocsr()


def sector_diag_numpy(hamiltonian: Hamiltonian, sector: Sector, store: ArtifactStore) -> float:
    operator = matrix(hamiltonian, store, sector_states(sector))
    if operator.shape[0] <= 32:
        return float(np.linalg.eigvalsh(operator.toarray())[0])
    return float(eigsh(operator, k=1, which="SA", return_eigenvectors=False)[0])


def sector_spin_eigenstate(hamiltonian: Hamiltonian, sector: Sector, store: ArtifactStore,
                           total_spin: int | None = None, *, return_wavefunction=False) -> tuple:
    operator = matrix(hamiltonian, store, sector_states(sector)).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    for energy, vector in zip(eigenvalues, eigenvectors.T):
        spin_squared = spin_square(vector, sector)
        if total_spin is None or abs(spin_squared - total_spin * (total_spin + 1)) < 1e-5:
            result = (float(energy), spin_squared)
            return (*result, vector) if return_wavefunction else result
    from chemflow.diagnostics import AdapterError

    raise AdapterError("CF-E020", None, "Requested spin state unavailable",
                       "Choose a feasible total spin")
