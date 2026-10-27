from itertools import combinations

import numpy as np
from pyscf.fci import direct_spin1, selected_ci
from scipy.sparse.linalg import LinearOperator, eigsh


def complete_spin_configurations(determinants: set[tuple[int, int]],
                                 n_alpha: int) -> set[tuple[int, int]]:
    """Complete each spatial occupancy pattern across spin assignments."""
    completed = set(determinants)
    for alpha, beta in determinants:
        doubly = alpha & beta
        open_shells = alpha ^ beta
        alpha_open = n_alpha - doubly.bit_count()
        orbitals = [1 << index for index in range(open_shells.bit_length())
                    if open_shells & (1 << index)]
        for chosen in combinations(orbitals, alpha_open):
            alpha_mask = doubly | sum(chosen)
            completed.add((alpha_mask, doubly | (open_shells ^ sum(chosen))))
    return completed


def diagonalize_configurations(h1: np.ndarray, eri: np.ndarray, n_orb: int,
                               electrons: tuple[int, int],
                               determinants: set[tuple[int, int]],
                               total_spin: int | None = None, *, return_roots: bool = False,
                               return_wavefunction: bool = False
                               ) -> tuple[float, dict[tuple[int, int], float], float] | tuple:
    """Diagonalize the Hamiltonian projected onto a determinant list."""
    ordered = sorted(determinants)
    alpha = np.asarray(sorted({pair[0] for pair in ordered}), dtype=np.int64)
    beta = np.asarray(sorted({pair[1] for pair in ordered}), dtype=np.int64)
    strings = (alpha, beta)
    alpha_index = {int(mask): index for index, mask in enumerate(alpha)}
    beta_index = {int(mask): index for index, mask in enumerate(beta)}
    rows = np.asarray([alpha_index[pair[0]] for pair in ordered])
    columns = np.asarray([beta_index[pair[1]] for pair in ordered])
    shape = (len(alpha), len(beta))
    effective_eri = direct_spin1.absorb_h1e(h1, eri, n_orb, electrons, 0.5)
    links = selected_ci._all_linkstr_index(strings, n_orb, electrons)

    def embed(vector):
        product = np.zeros(shape)
        product[rows, columns] = vector
        return selected_ci._as_SCIvector(product, strings)

    def contract(vector):
        if 0 in electrons:
            full = selected_ci.to_fci(embed(vector), n_orb, electrons)
            contracted = selected_ci.from_fci(
                direct_spin1.contract_2e(effective_eri, full, n_orb, electrons),
                strings, n_orb, electrons)
        else:
            contracted = selected_ci.contract_2e(effective_eri, embed(vector), n_orb,
                                                 electrons, link_index=links)
        return np.asarray(contracted)[rows, columns]

    dimension = len(ordered)
    if dimension <= 8:
        matrix = np.column_stack([contract(basis) for basis in np.eye(dimension)])
        energies, vectors = np.linalg.eigh(matrix)
        roots = dimension
    else:
        operator = LinearOperator((dimension, dimension), matvec=contract, dtype=float)
        roots = min(4, dimension - 1)
        while True:
            energies, vectors = eigsh(operator, k=roots, which="SA", tol=1e-10)
            candidates = []
            for index in range(roots):
                embedded = embed(vectors[:, index])
                s2 = float(selected_ci.spin_square(embedded, n_orb, electrons)[0])
                if total_spin is None or abs(s2 - total_spin * (total_spin + 1)) < 1e-8:
                    candidates.append((index, s2))
            if candidates or roots == dimension - 1:
                break
            roots = min(2 * roots, dimension - 1)
        if not candidates:
            matrix = np.column_stack([contract(basis) for basis in np.eye(dimension)])
            energies, vectors = np.linalg.eigh(matrix)
            roots = dimension
    candidates = []
    for index in range(roots):
        s2 = float(selected_ci.spin_square(embed(vectors[:, index]), n_orb, electrons)[0])
        if total_spin is None or abs(s2 - total_spin * (total_spin + 1)) < 1e-8:
            candidates.append((index, s2))
    if not candidates:
        raise ValueError("No eigenstate matches the requested spin")
    index, s2 = min(candidates, key=lambda item: energies[item[0]])
    result = (float(vectors[:, index] @ contract(vectors[:, index])),
              dict(zip(ordered, np.abs(vectors[:, index]) ** 2)), s2)
    if return_roots:
        result = (*result, roots)
    return (*result, embed(vectors[:, index])) if return_wavefunction else result


def complete_spin_strings(strings: set[int], n_orb: int,
                          max_product_dimension: int = 50000) -> tuple[set[int], str | None]:
    """Close a shared alpha/beta string set under spin-exchange coupling."""
    if max_product_dimension < 1:
        raise ValueError("max_product_dimension must be positive")
    completed = set(strings)
    if len(completed) ** 2 > max_product_dimension:
        return set(strings), "spin completion exceeds max_product_dimension"
    pending = list(completed)
    index = 0
    while index < len(pending):
        left = pending[index]
        index += 1
        for right in tuple(completed):
            left_only = left & ~right
            right_only = right & ~left
            while left_only:
                bit_left = left_only & -left_only
                left_only -= bit_left
                remaining = right_only
                while remaining:
                    bit_right = remaining & -remaining
                    remaining -= bit_right
                    for string in (left ^ bit_left ^ bit_right,
                                   right ^ bit_left ^ bit_right):
                        if string not in completed:
                            if (len(completed) + 1) ** 2 > max_product_dimension:
                                return set(strings), "spin completion exceeds max_product_dimension"
                            completed.add(string)
                            pending.append(string)
    return completed, None
