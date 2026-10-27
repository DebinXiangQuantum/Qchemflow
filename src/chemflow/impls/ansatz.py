from itertools import combinations, combinations_with_replacement

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from chemflow.impls.fci import _annihilate, _create, sector_states
from chemflow.impls.sector import hf_vector
from chemflow.types import Sector


def excitation(sector: Sector, annihilators: tuple[int, ...],
               creators: tuple[int, ...]) -> sparse.csr_matrix:
    states = sector_states(sector)
    positions = {bits: index for index, bits in enumerate(states)}
    rows, columns, values = [], [], []
    for column, bits in enumerate(states):
        phase = 1
        for orbital in annihilators:
            result = _annihilate(bits, orbital)
            if result is None:
                break
            bits, sign = result
            phase *= sign
        else:
            for orbital in creators:
                result = _create(bits, orbital)
                if result is None:
                    break
                bits, sign = result
                phase *= sign
            else:
                if bits in positions:
                    rows.append(positions[bits])
                    columns.append(column)
                    values.append(phase)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(states), len(states)),
                             dtype=float)


def uccsd_pool(sector: Sector, spin_adapted: bool = False
               ) -> tuple[list[str], list[sparse.csr_matrix]]:
    if spin_adapted:
        return _spin_adapted_pool(sector)
    n_orb = sector.n_orb
    occupied = [*range(sector.n_alpha),
                *(n_orb + orbital for orbital in range(sector.n_beta))]
    virtual = [*(orbital for orbital in range(sector.n_alpha, n_orb)),
               *(n_orb + orbital for orbital in range(sector.n_beta, n_orb))]
    groups: dict[str, sparse.csr_matrix] = {}
    for degree in (1, 2):
        for holes in combinations(occupied, degree):
            for particles in combinations(virtual, degree):
                if sorted(index // n_orb for index in holes) != sorted(
                        index // n_orb for index in particles):
                    continue
                operator = excitation(sector, holes, tuple(reversed(particles)))
                generator = operator - operator.T
                if not generator.nnz:
                    continue
                key = f"{holes}->{particles}"
                groups[key] = groups.get(key, sparse.csr_matrix(generator.shape)) + generator
    return list(groups), list(groups.values())


def _spin_adapted_pool(sector: Sector) -> tuple[list[str], list[sparse.csr_matrix]]:
    occupied = range(min(sector.n_alpha, sector.n_beta))
    virtual = range(max(sector.n_alpha, sector.n_beta), sector.n_orb)
    dimension = len(sector_states(sector))
    operators = {(particle, hole): sum(
        (excitation(sector, (hole + spin * sector.n_orb,),
                    (particle + spin * sector.n_orb,)) for spin in (0, 1)),
        sparse.csr_matrix((dimension, dimension)))
        for hole in occupied for particle in virtual}
    names: list[str] = []
    generators: list[sparse.csr_matrix] = []
    basis: list[sparse.csr_matrix] = []

    def add(name: str, operator: sparse.csr_matrix) -> None:
        generator = (operator - operator.T).tocsr()
        residual = generator.copy()
        for vector in basis:
            residual -= residual.multiply(vector).sum() * vector
        norm = np.sqrt(residual.multiply(residual).sum())
        if norm > 1e-10:
            names.append(name)
            generators.append(generator)
            basis.append(residual / norm)

    for hole in occupied:
        for particle in virtual:
            add(f"E{particle},{hole}", operators[particle, hole])
    for first_hole, second_hole in combinations_with_replacement(occupied, 2):
        for first_particle, second_particle in combinations_with_replacement(virtual, 2):
            pairings = [(first_particle, second_particle)]
            if first_hole != second_hole and first_particle != second_particle:
                pairings.append((second_particle, first_particle))
            for left, right in pairings:
                add(f"E{left},{first_hole} E{right},{second_hole}",
                    operators[left, first_hole] @ operators[right, second_hole])
    return names, generators


def uccsd_state(sector: Sector, parameters: np.ndarray,
                generators: list[sparse.csr_matrix]) -> np.ndarray:
    state = hf_vector(sector)
    for parameter, generator in zip(parameters, generators, strict=True):
        if parameter:
            state = expm_multiply(parameter * generator, state)
    return state


def generalized_pool(sector: Sector) -> tuple[list[str], list[sparse.csr_matrix]]:
    orbitals = range(2 * sector.n_orb)
    groups = {}
    for degree in (1, 2):
        for holes in combinations(orbitals, degree):
            for particles in combinations(orbitals, degree):
                if holes >= particles or set(holes) & set(particles):
                    continue
                if sorted(index // sector.n_orb for index in holes) != sorted(
                        index // sector.n_orb for index in particles):
                    continue
                operator = excitation(sector, holes, tuple(reversed(particles)))
                generator = operator - operator.T
                if generator.nnz:
                    groups[f"{holes}->{particles}"] = generator
    return list(groups), list(groups.values())
