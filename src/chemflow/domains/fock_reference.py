"""Independent sector ED oracle for AIMs with at most eight spatial orbitals.

No candidate operator algebra or integral builder is used. Only fixed-spin
determinants are allocated; the ground solve is sparse above 512 determinants.
Particle/hole sectors retain a complete dense spectral resolution.
"""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from chemflow.types import SpectralEstimate
from chemflow.units import Q

MAX_ORBITALS = 8


def _apply(state, mode, create):
    if bool(state & (1 << mode)) == create:
        return None
    return state ^ (1 << mode), (-1) ** (state & ((1 << mode) - 1)).bit_count()


def _sector_states(size, alpha, beta):
    if not 0 <= alpha <= size or not 0 <= beta <= size:
        return []
    left = [sum(1 << orbital for orbital in occupied)
            for occupied in combinations(range(size), alpha)]
    right = [sum(1 << orbital for orbital in occupied)
             for occupied in combinations(range(size), beta)]
    return sorted(first | (second << size) for first in left for second in right)


def _matrix(states, size, onsite, repulsion, hoppings):
    addresses = {state: index for index, state in enumerate(states)}
    rows, columns, values = [], [], []
    for column, state in enumerate(states):
        rows.append(column)
        columns.append(column)
        values.append(sum(onsite[orbital] * ((state >> orbital & 1) +
                      (state >> (size + orbital) & 1)) for orbital in range(size)) +
                      repulsion * (state & 1) * (state >> size & 1))
        for bath, hopping in enumerate(hoppings, start=1):
            for offset in (0, size):
                for source, destination in ((offset, offset + bath), (offset + bath, offset)):
                    removed = _apply(state, source, False)
                    added = _apply(removed[0], destination, True) if removed else None
                    if added:
                        rows.append(addresses[added[0]])
                        columns.append(column)
                        values.append(hopping * removed[1] * added[1])
    return coo_matrix((values, (rows, columns)), shape=(len(states), len(states))).tocsr()


def reference(target, params):
    size = 1 + len(params["eps_bath"])
    if size > MAX_ORBITALS:
        raise ValueError(f"Independent verification supports at most {MAX_ORBITALS} orbitals")
    if target.get("target_spin") is not None or any(
            target["sector"].get(key) is not None for key in ("s", "total_spin")):
        raise ValueError("Independent verification does not resolve total spin")
    scale = Q(1, params.get("unit", "hartree")).to("hartree").value
    repulsion = params["U"] * scale
    onsite = [-repulsion / 2, *(energy * scale for energy in params["eps_bath"])]
    hoppings = [hopping * scale for hopping in params["t_bath"]]
    alpha, beta = target["sector"]["n_alpha"], target["sector"]["n_beta"]
    states = _sector_states(size, alpha, beta)
    if not states or target["sector"]["n_orb"] != size:
        raise ValueError("Invalid verification sector")
    hamiltonian = _matrix(states, size, onsite, repulsion, hoppings)
    if len(states) > 512:
        energies, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-13,
                                 v0=np.random.default_rng(0).normal(size=len(states)))
    else:
        energies, vectors = np.linalg.eigh(hamiltonian.toarray())
    ground_energy = float(energies[0])
    if target["kind"] != "spectral_function":
        return ground_energy
    omega, eta = target["omega"], target["eta"]
    frequencies = np.linspace(omega["start"], omega["stop"], omega["n"])
    axis_scale = Q(1, omega["unit"]).to("hartree").value
    mode = target["orbital"] + (size if target["spin"] == "beta" else 0)
    green = np.zeros(len(frequencies), dtype=complex)
    for delta in (1, -1):
        excited = _sector_states(size, alpha + (delta if target["spin"] == "alpha" else 0),
                                 beta + (delta if target["spin"] == "beta" else 0))
        if not excited:
            continue
        addresses = {state: index for index, state in enumerate(excited)}
        image = np.zeros(len(excited))
        for state, amplitude in zip(states, vectors[:, 0]):
            transformed = _apply(state, mode, delta == 1)
            if transformed:
                image[addresses[transformed[0]]] += transformed[1] * amplitude
        poles, eigenvectors = np.linalg.eigh(
            _matrix(excited, size, onsite, repulsion, hoppings).toarray())
        weights = np.abs(eigenvectors.T @ image) ** 2
        green += (weights[None, :] / (frequencies[:, None] -
                  delta * (poles[None, :] - ground_energy) / axis_scale + 1j * eta)).sum(axis=1)
    return SpectralEstimate(omega=tuple(frequencies), spectral=tuple(-green.imag / np.pi),
                            unit=omega["unit"], relation="equivalent",
                            diagnostics={"ground_energy": ground_energy})
