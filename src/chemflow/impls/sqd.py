from collections import defaultdict
from math import comb

import numpy as np
from pyscf import fci

from chemflow.impls.spin_complete import (
    complete_spin_configurations,
    complete_spin_strings,
    diagonalize_configurations,
)
from chemflow.types import Samples, Sector


def _fix_spin_bits(bits: np.ndarray, target: int, occupancies: np.ndarray,
                   rng: np.random.Generator) -> None:
    occupied = np.flatnonzero(bits)
    vacant = np.flatnonzero(1 - bits)
    if len(occupied) > target:
        bits[rng.choice(occupied, len(occupied) - target, replace=False)] = 0
    elif len(occupied) < target:
        weights = occupancies[vacant]
        weights = weights / weights.sum() if weights.sum() else None
        bits[rng.choice(vacant, target - len(occupied), replace=False, p=weights)] = 1


def _spin_closed(masks: set[int], sector: Sector) -> list[int]:
    if sector.n_alpha != sector.n_beta:
        return sorted(masks)
    half = (1 << sector.n_orb) - 1
    strings = {mask & half for mask in masks} | {mask >> sector.n_orb for mask in masks}
    return sorted(alpha | (beta << sector.n_orb)
                  for alpha in strings for beta in strings)


def recover_configurations(batch, sector, occupancy, rng):
    restored = ((batch[:, None] >> np.arange(2 * sector.n_orb)) & 1).astype(np.uint8)
    for row in restored:
        _fix_spin_bits(row[:sector.n_orb], sector.n_alpha, occupancy[:sector.n_orb], rng)
        _fix_spin_bits(row[sector.n_orb:], sector.n_beta, occupancy[sector.n_orb:], rng)
    return {int(row @ (1 << np.arange(2 * sector.n_orb))) for row in restored}


def product_strings(masks, sector):
    half = (1 << sector.n_orb) - 1
    valid = {mask for mask in masks if (mask & half).bit_count() == sector.n_alpha
             and (mask >> sector.n_orb).bit_count() == sector.n_beta}
    if not valid:
        raise ValueError("No valid configurations after recovery")
    alpha = {mask & half for mask in valid}
    beta = {mask >> sector.n_orb for mask in valid}
    if sector.n_alpha == sector.n_beta:
        alpha |= beta
        beta = alpha
    return alpha, beta


def sqd_sector(data, sector: Sector, store, samples: Samples, *, seed: int = 7,
               n_batches: int = 10, batch_size: int = 3000, max_iter: int = 5,
               carryover_thresh: float = 1e-4, energy_tol: float = 1e-8,
               occ_tol: float = 1e-5, with_ext: bool = True,
               ext_keep: int = 2000, total_spin: int | None = None,
               max_product_dimension: int = 50000,
               spin_completion: str = "configurations", return_wavefunction: bool = False,
               wavefunction_format: str = "dense"):
    if wavefunction_format not in ("dense", "selected_ci"):
        raise ValueError("Unknown wavefunction format")
    if spin_completion not in ("configurations", "product_closure"):
        raise ValueError("Unknown spin completion method")
    if samples.basis != "computational" or n_batches < 1 or max_iter < 1:
        raise ValueError("SQD requires computational samples and positive batches/rounds")
    if (sector.n_alpha, sector.n_beta, sector.n_orb) != (
            data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb):
        raise ValueError("Sector differs from Hamiltonian")
    rng = np.random.default_rng(seed)
    width = 2 * sector.n_orb
    values = np.concatenate([np.full(count, int(mask), dtype=np.int64)
                             for mask, count in samples.counts.items() if count > 0])
    if not len(values) or min(values) < 0 or max(values) >= 1 << width:
        raise ValueError("Invalid configuration counts")
    rng.shuffle(values)
    # Do not bootstrap: batch_size bounds each direct-count batch, never invents shots.
    batches = np.array_split(values, min(n_batches, len(values)))
    batches = [batch[:batch_size] for batch in batches]
    raw = np.concatenate(batches)
    occupancy = ((raw[:, None] >> np.arange(width)) & 1).mean(axis=0)
    carryover: set[int] = set()
    trace = []
    previous_energy = None
    h1 = store.get(data.hamiltonian.h1_ref)
    eri = store.get(data.hamiltonian.h2_ref)
    constant = data.hamiltonian.ledger.total().value
    roots_examined = None
    wavefunction = None

    def export(vector):
        return (vector if wavefunction_format == "selected_ci" else
                fci.selected_ci.to_fci(vector, sector.n_orb,
                                       (sector.n_alpha, sector.n_beta)))

    def diagonalize(masks):
        nonlocal roots_examined, wavefunction
        alpha, beta = product_strings(masks, sector)
        dimension_before = len(alpha) * len(beta)
        completion_reason = None
        if ((total_spin is not None and spin_completion == "configurations") or
                min(sector.n_alpha, sector.n_beta) == 0):
            determinants = complete_spin_configurations(
                {(left, right) for left in alpha for right in beta}, sector.n_alpha)
            result = diagonalize_configurations(
                h1, eri, sector.n_orb, (sector.n_alpha, sector.n_beta),
                determinants, total_spin, return_roots=True,
                return_wavefunction=return_wavefunction)
            energy, probabilities, s2, roots_examined = result[:4]
            if return_wavefunction:
                wavefunction = export(result[4])
            weights = {left | (right << sector.n_orb): probability
                       for (left, right), probability in probabilities.items()}
            return (energy + constant, weights, s2, len(determinants),
                    dimension_before, None)
        if total_spin is not None and spin_completion == "product_closure" and (
                sector.n_alpha == sector.n_beta):
            alpha, completion_reason = complete_spin_strings(
                alpha, sector.n_orb, max_product_dimension)
            beta = alpha
        strings = (np.asarray(sorted(alpha), dtype=np.int64),
                   np.asarray(sorted(beta), dtype=np.int64))
        solver = fci.selected_ci.SelectedCI()
        solver.verbose = 0
        electrons = (sector.n_alpha, sector.n_beta)
        energy, vector = solver.kernel_fixed_space(h1, eri, sector.n_orb, electrons, strings)
        if return_wavefunction:
            wavefunction = export(vector)
        s2 = float(solver.spin_square(vector, sector.n_orb, electrons)[0])
        if total_spin is not None:
            determinants = {(int(left), int(right)) for left in strings[0]
                            for right in strings[1]}
            result = diagonalize_configurations(
                h1, eri, sector.n_orb, electrons, determinants, total_spin,
                return_roots=True, return_wavefunction=return_wavefunction)
            energy, probabilities, s2, roots_examined = result[:4]
            if return_wavefunction:
                wavefunction = export(result[4])
            weights = {int(left | (right << sector.n_orb)): float(probability)
                       for (left, right), probability in probabilities.items()}
            return (energy + constant, weights, s2, len(determinants),
                    dimension_before, completion_reason)
        weights = {int(left | (right << sector.n_orb)): float(abs(vector[i, j]) ** 2)
                   for i, left in enumerate(strings[0]) for j, right in enumerate(strings[1])}
        return (float(energy) + constant, weights, s2,
                len(strings[0]) * len(strings[1]), dimension_before, completion_reason)

    for iteration in range(1, max_iter + 1):
        energies = []
        probabilities = defaultdict(float)
        next_occupancy = np.zeros(width)
        union = set(carryover)
        for batch in batches:
            masks = recover_configurations(batch, sector, occupancy, rng)
            union |= masks
            energy, vector, _, _, _, _ = diagonalize(masks | carryover)
            energies.append(energy)
            for state, probability in vector.items():
                if probability:
                    probabilities[state] += probability / len(batches)
                    next_occupancy += probability / len(batches) * (
                        (state >> np.arange(width)) & 1)
        carryover |= {state for state, probability in probabilities.items()
                      if probability > carryover_thresh}
        energy, vector, s2, dimension, before, reason = diagonalize(union | carryover)
        delta_occupancy = float(np.mean(np.abs(next_occupancy - occupancy)))
        trace.append({"iteration": iteration, "batch_energies": energies,
                      "batch_spread": float(max(energies) - min(energies)),
                      "energy": energy, "subspace_dimension": dimension,
                      "dimension_before_completion": before,
                      "dimension_after_completion": dimension,
                      "sector_fraction": dimension / (comb(sector.n_orb, sector.n_alpha) *
                                                       comb(sector.n_orb, sector.n_beta)),
                      "spin_completion": spin_completion if total_spin is not None else None,
                      "spin_completion_reason": reason,
                      "carryover_size": len(carryover),
                      "delta_occupancy": delta_occupancy})
        if (previous_energy is not None and abs(energy - previous_energy) < energy_tol
                and delta_occupancy < occ_tol):
            break
        previous_energy, occupancy = energy, next_occupancy

    sqd_energy = energy
    if with_ext:
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)[:ext_keep]
        augmented = set(union | carryover)
        for state in ranked:
            for offset in (0, sector.n_orb):
                for occupied in range(offset, offset + sector.n_orb):
                    if not state & (1 << occupied):
                        continue
                    for vacant in range(offset, offset + sector.n_orb):
                        if not state & (1 << vacant):
                            augmented.add(state ^ (1 << occupied) ^ (1 << vacant))
        energy, vector, s2, dimension, before, reason = diagonalize(augmented)
    diagnostics = {"s2": s2, "seed": seed, "n_samples": samples.shots,
                    "determinants_ref": store.put(np.asarray(sorted(vector), dtype=np.int64)),
                    "subspace_dimension": dimension, "iterations": iteration,
                    "dimension_before_completion": before,
                    "dimension_after_completion": dimension,
                    "sector_fraction": dimension / (comb(sector.n_orb, sector.n_alpha) *
                                                     comb(sector.n_orb, sector.n_beta)),
                    "spin_completion": spin_completion if total_spin is not None else None,
                    "spin_completion_reason": reason,
                    "n_roots_examined": roots_examined,
                    "batch_energies": trace[-1]["batch_energies"],
                    "batch_spread": trace[-1]["batch_spread"], "trace": trace,
                    "energy_sqd": sqd_energy}
    return (energy, diagnostics, wavefunction) if return_wavefunction else (energy, diagnostics)
