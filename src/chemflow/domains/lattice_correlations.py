"""Classical site-basis observables and the conventional Allen–Dynes model."""

import math

import numpy as np

from chemflow.impls.fci import _annihilate, _create, sector_states

QUANTITIES = ("spin_spin", "density_density", "pair_s_wave")


def correlation_matrices(vector, sector):
    basis = sector_states(sector)
    vector = np.asarray(vector)
    if (vector.shape != (len(basis),) or not np.isfinite(vector).all() or
            not np.isclose(np.vdot(vector, vector), 1, atol=1e-10, rtol=0)):
        raise ValueError("Correlations require a normalized finite sector vector")
    index = {bits: position for position, bits in enumerate(basis)}

    def expectation(operators):
        value = 0j
        for column, bits in enumerate(basis):
            sign = 1
            for orbital, creation in reversed(operators):
                result = (_create if creation else _annihilate)(bits, orbital)
                if result is None:
                    break
                bits, parity = result
                sign *= parity
            else:
                if bits in index:
                    value += vector[index[bits]].conjugate() * sign * vector[column]
        if abs(value.imag) > 1e-10:
            raise ValueError("Complex correlations are outside the real-matrix contract")
        return float(value.real)

    size = sector.n_orb
    matrices = {key: np.zeros((size, size)) for key in QUANTITIES}
    for left in range(size):
        for right in range(size):
            for first, first_sign in ((left, 0.5), (left + size, -0.5)):
                for second, second_sign in ((right, 0.5), (right + size, -0.5)):
                    density = expectation([(first, True), (first, False),
                                           (second, True), (second, False)])
                    matrices["density_density"][left, right] += density
                    matrices["spin_spin"][left, right] += first_sign * second_sign * density
            matrices["spin_spin"][left, right] += 0.5 * (
                expectation([(left, True), (left + size, False),
                             (right + size, True), (right, False)]) +
                expectation([(left + size, True), (left, False),
                             (right, True), (right + size, False)]))
            matrices["pair_s_wave"][left, right] = expectation([
                (left, True), (left + size, True), (right + size, False), (right, False)])
    return {key: value.tolist() for key, value in matrices.items()}


def validate_selection(site_pair, quantity):
    if quantity not in QUANTITIES:
        raise ValueError("Unknown lattice correlation quantity")
    if site_pair is not None and (not isinstance(site_pair, (list, tuple)) or
            len(site_pair) != 2 or any(type(site) is not int or site < 0 for site in site_pair)):
        raise ValueError("site_pair must contain two nonnegative integer site indices")


def allen_dynes_tc(store=None, *, lambda_ep, omega_log_K, mu_star):
    values = (lambda_ep, omega_log_K, mu_star)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Allen–Dynes inputs must be finite numbers")
    denominator = lambda_ep - mu_star * (1 + 0.62 * lambda_ep)
    if lambda_ep <= 0 or omega_log_K <= 0 or mu_star < 0 or denominator <= 0:
        raise ValueError("Require positive lambda, frequency and pairing denominator; mu* >= 0")
    return {"tc_K": omega_log_K / 1.2 * math.exp(-1.04 * (1 + lambda_ep) / denominator),
            "model": "Allen-Dynes (McMillan)", "unit": "K",
            "input_source": "user or EPW",
            "recommended_domain": "lambda <= 1.5; mu_star approximately 0.1–0.15"}
