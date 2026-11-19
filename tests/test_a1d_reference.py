"""Dense tensor-product ED, independent of both determinant implementations."""

import numpy as np
import pytest

from chemflow.domains.fock_reference import reference


def dense_oracle(size, alpha, beta, model, spin, orbital, omega, eta):
    dimension = 2 ** size
    annihilators = []
    for mode in range(size):
        factors = [np.array([[0., 1.], [0., 0.]]) if site == mode else
                   np.diag([1., -1.]) if site < mode else np.eye(2)
                   for site in reversed(range(size))]
        operator = factors[0]
        for factor in factors[1:]:
            operator = np.kron(operator, factor)
        annihilators.append(operator)
    one_spin = -model["U"] / 2 * annihilators[0].T @ annihilators[0]
    for bath, (energy, hopping) in enumerate(zip(model["eps_bath"], model["t_bath"]), 1):
        one_spin += energy * annihilators[bath].T @ annihilators[bath]
        one_spin += hopping * (annihilators[0].T @ annihilators[bath] +
                               annihilators[bath].T @ annihilators[0])
    occupation = annihilators[0].T @ annihilators[0]

    def basis(count):
        return [state for state in range(dimension) if state.bit_count() == count]

    def solve(left, right):
        first, second = basis(left), basis(right)
        kinetic = np.kron(one_spin[np.ix_(first, first)], np.eye(len(second)))
        kinetic += np.kron(np.eye(len(first)), one_spin[np.ix_(second, second)])
        kinetic += model["U"] * np.kron(occupation[np.ix_(first, first)],
                                         occupation[np.ix_(second, second)])
        return np.linalg.eigh(kinetic)

    energies, vectors = solve(alpha, beta)
    green = np.zeros(len(omega), dtype=complex)
    for delta in (1, -1):
        excited_alpha = alpha + (delta if spin == "alpha" else 0)
        excited_beta = beta + (delta if spin == "beta" else 0)
        operator = annihilators[orbital].T if delta == 1 else annihilators[orbital]
        if spin == "alpha":
            transition = np.kron(operator[np.ix_(basis(excited_alpha), basis(alpha))],
                                 np.eye(len(basis(beta))))
        else:
            transition = (-1) ** alpha * np.kron(np.eye(len(basis(alpha))),
                operator[np.ix_(basis(excited_beta), basis(beta))])
        poles, excited = solve(excited_alpha, excited_beta)
        weights = np.abs(excited.T @ transition @ vectors[:, 0]) ** 2
        green += (weights / (omega[:, None] - delta * (poles - energies[0]) + 1j * eta)).sum(1)
    return energies[0], -green.imag / np.pi


@pytest.mark.parametrize("size,spin,orbital", [(5, "alpha", 0), (7, "beta", 2)])
def test_sector_reference_against_direct_dense_ed(size, spin, orbital, monkeypatch):
    from chemflow.domains import algorithms, dynamics

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference must not use candidate primitives")

    for module, name in ((algorithms, "ladder"), (algorithms, "rayleigh_ritz"),
                         (dynamics, "ladder"), (dynamics, "spectral_exact")):
        monkeypatch.setattr(module, name, forbidden)
    model = {"U": 2.7, "eps_bath": np.linspace(-1.3, 1.2, size - 1).tolist(),
             "t_bath": np.linspace(0.3, 0.7, size - 1).tolist(), "unit": "hartree"}
    target = {"kind": "spectral_function", "sector": {"n_alpha": size // 2,
              "n_beta": size // 2, "n_orb": size}, "spin": spin, "orbital": orbital,
              "omega": {"start": -5., "stop": 5., "n": 301, "unit": "hartree"}, "eta": 0.12}
    result = reference(target, model)
    energy, density = dense_oracle(size, size // 2, size // 2, model, spin, orbital,
                                   np.asarray(result.omega), target["eta"])
    assert abs(result.diagnostics["ground_energy"] - energy) < 1e-10
    assert np.trapezoid(np.abs(result.spectral - density), result.omega) < 1e-8
