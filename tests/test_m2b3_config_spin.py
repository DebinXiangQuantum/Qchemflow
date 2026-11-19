import numpy as np
from pyscf import fci

from chemflow.impls.spin_complete import complete_spin_configurations, diagonalize_configurations


def test_configuration_completion_commutes_with_spin_square():
    determinants = {(3, 5), (6, 9), (3, 3)}
    completed = complete_spin_configurations(determinants, 2)
    assert determinants <= completed
    assert len(completed) < 10 * 10
    vector = np.zeros((10, 10))
    rng = np.random.default_rng(42)
    for alpha, beta in completed:
        vector[fci.cistring.str2addr(5, 2, alpha),
               fci.cistring.str2addr(5, 2, beta)] = rng.normal()
    transformed = fci.spin_op.contract_ss(vector, 5, (2, 2))
    outside = np.ones_like(vector, dtype=bool)
    for alpha, beta in completed:
        outside[fci.cistring.str2addr(5, 2, alpha),
                fci.cistring.str2addr(5, 2, beta)] = False
    assert np.max(np.abs(transformed[outside])) < 1e-10


def test_projected_solver_matches_dense_and_resolves_spin():
    rng = np.random.default_rng(7)
    n_orb = 3
    h1 = rng.normal(size=(n_orb, n_orb))
    h1 = h1 + h1.T
    eri = np.zeros((n_orb, n_orb, n_orb, n_orb))
    for index in range(n_orb):
        eri[index, index, index, index] = 0.7
    completed = complete_spin_configurations({(1, 2), (1, 1)}, 1)
    strings = fci.cistring.make_strings(range(n_orb), 1)
    effective = fci.direct_spin1.absorb_h1e(h1, eri, n_orb, (1, 1), 0.5)
    full = np.column_stack([fci.direct_spin1.contract_2e(effective, basis.reshape(3, 3),
                                                          n_orb, (1, 1)).ravel()
                            for basis in np.eye(9)])
    indices = [fci.cistring.str2addr(n_orb, 1, alpha) * len(strings) +
               fci.cistring.str2addr(n_orb, 1, beta) for alpha, beta in sorted(completed)]
    dense_energies, dense_vectors = np.linalg.eigh(full[np.ix_(indices, indices)])
    for spin in (0, 1):
        candidates = []
        for index, energy in enumerate(dense_energies):
            vector = np.zeros(9)
            vector[indices] = dense_vectors[:, index]
            s2 = fci.spin_op.spin_square(vector.reshape(3, 3), n_orb, (1, 1))[0]
            if abs(s2 - spin * (spin + 1)) < 1e-10:
                residual = fci.spin_op.contract_ss(vector.reshape(3, 3), n_orb, (1, 1))
                assert np.linalg.norm(residual - spin * (spin + 1) * vector.reshape(3, 3)) < 1e-10
                candidates.append(energy)
        energy, weights, s2 = diagonalize_configurations(h1, eri, n_orb, (1, 1),
                                                         completed, spin)
        assert abs(energy - min(candidates)) < 1e-10
        assert abs(s2 - spin * (spin + 1)) < 1e-10
        assert abs(sum(weights.values()) - 1) < 1e-10
