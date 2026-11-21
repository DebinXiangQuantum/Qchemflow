from itertools import combinations

import numpy as np
import pytest

from chemflow.domains.algorithms import (
    extend_subspace,
    full_subspace,
    ladder,
    rayleigh_ritz,
    subspace_from_samples,
)
from chemflow.domains.dynamics import impurity_model
from chemflow.impls.fci import matrix
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Samples, Sector, Spectrum, Subspace

AIM = {"U": 2.0, "eps_bath": [-1.2, 0.0, 1.2], "t_bath": [0.5, 0.6, 0.5],
       "unit": "hartree"}
SECTOR = {"n_alpha": 2, "n_beta": 2, "n_orb": 4}
PACKS = ["molecular", "algorithms", "dynamics"]


@pytest.fixture
def model():
    store = ArtifactStore()
    return store, impurity_model(store, **AIM)


def test_subspace_requires_distinct_sector_determinants():
    sector = Sector(**SECTOR)
    for determinants in ((0,), (51, 51), (-1,), (256,), (49,)):
        with pytest.raises(ValueError):
            Subspace(sector=sector, determinants=determinants)
    subspace = Subspace(sector=sector, determinants=(51,))
    assert subspace.coverage == pytest.approx(1 / 36)


@pytest.mark.parametrize("spin", ["alpha", "beta"])
@pytest.mark.parametrize("create", [False, True])
def test_ladder_signs_against_ordered_wedge_basis(spin, create):
    size = 3
    for count in range(2 * size + 1):
        for occupied in combinations(range(2 * size), count):
            state = sum(1 << orbital for orbital in occupied)
            for orbital in range(size):
                mode = orbital + (size if spin == "beta" else 0)
                result = ladder(state, mode, create)
                if (mode in occupied) == create:
                    assert result is None
                    continue
                swaps = sum(occupied_mode < mode for occupied_mode in occupied)
                remaining = set(occupied)
                if create:
                    remaining.add(mode)
                else:
                    remaining.remove(mode)
                assert result == (sum(1 << index for index in remaining), (-1) ** swaps)


def test_extension_is_one_singles_layer_and_has_correct_sector():
    sector = Sector(n_alpha=1, n_beta=1, n_orb=3)
    source = Subspace(sector=sector, determinants=(9,))
    target = {"n_alpha": 2, "n_beta": 1, "n_orb": 3}
    bare = extend_subspace(None, source, target, "create", 1, "alpha")
    assert bare.determinants == (11,)
    extended = extend_subspace(None, source, target, "create", 1, "alpha", "singles")
    assert set(extended.determinants) == {11, 13, 14, 19, 35}
    with pytest.raises(ValueError, match="N±1"):
        extend_subspace(None, source, sector, "create", 1, "alpha")
    with pytest.raises(ValueError, match="Invalid ladder"):
        extend_subspace(None, source, target, "create", 3, "alpha")


def test_projected_hamiltonian_normalisation_ordering_and_interlacing(model):
    store, data = model
    full = full_subspace(store, SECTOR)
    selected = Subspace(sector=full.sector, determinants=full.determinants[:-1])
    full_spectrum = rayleigh_ritz(store, data, full, 36)
    spectrum = rayleigh_ritz(store, data, selected, 35)
    projected = matrix(data.hamiltonian, store, list(selected.determinants)).toarray()
    vectors = store.get(spectrum.vectors_ref)
    energies = np.asarray(spectrum.energies)
    np.testing.assert_allclose(projected, projected.T, atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(35), atol=1e-12)
    np.testing.assert_allclose(projected @ vectors, vectors * energies, atol=1e-11)
    assert np.all(energies >= np.asarray(full_spectrum.energies[:-1]) - 1e-10)
    assert np.all(energies <= np.asarray(full_spectrum.energies[1:]) + 1e-10)
    assert spectrum.coverage == pytest.approx(35 / 36)
    with pytest.raises(ValueError, match="ordered roots"):
        Spectrum.model_validate({**spectrum.model_dump(), "energies": [1.0, -1.0]})


def test_leading_roots_and_complete_resolution_have_explicit_shapes(model):
    store, data = model
    spectrum = rayleigh_ritz(store, data, full_subspace(store, SECTOR), 3)
    assert len(spectrum.energies) == 3
    assert store.get(spectrum.vectors_ref).shape == (36, 3)
    assert store.get(spectrum.resolution_vectors_ref).shape == (36, 36)


def test_configuration_recovery_corrects_particle_counts_and_is_reproducible():
    samples = Samples(counts={"0": 10, "255": 10}, shots=20, encoding_id="sector_state",
                      basis="computational", source="quantum_simulator", seed=7)
    first = subspace_from_samples(None, samples, SECTOR)
    assert first == subspace_from_samples(None, samples, SECTOR)
    assert all((state & 15).bit_count() == (state >> 4).bit_count() == 2
               for state in first.determinants)
    for counts in ({"0": -1}, {"256": 20}, {}):
        with pytest.raises(ValueError):
            subspace_from_samples(None, samples.model_copy(update={"counts": counts}), SECTOR)
