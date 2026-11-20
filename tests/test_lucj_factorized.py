"""The spin-factorized LUCJ state equals the direct sparse construction."""

import numpy as np
import pytest

from chemflow.impls.lucj import LucjParams, lucj_state, lucj_state_reference
from chemflow.types import Sector


@pytest.mark.parametrize("n_orb,n_alpha,n_beta,layers", [
    (2, 1, 1, 1), (4, 2, 2, 1), (5, 3, 2, 2), (6, 3, 3, 2)])
def test_factorized_state_matches_reference(n_orb, n_alpha, n_beta, layers):
    rng = np.random.default_rng(n_orb * 10 + layers)
    sector = Sector(n_alpha=n_alpha, n_beta=n_beta, n_orb=n_orb)
    params = [LucjParams.random(n_orb, rng, scale=0.4) for _ in range(layers)]
    fast, reference = lucj_state(sector, params), lucj_state_reference(sector, params)
    assert np.allclose(fast, reference, atol=1e-12)
    assert np.linalg.norm(fast) == pytest.approx(1.0, abs=1e-12)
