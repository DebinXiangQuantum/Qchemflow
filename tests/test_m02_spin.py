import numpy as np
import pytest

from chemflow.impls.fci import sector_states, spin_square
from chemflow.types import Sector


def test_spin_square_polarized_determinant() -> None:
    sector = Sector(n_alpha=2, n_beta=0, n_orb=5)
    vector = np.zeros(len(sector_states(sector)))
    vector[0] = 1
    assert spin_square(vector, sector) == pytest.approx(2)


def test_spin_square_singlet_determinant() -> None:
    sector = Sector(n_alpha=1, n_beta=1, n_orb=5)
    vector = np.zeros(len(sector_states(sector)))
    vector[0] = 1
    assert spin_square(vector, sector) == pytest.approx(0)
