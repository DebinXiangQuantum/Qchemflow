import itertools

import numpy as np
import pytest

from chemflow.agent.task import TaskStore
from chemflow.domains.materials import lattice_arrays
from chemflow.relations import TargetRelation, combine_difference, combine_linear
from chemflow.units import Q


def freeze(spec):
    store = TaskStore("m6a-tests")
    return store.freeze(spec, "test"), store


def test_linear_relation_table():
    for left, right in itertools.product(TargetRelation, repeat=2):
        assert combine_linear([(1, left), (-1, right)]) == combine_difference(left, right)
    exact, upper = TargetRelation.equivalent, TargetRelation.variational_upper_bound
    assert combine_linear([(-3, exact), (2, upper), (1, upper)]) == upper
    assert combine_linear([(0, upper)]) == TargetRelation.approximate


def test_square_lattice_integrals_and_invalid_geometry():
    h1, eri = lattice_arrays(geometry="square", shape=[2, 3], t=2, U=5, unit="eV")
    assert h1.shape == (6, 6)
    assert np.count_nonzero(h1) == 14
    assert h1[0, 3] == pytest.approx(-Q(2, "eV").to("hartree").value)
    assert h1[2, 3] == 0
    assert eri[4, 4, 4, 4] == pytest.approx(Q(5, "eV").to("hartree").value)
    assert np.count_nonzero(eri) == 6
    for params in ({"geometry": "square", "shape": [2, 2], "n_sites": 3},
                   {"geometry": "ring", "n_sites": 0}, {"geometry": "triangle"}):
        with pytest.raises(ValueError):
            lattice_arrays(**params)
