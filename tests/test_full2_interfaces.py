from copy import deepcopy

import numpy as np

from chemflow.domains.fock_reference import reference

MODEL = {"U": 2., "eps_bath": [-1.], "t_bath": [.5], "unit": "hartree"}
TARGET = {"kind": "spectral_function", "model": {"impurity_model": MODEL},
          "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
          "orbital": 0, "spin": "alpha",
          "omega": {"start": -4., "stop": 4., "n": 101, "unit": "hartree"}, "eta": .1}


def test_independent_atomic_lehmann_reference():
    target = deepcopy(TARGET)
    target["sector"] = {"n_alpha": 1, "n_beta": 0, "n_orb": 1}
    model = {"U": 2., "eps_bath": [], "t_bath": [], "unit": "hartree"}
    result = reference(target, model)
    frequencies = np.asarray(result.omega)
    expected = .1 / (np.pi * ((frequencies + 1) ** 2 + .1 ** 2))
    assert np.allclose(result.spectral, expected, atol=1e-12)
