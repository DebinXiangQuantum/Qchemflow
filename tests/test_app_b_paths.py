import math

import numpy as np
import pytest

from chemflow.domains.paths import (
    bond_graph,
    coordinates,
    geometry_transform,
    rate_constant,
)

WATER = {"atoms": "O 0 0 0; H 0 .757 .587; H 0 -.757 .587"}


def test_angle_rigid_side_and_bohr_input():
    moved = geometry_transform(None, WATER, "set_angle", [1, 0, 2], 115)
    _, before = coordinates(WATER)
    _, after = coordinates(moved)
    first, second = after[1] - after[0], after[2] - after[0]
    assert math.degrees(math.acos(first @ second / np.linalg.norm(first) /
                                 np.linalg.norm(second))) == pytest.approx(115)
    assert np.linalg.norm(after[2] - after[0]) == pytest.approx(np.linalg.norm(before[2]))
    assert np.allclose(after[:2], before[:2])
    bohr = {"atoms": "H 0 0 0; H 0 0 1.4", "unit": "bohr"}
    result = geometry_transform(None, bohr, "set_bond", [0, 1], 1.2)
    assert result["unit"] == "angstrom"
    assert coordinates(result)[1][1, 2] == pytest.approx(1.2)


def test_connectivity_handles_tiny_coordinates():
    assert bond_graph({"atoms": "H -1e-12 0 0; H 0 0 .74"}) == {(0, 1)}


@pytest.mark.parametrize("operation,atoms,value,fragment", [
    ("set_bond", [0, 0], 1, None), ("set_angle", [0, 1, 2], 180, None),
    ("set_bond", [0, 3], 1, None), ("set_bond", [0, 1], -1, None),
    ("translate_fragment", [], 1, [0, 1, 2]),
    ("translate_fragment", [], float("nan"), [2]),
    ("set_angle", [1, 0, 2], 100, [1]),
])
def test_invalid_transforms(operation, atoms, value, fragment):
    with pytest.raises(ValueError):
        geometry_transform(None, WATER, operation, atoms, value, fragment)


def test_wigner_supplied_frequency_and_validation():
    bare = rate_constant(None, .03, 298.15)
    corrected = rate_constant(None, .03, 298.15, "wigner", -1100)
    kappa = 1 + (6.62607015e-34 * 29979245800 * 1100 / (1.380649e-23 * 298.15)) ** 2 / 24
    assert corrected["k_per_s"] / bare["k_per_s"] == pytest.approx(kappa)
    with pytest.raises(ValueError, match="wavenumber"):
        rate_constant(None, .03, 298.15, "wigner")
    with pytest.raises(ValueError):
        rate_constant(None, .03, 0)
