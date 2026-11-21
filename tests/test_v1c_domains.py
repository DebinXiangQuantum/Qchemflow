import copy

import pytest
from pydantic import ValidationError

from chemflow.agent.task import TaskSpec
from chemflow.ids import content_hash


def specification(kind="lattice"):
    sector = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
    targets = {
        "lattice": {"kind": "lattice_ground_energy", "lattice": {
            "model": "hubbard", "n_sites": 2, "t": 1., "U": 2., "unit": "hartree"},
            "sector": sector},
        "impurity": {"kind": "spectral_function", "model": {"impurity_model": {
            "U": 2., "eps_bath": [0.], "t_bath": [.5], "unit": "hartree"}},
            "sector": sector, "orbital": 0, "spin": "alpha",
            "omega": {"start": -4., "stop": 4., "n": 21, "unit": "hartree"}, "eta": .1},
        "molecular": {"molecule": {"atoms": "H 0 0 0; H 0 0 .74"},
            "electronic_model": {"basis": "sto-3g"}, "sector": sector,
            "active": {"n_elec": 2, "n_orb": 2}, "target_spin": 0}}
    spectral = kind == "impurity"
    axis = {"lattice": "hubbard_u", "impurity": "impurity_u",
            "molecular": "geometry_scale"}[kind]
    return {"question": "Family regression", "target": targets[kind],
            "gates": [{"kind": "l1_error_vs_exact" if spectral else "abs_error_vs_exact",
                       "threshold": .0016, "unit": "dimensionless" if spectral else "hartree"}],
            "budget": {"wall_s": 300}, "allowed_effects": ["pure", "simulator"],
            "decision": "d", "validity": [{"axis": axis,
                "range": [.7, 2.5] if kind == "molecular" else [0., 4.],
                "points": 3, "resolution": .1}]}


@pytest.mark.parametrize("change", [
    {"points": 2}, {"points": 34}, {"points": 3.5}, {"points": True},
    {"range": [2., 2.]}, {"range": [4., 0.]}, {"range": [3., 4.]},
    {"range": [0., float("inf")]}, {"resolution": 0.}, {"resolution": 4.},
    {"resolution": float("nan")}, {"axis": "geometry_scale"}, {"extra": 1},
    {"range": [False, 4.]}, {"resolution": ".1"}])
def test_domain_validation(change):
    spec = specification()
    spec["validity"][0].update(change)
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(spec)


def test_signed_fields_are_not_dropped():
    spec = specification()
    original = TaskSpec.model_validate(spec)
    changed = copy.deepcopy(spec)
    changed["validity"][0]["resolution"] = .05
    assert content_hash(original) != content_hash(TaskSpec.model_validate(changed))
    for payload in (spec | {"unknown": 1},
                    spec | {"target": spec["target"] | {"unknown": 1}}):
        with pytest.raises(ValidationError):
            TaskSpec.model_validate(payload)
