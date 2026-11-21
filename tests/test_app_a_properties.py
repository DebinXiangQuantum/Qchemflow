from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemflow.agent.task import FrontierTarget, IonizationTarget

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def record(target):
    return SimpleNamespace(spec=SimpleNamespace(target=target))


def frontier(**changes):
    return FrontierTarget.model_validate({"kind": "frontier_orbitals",
        "molecule": {"atoms": WATER}, "electronic_model": {"basis": "6-31g"},
        "level": {"method": "hf", "functional": None, "dispersion": None},
        "quantity": "gap", "spin": "alpha", **changes})


def ionization(**changes):
    return IonizationTarget.model_validate({"kind": "ionization_energy",
        "molecule": {"atoms": "Li 0 0 0; H 0 0 1.6"},
        "electronic_model": {"basis": "sto-3g"}, "final_spin": 0.5, **changes})


@pytest.mark.parametrize("changes", [{"level": {"method": "dft"}},
    {"level": {"method": "hf", "dispersion": "d4"}}, {"spin": "gamma"},
    {"quantity": "ip"}, {"unexpected": True}, {"level": {"method": "gfn2"}}])
def test_frontier_rejects_unsupported_schema(changes):
    with pytest.raises(ValidationError):
        frontier(**changes)
