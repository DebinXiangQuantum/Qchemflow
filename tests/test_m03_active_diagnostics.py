import pytest

from chemflow import build
from chemflow.check import check
from chemflow.ops.molecular import molecular_registry


@pytest.mark.parametrize(("active", "code"), [
    ({"n_orb": 2}, "CF-E080"),
    ({"n_orb": 2, "n_elec": 5}, "CF-E020"),
    ({"n_orb": 0, "n_elec": 0}, "CF-E020"),
    ({"n_orb": "bad", "n_elec": 2}, "CF-E080"),
])
def test_active_space_errors_are_diagnostics(active: dict, code: str) -> None:
    with build.program("bad_active") as builder:
        builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                            builder.electronic_model(basis="sto-3g"), active=active)
    assert code in {issue.code for issue in check(builder.to_ir(), molecular_registry())}
