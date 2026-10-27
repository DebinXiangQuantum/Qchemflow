import numpy as np
import pytest

import qcl
from chemflow.domains.fock_reference import reference

MODEL = {"U": 2.0, "eps_bath": [0.2], "t_bath": [0.5], "unit": "hartree"}
SECTOR = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
OMEGA = {"start": -4.0, "stop": 4.0, "n": 201, "unit": "hartree"}


@qcl.workflow
def spectral(model, sector, device, *, orbital=0):
    return qcl.dynamics.spectral_exact(model, sector, orbital=orbital,
                                       spin="alpha", omega=OMEGA, eta=0.1).to(device)


def test_spectral_exact_matches_independent_fock_reference():
    plan = spectral.compile(MODEL, SECTOR, qcl.Device("cpu0", "cpu"))
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        "dynamics.spectral_exact/chemflow-lehmann")
    result = plan.run()
    expected = reference({"kind": "spectral_function", "sector": SECTOR,
                          "orbital": 0, "spin": "alpha", "omega": OMEGA,
                          "eta": 0.1}, MODEL)
    np.testing.assert_allclose(result.spectral, expected.spectral, atol=1e-10)
    assert result.omega == expected.omega
    assert result.relation == "equivalent"
    assert result.provenance["question"]["property"] == "spectral_function"


def test_spectral_rejects_gpu_and_mismatched_sector():
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        spectral.compile(MODEL, SECTOR, qcl.Device("gpu0", "gpu"))
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        spectral.compile(MODEL, {**SECTOR, "n_orb": 3}, qcl.Device("cpu0", "cpu"))
