import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import constants as sc

from chemflow.agent.task import TaskSpec
from chemflow.diagnostics import AdapterError
from chemflow.domains.thermochemistry import thermochemistry
from chemflow.signing import TaskSigner, generate_keypair
from chemflow.types import ElectronicModel

WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"


def thermal(modes=(1000., 1600., 3600.), **params):
    return thermochemistry(None, {"atoms": WATER, **params.pop("molecule", {})},
        ElectronicModel(basis="sto-3g"),
        {"freq_wavenumber": modes, "max_gradient": params.pop("gradient", 0.)}, **params)


def task(quantity=None):
    return TaskSpec.model_validate({
        "question": "H2 geometry relaxation thermochemistry", "target": {
            "kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
            "species": [{"name": name, "molecule": {"atoms": f"H 0 0 0; H 0 0 {distance}"},
                         "coefficient": coefficient, "active": {"n_elec": 2, "n_orb": 2},
                         "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0}
                        for name, distance, coefficient in (("a", .74, -1), ("b", .8, 1))],
            **({"thermo": {"quantity": quantity}} if quantity else {})},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": .0016, "unit": "hartree"}],
        "budget": {"wall_s": 600}, "allowed_effects": ["pure"], "decision": "compare"})


def test_ts_excludes_and_reports_one_imaginary_mode():
    result = thermal((-500., 1600., 3600.), molecule={"stationary_kind": "ts"})
    positive = thermal((1600., 3600.))
    assert result["excluded_imaginary_modes"] == [-500.]
    assert result["n_imaginary"] == 1
    for key in ("zpe", "u_corr", "h_corr", "s", "g_corr"):
        assert result[key] == positive[key]
    for modes, kind in (((-500., 1600.), "minimum"), ((-500., -100.), "ts")):
        with pytest.raises(AdapterError, match="CF-E020.*stationary"):
            thermal(modes, molecule={"stationary_kind": kind})
    with pytest.raises(AdapterError, match="CF-E020.*stationary"):
        thermal(gradient=1.01e-4)


def test_temperature_pressure_and_symmetry_dependence():
    baseline = thermal(symmetry_number=1)
    hot = thermal(temperature_K=500., symmetry_number=1)
    dense = thermal(pressure_Pa=202650., symmetry_number=1)
    symmetric = thermal(symmetry_number=2)
    boltzmann = sc.k / sc.physical_constants["Hartree energy"][0]
    assert hot["zpe"] == baseline["zpe"]
    assert hot["u_corr"] > baseline["u_corr"]
    assert hot["s"] > baseline["s"]
    assert hot["g_corr"] < baseline["g_corr"]
    for result in (dense, symmetric):
        assert result["s"] - baseline["s"] == pytest.approx(-boltzmann * np.log(2))
        assert result["g_corr"] - baseline["g_corr"] == pytest.approx(
            298.15 * boltzmann * np.log(2))
        assert result["h_corr"] == baseline["h_corr"]
    assert symmetric["symmetry_number"] == 2
    assert baseline["h_corr"] - baseline["u_corr"] == pytest.approx(298.15 * boltzmann)


@pytest.mark.parametrize("params", [{"temperature_K": 0}, {"pressure_Pa": -1},
                                    {"symmetry_number": 0}, {"symmetry_number": 1.5}])
def test_invalid_thermal_parameters(params):
    with pytest.raises(ValueError):
        thermal(**params)


@pytest.mark.parametrize("quantity,label", [("enthalpy", "ΔH(298.15 K)"),
                                           ("gibbs", "ΔG(298.15 K, 1 atm)")])
def test_agent_cli_thermal_reaction(tmp_path, quantity, label):
    private, public = generate_keypair(b"test")
    record = TaskSigner(private, b"test").freeze(task(quantity), "test")
    task_path, key_path, output = tmp_path / "task.json", tmp_path / "key.pem", tmp_path / "out"
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public)
    completed = subprocess.run([str(Path(sys.executable).parent / "chemflow"), "agent", "run",
        "--task", str(task_path), "--public-key", str(key_path), "--out", str(output)],
        check=False, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert completed.returncode == 0, completed.stderr
    report = (output / "report.md").read_text()
    assert label in report and "hartree" in report
    summary = json.loads((output / "summary.json").read_text())
    assert summary["record_hash"] == record.hash
    assert summary["results"]


def test_atomic_electronic_degeneracy():
    result = thermochemistry(None, {"atoms": "H 0 0 0", "spin": 1},
        ElectronicModel(basis="sto-3g"), {"freq_wavenumber": [], "max_gradient": 0.})
    assert result["spin_multiplicity"] == 2
    assert result["zpe"] == 0.
    assert result["h_corr"] / result["u_corr"] == pytest.approx(5 / 3)
