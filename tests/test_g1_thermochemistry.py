import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import constants as sc

from chemflow import build
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.check import check
from chemflow.check.static import infer_context
from chemflow.diagnostics import AdapterError
from chemflow.domains import get_pack, load_registry
from chemflow.domains.reactions import identity
from chemflow.domains.thermochemistry import thermochemistry
from chemflow.runtime import Executor
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


def plan_for(quantity=None):
    store = TaskStore("g1")
    record = store.freeze(task(quantity), "test")
    plan, _ = get_pack("reactions").base_plan(record)
    return plan, record, store


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


@pytest.mark.parametrize("quantity", [None, "enthalpy", "gibbs"])
def test_plan_identity_and_enthalpy_path(quantity):
    plan, record, store = plan_for(quantity)
    registry = load_registry(["molecular", "reactions"])
    contexts, issues = infer_context(plan, registry)
    assert not issues and not identity(plan, record, contexts)
    assert sum(node.op == "thermochemistry" for node in plan.nodes) == (2 if quantity else 0)
    result = Executor(registry).run_plan(plan, record, store)
    reference = next(node for node in plan.nodes if node.id == plan.references[0])
    expected = sum(coefficient * (result[reference.inputs[port]].value.value +
        (result[reference.inputs[f"thermo{port.removeprefix('species')}"]][
            "h_corr" if quantity == "enthalpy" else "g_corr"] if quantity else 0))
        for port, coefficient in reference.params["coefficients"].items())
    assert result[reference.id].value.value == pytest.approx(expected)
    assert result[reference.id].kind == {None: "reaction_energy", "enthalpy": "reaction_enthalpy",
                                       "gibbs": "reaction_gibbs_energy"}[quantity]
    assert result[reference.id].relation == "equivalent"
    assert all(gate.passed for gate in result["gates"] if gate.node == reference.id)
    other_plan, other_record, _ = plan_for(None if quantity else "gibbs")
    assert other_plan.hash != plan.hash
    assert identity(plan, other_record, contexts)
    if quantity:
        thermal_node = next(node for node in plan.nodes if node.op == "thermochemistry")
        thermal_node.params["temperature_K"] = 500.
        assert identity(plan, record, contexts)


@pytest.mark.parametrize("condition,value", [("temperature_K", 500.), ("pressure_Pa", 1e5),
                                             ("thermal_model", "other"),
                                             ("frequency_level", "other")])
def test_mixed_thermal_conditions_are_rejected(condition, value):
    from chemflow.domains.reactions import context_check

    plan, _, _ = plan_for("gibbs")
    registry = load_registry(["molecular", "reactions"])
    contexts, _ = infer_context(plan, registry)
    reference = next(node for node in plan.nodes if node.id == plan.references[0])
    inputs = {port: dict(contexts[source]) for port, source in reference.inputs.items()}
    inputs["thermo0"][condition] = value
    issues = context_check(reference, inputs, {}, {node.id: node for node in plan.nodes})
    assert any(issue.code == "CF-E011" and condition in issue.message for issue in issues)


def test_frequency_level_and_geometry_are_checked():
    plan, _, _ = plan_for("gibbs")
    registry = load_registry(["molecular", "reactions"])
    modes = [node for node in plan.nodes if node.op == "frequencies"]
    modes[0].params["level"] = {"method": "uhf"}
    assert any(issue.code == "CF-E011" and "frequency_level" in issue.message
               for issue in check(plan, registry))
    modes[0].inputs["molecule"] = modes[1].inputs["molecule"]
    assert any(issue.code == "CF-E011" and "geometry" in issue.message
               for issue in check(plan, registry))


def test_candidates_keep_thermal_corrections():
    plan, _, _ = plan_for("gibbs")
    expanded, identifiers = Orchestrator._candidate_plan(
        plan, [{"template": "sci", "select_cutoff": .001}])
    assert identifiers
    node = next(node for node in expanded.nodes if node.id in identifiers.values())
    reference = next(node for node in plan.nodes if node.id == plan.references[0])
    for port in ("thermo0", "thermo1"):
        assert node.inputs[port] == reference.inputs[port]
    assert not check(expanded, load_registry(["molecular", "reactions"]))


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


def test_default_operator_parameters():
    with build.program("defaults") as builder:
        model = builder.electronic_model(basis="sto-3g")
        molecule = builder.node("geometry_optimize", model=model,
                                molecule=builder.molecule("H 0 0 0; H 0 0 .74"))
        modes = builder.node("frequencies", molecule=molecule, model=model)
        result = builder.node("thermochemistry", molecule=molecule, model=model, frequencies=modes)
    computed = Executor(load_registry(["molecular", "reactions"])).run(builder.to_ir())
    assert computed[result.id]["temperature_K"] == 298.15
    assert computed[result.id]["pressure_Pa"] == 101325.
