"""Every catalogue application names a QCL program that defines a workflow."""

import importlib.util
from pathlib import Path

import qcl
from chemflow.apps import catalog

ROOT = Path(__file__).resolve().parents[1]
Workflow = type(qcl.workflow(lambda: None))


def test_every_catalogue_application_has_a_qcl_program():
    for entry in catalog():
        path = ROOT / entry["qcl_program"]
        assert path.is_file(), entry["id"]
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert any(isinstance(value, Workflow) for value in vars(module).values()), path


def test_ground_state_energy_program_returns_exact_and_sqd(tmp_path, monkeypatch):
    path = ROOT / "qcl/apps/ground_state_energy.py"
    spec = importlib.util.spec_from_file_location("ground_state_energy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1000\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    simulator = qcl.Device("sim0", "qpu", qubits=8, simulator_host="cpu0",
                           simulator_method="statevector")
    result = module.ground_state_energy.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), qcl.Active(2, 2),
        qcl.Cluster.local().cpus[0], simulator, shots=200, strict_transfers=True).run()
    assert abs(result["casci"].value - (-1.1372838344885)) < 1e-8
    assert abs(result["sqd"].value - result["casci"].value) < 1e-8
    assert result["hf"].value > result["casci"].value
