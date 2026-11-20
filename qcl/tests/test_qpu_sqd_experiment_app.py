"""The hardware SQD experiment compiles to QPU + simulator + CPU/GPU plans; SQD ablations."""

import importlib.util
from pathlib import Path

import pytest

import qcl

APP = Path(__file__).resolve().parents[1] / "apps" / "qpu_sqd_experiment.py"
CPU = qcl.Device("cpu0", "cpu", threads=8)
GPUS = [qcl.Device(f"gpu{index}", "gpu", memory_gb=32.0) for index in range(2)]
CLOUD = qcl.Device("mq02", "qpu", qubits=24, site="remote", billing="metered",
                   provider="lqcloud", backend="MQ02")
SIMULATOR = qcl.Device("sim-ideal", "qpu", qubits=20, simulator_host="gpu0",
                       simulator_method="statevector")
CPU_SIMULATOR = qcl.Device("sim0", "qpu", qubits=20, simulator_host="cpu0",
                           simulator_method="statevector")


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("qpu_sqd_experiment", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_point_samples_hardware_and_simulator_and_runs_four_sqds(app):
    molecules = tuple(qcl.Molecule(f"N 0 0 0; N 0 0 {r}", "cc-pvdz") for r in (1.1, 2.2))
    plan = app.hardware_sqd.compile(molecules, CPU, GPUS, CLOUD, SIMULATOR, qcl.Active(10, 8),
                                    4000, 100000, strict_transfers=True)
    nodes = plan.graph_json()["nodes"]
    samples = [(node["device"]["id"], node["params"]["shots"]) for node in nodes
               if node["op"] == "sample"]
    assert samples == [("mq02", 4000), ("sim-ideal", 100000)] * 2
    sqd = [node["params"] for node in nodes if node["op"] == "sqd.diagonalize"]
    assert [params.get("recovery", True) for params in sqd] == [True, False, True, False] * 2
    assert {node["device"]["id"] for node in nodes if node["op"] == "scf.rhf"} == {"gpu0", "gpu1"}
    assert {transfer["kind"] for transfer in plan.explain(format="json")["transfers"]} == {
        "explicit"}
    estimate = plan.estimate()
    assert estimate["qpu_cost"] > 0 and estimate["shots"] == 2 * 104000


@qcl.workflow
def ablation(cpu, qpu):
    mean_field = qcl.scf.hf(qcl.Molecule("H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7",
                                         "sto-3g")).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(4, 4)).to(cpu)
    kernel = qcl.kernels.lucj(space, params=qcl.ccsd.amplitudes(space).to(cpu))
    bits = qcl.sample(kernel, shots=60, seed=5).to(qpu)
    return {"full": qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu),
            "raw": qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector,
                                       recovery=False, extend=False).to(cpu),
            "exact": qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy}


def test_sqd_ablation_reports_what_the_samples_alone_give(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1000\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = ablation.compile(qcl.Cluster.local().cpus[0], CPU_SIMULATOR).run()
    exact = result["exact"].value
    assert result["raw"].value >= result["full"].value - 1e-9 >= exact - 2e-9
    raw, full = result["raw"].provenance, result["full"].provenance
    assert (raw["recovery"], raw["extend"], full["recovery"], full["extend"]) == (
        False, False, True, True)
    assert raw["subspace_dimension"] <= full["subspace_dimension"] <= full["sector_dimension"]
    assert full["sector_dimension"] == 36
    assert 0 < raw["valid_fraction"] <= 1


def test_sqd_options_must_be_booleans():
    @qcl.workflow
    def bad(cpu, qpu):
        mean_field = qcl.scf.hf(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        bits = qcl.sample(qcl.kernels.hf(space), shots=10).to(qpu)
        return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector,
                                   recovery="no").to(cpu)

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        bad.compile(qcl.Cluster.local().cpus[0], CPU_SIMULATOR)
