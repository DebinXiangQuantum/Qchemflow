import pytest

import qcl


@qcl.workflow
def hf_pauli_energy(molecule, active, cpu, qpu, n_groups=5):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    kernel = qcl.kernels.hf(space)
    return qcl.estimate(operator, kernel, n_groups=n_groups,
                        shots_per_group=128, seed=7).to(qpu)


def _inputs():
    return qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), qcl.Active(2, 2)


def test_pauli_estimate_on_local_simulator(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    plan = hf_pauli_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0],
                                   strict_transfers=True)
    assert plan.estimate()["shots"] == 640
    assert plan.estimate()["qpu_cost"] == 0
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.unit == "hartree"
    assert result.value < -1.0
    assert result.variance >= 0
    assert result.provenance["sample_shots"] == 640
    assert result.provenance["sample_source"] == "quantum_simulator"


def test_pauli_estimate_cloud_forecast_and_fake_jobs():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    calls = []

    class RecordingBackend:
        def sample(self, store, prep, shots, seed, **_kwargs):
            calls.append(prep.circuit.measurement_basis)
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=(f"offline-{len(calls)}",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots, "qpu_s": 0.1},
                                   job_ids=samples.job_ids)

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    plan = hf_pauli_energy.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud)
    assert plan.estimate()["shots"] == 640
    assert plan.estimate()["qpu_cost"] > 0
    payload = {"graph": plan.graph_json(), "output": plan._output.node_id}
    result = execute(payload, qpu_backend_factory=lambda _: RecordingBackend())["result"]
    assert len(calls) == 5
    assert len(result["provenance"]["job_ids"]) == 5


def test_pauli_estimate_rejects_wrong_operator_and_group_count():
    from qcl._worker import execute

    @qcl.workflow
    def wrong_type(molecule, active, cpu, qpu):
        space = qcl.active_space(qcl.scf.hf(molecule).to(cpu), active).to(cpu)
        return qcl.estimate(space.hamiltonian, qcl.kernels.hf(space),
                            n_groups=5, shots_per_group=128).to(qpu)

    cluster = qcl.Cluster.local()
    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        wrong_type.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0])

    plan = hf_pauli_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0],
                                   n_groups=4)
    with pytest.raises(ValueError, match="n_groups"):
        execute({"graph": plan.graph_json(), "output": plan._output.node_id})


@qcl.quantum_kernel
def four_qubit_kernel(qubits: qcl.Qubits[4], theta: qcl.Params[1]):
    qcl.ry(qubits[0], theta[0])


@qcl.quantum_kernel
def three_qubit_kernel(qubits: qcl.Qubits[3], theta: qcl.Params[1]):
    qcl.ry(qubits[0], theta[0])


@qcl.workflow
def custom_pauli_energy(molecule, active, cpu, qpu, kernel):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.estimate(operator, kernel, params=[0.0], n_groups=5,
                        shots_per_group=128, seed=7).to(qpu)


def test_custom_kernel_pauli_estimate_and_qubit_shape(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        custom_pauli_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0],
                                    three_qubit_kernel)
    plan = custom_pauli_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0],
                                       four_qubit_kernel)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.unit == "hartree"
    assert result.provenance["method"]["ansatz"] == "custom"

    gpu_sim = qcl.Device("sim-gpu", "qpu", qubits=20,
                         simulator_host="gpu0", simulator_method="statevector")
    gpu_plan = custom_pauli_energy.compile(*_inputs(), cluster.cpus[0], gpu_sim,
                                           four_qubit_kernel)
    assert gpu_plan.explain(format="json")["operations"][-1]["implementation"] == (
        "estimate.quantum/qcl-cupy-statevector")


def test_cloud_pauli_estimate_respects_shot_budget(tmp_path, monkeypatch):
    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    plan = hf_pauli_energy.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 639\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        plan.run()


def test_cloud_pauli_estimate_preserves_completed_jobs_on_later_failure():
    from qcl._worker import execute

    from chemflow.diagnostics import AdapterError
    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    class FailingBackend:
        calls = 0

        def sample(self, store, prep, shots, seed, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise AdapterError("CF-E080", None, "Second group failed",
                                   "Reconcile this job", nothing_consumed=True)
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=("first-job",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots, "qpu_s": 0.1},
                                   job_ids=samples.job_ids)

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    plan = hf_pauli_energy.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud)
    with pytest.raises(AdapterError) as failure:
        execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                qpu_backend_factory=lambda _: FailingBackend())
    assert failure.value.job_ids == ("first-job",)
    assert failure.value.incurred_cost["qpu_shots"] == 128
    assert failure.value.nothing_consumed is False


@qcl.workflow
def distributed_pauli_energy(molecule, active, cpu, qpus):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.estimate(operator, qcl.kernels.hf(space),
                        n_groups=5, shots_per_group=128, seed=7).to(qpus)


def test_pauli_measurement_groups_split_across_two_qpus(tmp_path, monkeypatch):
    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device(f"sim{index}", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector")
            for index in range(2)]
    plan = distributed_pauli_energy.compile(*_inputs(), cpu, qpus,
                                             strict_transfers=True)
    operations = [item for item in plan.explain(format="json")["operations"]
                  if item["op"] == "estimate.quantum"]
    assert len(operations) == 2
    assert plan.estimate()["shots"] == 640
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.value < -1.0
    assert result.variance >= 0
    assert result.provenance["device"] == "multi"
    assert [part["sample_shots"] for part in result.provenance["parts"]] == [384, 256]
    single = hf_pauli_energy.compile(*_inputs(), cpu, qpus[0]).run()
    assert result.value == pytest.approx(single.value, abs=1e-12)
    assert result.variance == pytest.approx(single.variance, abs=1e-12)


def test_pauli_group_split_requires_one_group_per_qpu():
    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device(f"sim{index}", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector")
            for index in range(6)]
    with pytest.raises(qcl.CompileError, match="QCL-E204"):
        distributed_pauli_energy.compile(*_inputs(), cpu, qpus)


def test_pauli_groups_split_across_two_fake_cloud_qpus():
    from qcl._worker import execute

    from chemflow.backends.lqcloud import load_pricing
    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    calls = []

    class RecordingBackend:
        def __init__(self, device):
            self.device = device

        def sample(self, store, prep, shots, seed, **_kwargs):
            calls.append(self.device)
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device=self.device,
                              job_ids=(f"{self.device}-{len(calls)}",))
            return ExecutionResult(samples, device=self.device,
                                   cost={"qpu_shots": shots, "qpu_s": 0.1},
                                   job_ids=samples.job_ids)

    qpus = [qcl.Device(f"mq{index}", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
            for index in range(2)]
    plan = distributed_pauli_energy.compile(*_inputs(), qcl.Cluster.local().cpus[0], qpus)
    pricing = load_pricing("MQ02")
    assert plan.estimate()["shots"] == 640
    assert plan.estimate()["qpu_cost"] == pytest.approx(
        5 * (pricing.per_task + 128 * pricing.per_shot))
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda device: RecordingBackend(device["id"]))["result"]
    assert calls == ["mq0"] * 3 + ["mq1"] * 2
    assert len(result["provenance"]["job_ids"]) == 5


def test_cpu_gpu_qpu_lanes_agree_for_custom_pauli_kernel(tmp_path, monkeypatch):
    from chemflow.backends.gpu import cuda_available

    if not cuda_available():
        pytest.skip("CUDA is unavailable")
    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device("sim-cpu", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector"),
            qcl.Device("sim-gpu", "qpu", qubits=4,
                       simulator_host="gpu0", simulator_method="statevector")]
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = custom_pauli_energy.compile(*_inputs(), cpu, qpus,
                                          four_qubit_kernel).run()
    reference = custom_pauli_energy.compile(*_inputs(), cpu, qpus[0],
                                             four_qubit_kernel).run()
    assert result.provenance["device"] == "multi"
    assert result.value == pytest.approx(reference.value, abs=0.1)
