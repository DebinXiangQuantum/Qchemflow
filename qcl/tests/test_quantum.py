import json

import pytest

import qcl


@qcl.workflow
def hf_samples(molecule, active, cpu, qpu, shots=128):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sample(qcl.kernels.hf(space), shots=shots, seed=7).to(qpu)


@qcl.workflow
def sqd_energy(molecule, active, cpu, qpu, shots=128):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    bits = qcl.sample(qcl.kernels.hf(space), shots=shots, seed=7).to(qpu)
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector,
                               seed=7).to(cpu)


@qcl.workflow
def sampled_vqe_energy(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.vqe.sampled(space.hamiltonian, space.sector, ansatz="lucj",
                           optimizer="cobyla", max_evaluations=3,
                           shots_per_group=128, n_groups=5, seed=7).to(qpu)


def _inputs():
    return qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g"), qcl.Active(2, 2)


def test_hf_sampling_on_local_simulator_has_no_monetary_cost(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    plan = hf_samples.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0])
    assert plan.explain(format="json")["operations"][-1]["op"] == "sample"
    estimate = plan.estimate()
    assert estimate["shots"] == 128
    assert estimate["qpu_cost"] == 0.0

    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.shots == 128
    assert result.counts == {"5": 128}
    assert result.source == "quantum_simulator"


def test_sqd_energy_from_simulated_qpu_samples_matches_fci(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    plan = sqd_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0])
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)
    assert result.provenance["sample_source"] == "quantum_simulator"
    assert result.provenance["implementation"] == "sqd.diagonalize/chemflow-pyscf"


def test_sampled_vqe_uses_qpu_simulator_and_declared_shot_budget(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    plan = sampled_vqe_energy.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0])
    assert plan.estimate()["shots"] == 3 * 5 * 128
    assert plan.estimate()["qpu_cost"] == 0
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1920\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run(timeout_s=120)
    assert result.value < -1.0
    assert result.provenance["sample_source"] == "quantum_simulator"
    assert result.provenance["sample_shots"] == result.provenance["evaluations"] * 5 * 128


def test_sampled_vqe_checks_operator_and_cloud_cost(tmp_path, monkeypatch):
    molecule, active = _inputs()
    cpu = qcl.Cluster.local().cpus[0]
    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", billing_account="test-account",
                       provider="lqcloud", backend="MQ02")
    plan = sampled_vqe_energy.compile(molecule, active, cpu, cloud)
    assert plan.estimate()["shots"] == 1920
    assert plan.estimate()["qpu_cost"] > 0
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1920\n"
                      "[cloud_qpu]\nqpu_cost = 1\ncurrency = 'CNY'\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        plan.run()
    policy.write_text("[budget]\nqpu_shots = 1920\n"
                      "[cloud_qpu]\nqpu_cost = 1000\ncurrency = 'CNY'\n"
                      "approval_above = 1\n")
    with pytest.raises(qcl.CompileError, match="QCL-E302"):
        plan.run()

    @qcl.workflow
    def wrong(molecule, active, cpu, qpu):
        mf = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mf, active).to(cpu)
        return qcl.vqe.sampled(space.sector, space.hamiltonian,
                               n_groups=5).to(qpu)

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        wrong.compile(molecule, active, cpu, cloud)
    with pytest.raises(qcl.CompileError, match="QCL-E209"):
        sampled_vqe_energy.compile(molecule, active, cpu,
                                   qcl.Device("small", "qpu", qubits=3,
                                              simulator_host="cpu0",
                                              simulator_method="statevector"))
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        sampled_vqe_energy.compile(molecule, active, cpu,
                                   qcl.Device("gpu-sim", "qpu", qubits=4,
                                              simulator_host="gpu0",
                                              simulator_method="statevector"))


def test_sampled_vqe_uses_fake_mq02_provider_without_live_submission():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    calls = []

    class RecordingBackend:
        def sample(self, store, prep, shots, seed, *, repetition):
            calls.append((prep.circuit.measurement_basis, repetition))
            samples = Samples(counts={"0": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=(f"offline-{len(calls)}",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots, "qpu_s": 0.1},
                                   job_ids=samples.job_ids)

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", billing_account="test-account",
                       provider="lqcloud", backend="MQ02")
    plan = sampled_vqe_energy.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud)
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: RecordingBackend())["result"]
    assert len(calls) == result["provenance"]["evaluations"] * 5
    assert len(result["provenance"]["sample_job_ids"]) == len(calls)
    assert result["provenance"]["sample_source"] == "qpu"
    assert result["provenance"]["resources"]["qpu_shots"] == len(calls) * 128


def test_sampled_vqe_rejects_group_mismatch_before_provider_submission():
    from qcl._worker import execute

    from chemflow.diagnostics import AdapterError

    @qcl.workflow
    def wrong_groups(molecule, active, cpu, qpu):
        mf = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mf, active).to(cpu)
        return qcl.vqe.sampled(space.hamiltonian, space.sector,
                               max_evaluations=3, shots_per_group=128,
                               n_groups=4).to(qpu)

    class NoSubmissionBackend:
        def sample(self, *_args, **_kwargs):
            raise AssertionError("provider submission must not occur")

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    plan = wrong_groups.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud)
    with pytest.raises(AdapterError, match="n_groups"):
        execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                qpu_backend_factory=lambda _: NoSubmissionBackend())


def test_qpu_capacity_must_be_known_for_sampling_and_vqe():
    molecule, active = _inputs()
    cpu = qcl.Cluster.local().cpus[0]
    unknown = qcl.Device("sim-unknown", "qpu", simulator_host="cpu0",
                         simulator_method="statevector")
    for workflow in (hf_samples, sampled_vqe_energy):
        with pytest.raises(qcl.CompileError, match="QCL-E209"):
            workflow.compile(molecule, active, cpu, unknown)


def test_multi_qpu_shots_merge_with_per_device_provenance(tmp_path, monkeypatch):
    molecule, active = _inputs()
    cpu = qcl.Device("cpu0", "cpu")
    qpus = [qcl.Device(name, "qpu", qubits=4, site="local",
                       simulator_host="cpu0", simulator_method="statevector")
            for name in ("sim0", "sim1")]
    plan = hf_samples.compile(molecule, active, cpu, qpus, shots=[80, 48])
    operations = plan.explain(format="json")["operations"]
    assert [(item["op"], item["device"]) for item in operations[-2:]] == [
        ("sample", "sim0"), ("sample", "sim1")]
    samples = [node for node in plan.graph_json()["nodes"] if node["op"] == "sample"]
    assert [node["params"]["shots"] for node in samples] == [80, 48]
    assert [node["params"]["repetition"] for node in samples] == [0, 1]
    assert plan.estimate()["shots"] == 128

    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = plan.run()
    assert result.counts == {"5": 128}
    assert result.shots == 128
    assert [(part["device"], part["shots"]) for part in result.parts] == [
        ("sim0", 80), ("sim1", 48)]
    assert all(part["source"] == "quantum_simulator" for part in result.parts)

    energy = sqd_energy.compile(molecule, active, cpu, qpus).run()
    assert energy.value == pytest.approx(-1.1372838344885, abs=1e-8)
    assert len(energy.provenance["sample_parts"]) == 2


def test_device_lists_require_a_parallel_axis_and_matching_shot_split():
    molecule, active = _inputs()
    cpu = qcl.Device("cpu0", "cpu")
    qpus = [qcl.Device(name, "qpu", qubits=4, simulator_host="cpu0",
                       simulator_method="statevector") for name in ("sim0", "sim1")]
    with pytest.raises(qcl.CompileError) as error:
        hf_samples.compile(molecule, active, cpu, qpus, shots=[128])
    assert error.value.diagnostics[0]["code"] == "QCL-E204"

    @qcl.workflow
    def invalid(molecule, devices):
        return qcl.scf.rhf(molecule).to(devices).energy

    with pytest.raises(qcl.CompileError) as error:
        invalid.compile(molecule, [cpu, cpu])
    assert error.value.diagnostics[0]["code"] == "QCL-E204"


def test_multi_mq02_dispatch_uses_distinct_reservations():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    repetitions = []

    class RecordingBackend:
        def sample(self, store, prep, shots, seed, *, repetition):
            repetitions.append(repetition)
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=(f"job-{repetition}",))
            return ExecutionResult(samples, device="lqcloud:MQ02", cost={},
                                   job_ids=samples.job_ids)

    qpus = [qcl.Device(name, "qpu", qubits=24, site="remote", billing="metered",
                       provider="lqcloud", backend="MQ02") for name in ("mq0", "mq1")]
    plan = hf_samples.compile(*_inputs(), qcl.Device("cpu0", "cpu"), qpus)
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: RecordingBackend())["result"]
    assert repetitions == [0, 1]
    assert result["shots"] == 128
    assert result["job_ids"] == ["job-0", "job-1"]


def test_sqd_rejects_mismatched_space_and_gpu_placement():
    molecule, active = _inputs()
    cpu = qcl.Device("cpu0", "cpu")
    qpu = qcl.Device("sim0", "qpu", qubits=4, simulator_host="cpu0",
                     simulator_method="statevector")

    @qcl.workflow
    def mismatch(molecule, active, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        first = qcl.active_space(mean_field, active).to(cpu)
        second = qcl.active_space(mean_field, active).to(cpu)
        bits = qcl.sample(qcl.kernels.hf(first), shots=128).to(qpu)
        return qcl.sqd.diagonalize(second.hamiltonian, bits, second.sector).to(cpu)

    with pytest.raises(qcl.CompileError) as error:
        mismatch.compile(molecule, active, cpu, qpu)
    assert error.value.diagnostics[0]["code"] == "QCL-E103"

    @qcl.workflow
    def gpu_sqd(molecule, active, cpu, qpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        bits = qcl.sample(qcl.kernels.hf(space), shots=128).to(qpu)
        return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(gpu)

    with pytest.raises(qcl.CompileError) as error:
        gpu_sqd.compile(molecule, active, cpu, qpu, qcl.Device("gpu0", "gpu"))
    assert error.value.diagnostics[0]["code"] == "QCL-E203"


def test_qpu_shot_budget_is_required_and_enforced(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    plan = hf_samples.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0])
    monkeypatch.delenv("QCL_POLICY_FILE", raising=False)
    with pytest.raises(qcl.CompileError) as error:
        plan.run()
    assert error.value.diagnostics[0]["code"] == "QCL-E301"

    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 127\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError) as error:
        plan.run()
    assert error.value.diagnostics[0]["code"] == "QCL-E301"

    policy.write_text("this is not TOML")
    with pytest.raises(qcl.CompileError) as error:
        plan.run()
    assert error.value.diagnostics[0]["code"] == "QCL-E301"


def test_qpu_capacity_and_unsupported_device_report_compile_errors():
    molecule, active = _inputs()
    cpu = qcl.Device("cpu0", "cpu")
    small_qpu = qcl.Device("mq02", "qpu", qubits=2, site="remote",
                           billing="metered", provider="lqcloud", backend="MQ02")
    with pytest.raises(qcl.CompileError) as error:
        hf_samples.compile(molecule, active, cpu, small_qpu)
    assert any(item["code"] == "QCL-E209" for item in error.value.diagnostics)

    with pytest.raises(qcl.CompileError) as error:
        hf_samples.compile(molecule, active, cpu, qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_metered_cloud_cost_is_separate_from_simulator_cost():
    molecule, active = _inputs()
    cpu = qcl.Device("cpu0", "cpu")
    free = qcl.Device("local-free", "qpu", qubits=4, site="local",
                      simulator_host="cpu0", simulator_method="statevector")
    metered = qcl.Device("mq02", "qpu", qubits=4, site="remote",
                         billing="metered", provider="lqcloud", backend="MQ02")
    assert hf_samples.compile(molecule, active, cpu, free).estimate()["qpu_cost"] == 0.0
    cost = hf_samples.compile(molecule, active, cpu, metered).estimate()
    assert cost["currency"] == "CNY"
    assert cost["qpu_cost"] == pytest.approx(5.0 + 0.005 * 128)


def test_unregistered_qpu_is_rejected_before_runtime():
    molecule, active = _inputs()
    with pytest.raises(qcl.CompileError) as error:
        hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"),
                           qcl.Device("qpu0", "qpu", qubits=4, site="remote"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)

    with pytest.raises(qcl.CompileError) as error:
        hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"),
                           qcl.Device("remote-sim", "qpu", qubits=4, site="remote",
                                      simulator_host="cpu0", simulator_method="statevector"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_metered_cloud_approval_is_bound_to_account_and_plan(tmp_path, monkeypatch):
    molecule, active = _inputs()
    device = qcl.Device("mq02", "qpu", qubits=4, site="remote",
                        billing="metered", billing_account="lab-a",
                        provider="lqcloud", backend="MQ02")
    plan = hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"), device)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n[cloud_qpu]\n"
                      "qpu_cost = 10.0\ncurrency = 'CNY'\napproval_above = 1.0\n"
                      "[cloud_qpu.approval]\nplan_hash = 'wrong'\n"
                      "billing_account = 'lab-a'\ncurrency = 'CNY'\nmax_cost = 10.0\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError) as error:
        plan._check_qpu_policy()
    assert error.value.diagnostics[0]["code"] == "QCL-E302"

    policy.write_text(policy.read_text().replace("wrong", plan.hash)
                      .replace("lab-a", "lab-b"))
    with pytest.raises(qcl.CompileError) as error:
        plan._check_qpu_policy()
    assert error.value.diagnostics[0]["code"] == "QCL-E302"

    policy.write_text(policy.read_text().replace("lab-b", "lab-a"))
    assert plan._check_qpu_policy() == {
        "max_cost": 10.0, "forecast_cost": pytest.approx(5.64), "currency": "CNY",
    }


def test_qcl_worker_passes_cloud_cost_ceiling_to_mq02(monkeypatch):
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    ceilings = []

    class RecordingBackend:
        def __init__(self, *, backend, max_billing_multiplier):
            assert backend == "MQ02"
            ceilings.append(max_billing_multiplier)

        def sample(self, store, prep, shots, seed):
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=("fake-job",))
            return ExecutionResult(samples, device="lqcloud:MQ02", cost={},
                                   job_ids=("fake-job",))

    monkeypatch.setattr("chemflow.backends.lqcloud.LqcloudBackend", RecordingBackend)
    device = qcl.Device("mq02", "qpu", qubits=24, site="remote", billing="metered",
                        provider="lqcloud", backend="MQ02")
    plan = hf_samples.compile(*_inputs(), qcl.Device("cpu0", "cpu"), device)
    execute({"graph": plan.graph_json(), "output": plan._output.node_id,
             "cloud_qpu_policy": {"max_cost": 6.0, "forecast_cost": 5.64,
                                  "currency": "CNY"}})
    assert ceilings == [pytest.approx(6.0 / 5.64)]


def test_qcl_dispatches_to_mq02_adapter_and_preserves_qpu_identity():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    class FakeBackend:
        def sample(self, store, prep, shots, seed):
            assert prep.circuit is not None
            assert (shots, seed) == (128, 7)
            samples = Samples(counts={"5": 128}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=("fake-job-1",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots, "qpu_s": 0.5},
                                   job_ids=("fake-job-1",))

    molecule, active = _inputs()
    device = qcl.Device("mq02", "qpu", qubits=4, site="remote", billing="metered",
                        provider="lqcloud", backend="MQ02")
    plan = hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"), device)
    graph = json.loads(plan._graph.graph_json())
    result = execute({"graph": graph, "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: FakeBackend())["result"]
    assert result["source"] == "qpu"
    assert result["counts"] == {"5": 128}
    assert result["job_ids"] == ("fake-job-1",)

    assert result["provenance"]["implementation"] == "sample/chemflow-lqcloud-mq02"


def test_qcl_rejects_mislabelled_qpu_samples():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    class WrongBackend:
        def sample(self, store, prep, shots, seed):
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="quantum_simulator", seed=seed)
            return ExecutionResult(samples, device="sim", cost={})

    molecule, active = _inputs()
    device = qcl.Device("mq02", "qpu", qubits=4, site="remote", billing="metered",
                        provider="lqcloud", backend="MQ02")
    plan = hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"), device)
    with pytest.raises(ValueError, match="inconsistent source"):
        execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                qpu_backend_factory=lambda _: WrongBackend())


def test_qcl_mq02_adapter_reserves_before_fake_submission(tmp_path, monkeypatch):
    pytest.importorskip("qiskit")
    pytest.importorskip("lqcloud")

    from qcl._worker import execute

    from chemflow.backends.calibration import load_calibration
    from chemflow.backends.lqcloud import LqcloudBackend
    from chemflow.diagnostics import AdapterError

    class FakeResult:
        success = True

        def __init__(self, shots):
            self.data = {"memory": ["1010"] * shots}

    class FakeJob:
        job_id = "fake-job-1"
        _params_resubmit = object()

        def __init__(self, shots):
            self.shots = shots

        def result(self, **kwargs):
            assert self._params_resubmit is None
            return FakeResult(self.shots)

        def status_info(self):
            return {"status": "completed", "cost": 5.64, "currency": "CNY",
                    "execution_time": 0.5, "error_type": None}

    class FakeDevice:
        def __init__(self):
            self.config = {**load_calibration("MQ02"), "status": "active"}
            self.reserved_before_submit = False

        def run(self, program, *, shots, **kwargs):
            self.reserved_before_submit = bool(list(tmp_path.glob("*.json")))
            assert program is not None
            return FakeJob(shots)

    fake = FakeDevice()

    class FakeProvider:
        def get_backend(self, name):
            assert name == "MQ02"
            return fake

    monkeypatch.setenv("LQCLOUD_API_KEY", "test-key")
    molecule, active = _inputs()
    device = qcl.Device("mq02", "qpu", qubits=24, site="remote", billing="metered",
                        provider="lqcloud", backend="MQ02")
    plan = hf_samples.compile(molecule, active, qcl.Device("cpu0", "cpu"), device)
    fake.config["billing_multiplier"] = 2.0
    limited = LqcloudBackend(state_dir=tmp_path / "denied",
                             provider_factory=lambda _: FakeProvider(),
                             max_billing_multiplier=1.0)
    with pytest.raises(AdapterError, match="Live QPU price exceeds"):
        execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                qpu_backend_factory=lambda _: limited)
    assert not (tmp_path / "denied").exists()
    fake.config["billing_multiplier"] = 1.0
    backend = LqcloudBackend(state_dir=tmp_path, provider_factory=lambda _: FakeProvider())
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: backend)["result"]
    assert fake.reserved_before_submit
    assert result["source"] == "qpu"
    assert result["counts"] == {"5": 128}
    assert result["job_ids"] == ("fake-job-1",)
    assert result["provenance"]["billing"] == {
        "cost": 5.64, "currency": "CNY", "source": "platform",
    }

    energy_plan = sqd_energy.compile(molecule, active, qcl.Device("cpu0", "cpu"), device)
    energy = execute({"graph": energy_plan.graph_json(),
                      "output": energy_plan._output.node_id},
                     qpu_backend_factory=lambda _: backend)["result"]
    assert energy["value"] == pytest.approx(-1.1372838344885, abs=1e-8)
    assert energy["provenance"]["sample_source"] == "qpu"
    assert energy["provenance"]["sample_job_ids"] == ("fake-job-1",)
    assert energy["provenance"]["sample_billing"] == {
        "cost": 5.64, "currency": "CNY", "source": "platform",
    }
