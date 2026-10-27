import pytest

import qcl


@qcl.workflow
def lucj_samples(molecule, active, cpu, qpu, shots=512):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    params = qcl.ccsd.amplitudes(space).to(cpu)
    kernel = qcl.kernels.lucj(space, params=params)
    return qcl.sample(kernel, shots=shots, seed=19).to(qpu)


def test_ccsd_seeded_lucj_samples_are_reproducible(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = lucj_samples.compile(molecule, qcl.Active(2, 2),
                                cluster.cpus[0], cluster.qpus[0])
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 512\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    first = plan.run()
    second = plan.run()
    assert first.counts == second.counts
    assert first.shots == 512
    assert first.provenance["method"]["family"] == "LUCJ sampling"
    assert sum(first.counts.values()) == 512


def test_lucj_shape_and_amplitude_device_are_checked():
    @qcl.workflow
    def wrong(molecule, active, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.sample(qcl.kernels.lucj(space, params=[1.0]), shots=16).to(qpu)

    cluster = qcl.Cluster.local()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(qcl.CompileError) as error:
        wrong.compile(molecule, qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    assert any(item["code"] == "QCL-E104" for item in error.value.diagnostics)

    @qcl.workflow
    def gpu_seed(molecule, active, cpu, gpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.ccsd.amplitudes(space).to(gpu)

    with pytest.raises(qcl.CompileError) as error:
        gpu_seed.compile(molecule, qcl.Active(2, 2), cluster.cpus[0],
                         qcl.Device("gpu0", "gpu"))
    assert any(item["code"] == "QCL-E203" for item in error.value.diagnostics)


def test_lucj_mq02_adapter_receives_parameterized_circuit():
    from qcl._worker import execute

    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    class FakeBackend:
        def sample(self, store, prep, shots, seed):
            assert prep.circuit.ansatz == "lucj"
            assert len(prep.circuit.layers) == 1
            assert len(prep.circuit.layers[0]) == 3
            samples = Samples(counts={"5": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=("fake-lucj-job",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots},
                                   job_ids=("fake-lucj-job",))

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    qpu = qcl.Device("mq02", "qpu", qubits=4, site="remote", billing="metered",
                     provider="lqcloud", backend="MQ02")
    plan = lucj_samples.compile(molecule, qcl.Active(2, 2),
                                qcl.Device("cpu0", "cpu"), qpu, shots=128)
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: FakeBackend())["result"]
    assert result["job_ids"] == ("fake-lucj-job",)


def test_ccsd_parameter_result_has_three_lucj_angles():
    @qcl.workflow
    def parameters(molecule, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
        return qcl.ccsd.amplitudes(space).to(cpu)

    result = parameters.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                                qcl.Cluster.local().cpus[0]).run()
    assert isinstance(result, qcl.LucjParameters)
    assert len(result.vector) == 3
    assert result.provenance["implementation"] == "ccsd.amplitudes/chemflow-pyscf"


def test_lucj_uses_active_space_after_explicit_cpu_copy():
    @qcl.workflow
    def copied(molecule, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu).to(cpu)
        params = qcl.ccsd.amplitudes(space).to(cpu)
        kernel = qcl.kernels.lucj(space, params=params)
        return qcl.sample(kernel, shots=16).to(qpu)

    cluster = qcl.Cluster.local()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = copied.compile(molecule, cluster.cpus[0], cluster.qpus[0])
    assert any(node["op"] == "kernel.lucj" for node in plan.graph_json()["nodes"])


def test_copied_active_space_still_checks_qpu_qubit_capacity():
    @qcl.workflow
    def copied(molecule, cpu, qpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu).to(cpu)
        return qcl.sample(qcl.kernels.hf(space), shots=16).to(qpu)

    qpu = qcl.Device("sim0", "qpu", qubits=3,
                     simulator_host="cpu0", simulator_method="statevector")
    with pytest.raises(qcl.CompileError) as error:
        copied.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                       qcl.Cluster.local().cpus[0], qpu)
    assert any(item["code"] == "QCL-E209" for item in error.value.diagnostics)
