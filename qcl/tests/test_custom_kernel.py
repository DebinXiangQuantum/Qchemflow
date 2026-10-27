import math

import pytest

import qcl


@qcl.quantum_kernel
def hea(qubits: qcl.Qubits[8], theta: qcl.Params[16]):
    for index in range(8):
        qcl.ry(qubits[index], theta[index])
    for index in range(7):
        qcl.cz(qubits[index], qubits[index + 1])
    for index in range(8):
        qcl.ry(qubits[index], theta[8 + index])


@qcl.workflow
def hea_samples(qpu, params):
    return qcl.sample(hea, params=params, shots=128, seed=7).to(qpu)


def test_hea_kernel_traces_and_samples_on_local_qpu(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    qpu = qcl.Cluster.local().qpus[0]
    plan = hea_samples.compile(qpu, [0.0] * 16)
    kernel = next(node for node in plan.graph_json()["nodes"]
                  if node["op"] == "kernel.custom")
    assert len(kernel["params"]["gates"]) == 23
    result = plan.run()
    assert result.counts == {"0": 128}
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        result.provenance["implementation"])

    turned = [0.0] * 16
    turned[0] = math.pi
    assert hea_samples.compile(qpu, turned).run().counts == {"1": 128}


def test_hea_rejects_parameter_shape_and_qubit_capacity():
    qpu = qcl.Cluster.local().qpus[0]
    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        hea_samples.compile(qpu, [0.0] * 15)
    small = qcl.Device("sim-small", "qpu", qubits=7,
                       simulator_host="cpu0", simulator_method="statevector")
    with pytest.raises(qcl.CompileError, match="QCL-E209"):
        hea_samples.compile(small, [0.0] * 16)


def test_hea_lowers_for_fake_mq02_without_live_job():
    from qcl._worker import execute

    from chemflow.backends.circuits import build_circuit
    from chemflow.runtime import ExecutionResult
    from chemflow.types import Samples

    circuits = []

    class RecordingBackend:
        def sample(self, store, prep, shots, seed, **_kwargs):
            circuits.append(build_circuit(prep.circuit))
            samples = Samples(counts={"0": shots}, shots=shots,
                              encoding_id=prep.encoding_id, basis="computational",
                              source="qpu", seed=seed, device="lqcloud:MQ02",
                              job_ids=("offline-hea",))
            return ExecutionResult(samples, device="lqcloud:MQ02",
                                   cost={"qpu_shots": shots, "qpu_s": 0.1},
                                   job_ids=samples.job_ids)

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    plan = hea_samples.compile(cloud, [0.0] * 16)
    result = execute({"graph": plan.graph_json(), "output": plan._output.node_id},
                     qpu_backend_factory=lambda _: RecordingBackend())["result"]
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 8
    assert result["job_ids"] == ("offline-hea",)


def test_custom_statevector_matches_qiskit_for_entangling_angles():
    import numpy as np
    from qcl._worker import _custom_statevector
    from qiskit.quantum_info import Statevector

    from chemflow.backends.circuits import build_circuit
    from chemflow.types import CircuitSpec

    spec = CircuitSpec(ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                       n_qubits=8, gates=hea.gates,
                       layers=(tuple((index + 1) / 13 for index in range(16)),))
    actual = _custom_statevector(spec, np)
    expected = Statevector.from_instruction(build_circuit(spec)).data
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_custom_statevector_matches_pauli_measurement_rotations():
    import numpy as np
    from qcl._worker import _custom_statevector
    from qiskit.quantum_info import Statevector

    from chemflow.backends.circuits import build_circuit
    from chemflow.types import CircuitSpec

    spec = CircuitSpec(ansatz="custom", n_orb=0, n_alpha=0, n_beta=0,
                       n_qubits=8, gates=hea.gates,
                       layers=(tuple((index + 1) / 13 for index in range(16)),),
                       measurement_basis="XYZIXYZI")
    np.testing.assert_allclose(_custom_statevector(spec, np),
                               Statevector.from_instruction(build_circuit(spec)).data,
                               atol=1e-14)


@qcl.quantum_kernel
def feedback_kernel(qubits: qcl.Qubits[1], theta: qcl.Params[1]):
    qcl.x(qubits[0])
    observed = qcl.measure(qubits[0])
    qcl.cond(observed, lambda: qcl.x(qubits[0]), lambda: None)


@qcl.workflow
def feedback_samples(qpu):
    return qcl.sample(feedback_kernel, params=[0.0], shots=32, seed=4).to(qpu)


def test_mid_circuit_feedback_runs_on_cpu_simulator(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 32\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    cpu_qpu = qcl.Device("sim-cpu", "qpu", qubits=2,
                         simulator_host="cpu0", simulator_method="statevector")
    plan = feedback_samples.compile(cpu_qpu)
    result = plan.run()
    assert result.counts == {"0": 32}
    assert plan.explain(format="json")["operations"][0]["implementation"] == (
        result.provenance["implementation"])


def test_mid_circuit_feedback_rejects_unsupported_targets():
    gpu_qpu = qcl.Device("sim-gpu", "qpu", qubits=2,
                         simulator_host="gpu0", simulator_method="statevector")
    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    for device in (gpu_qpu, cloud):
        with pytest.raises(qcl.CompileError, match="QCL-E206"):
            feedback_samples.compile(device)


def test_measured_bit_rejects_python_branch():
    with pytest.raises(qcl.CompileError, match="QCL-E210"):

        @qcl.quantum_kernel
        def invalid(qubits: qcl.Qubits[1], theta: qcl.Params[1]):
            if qcl.measure(qubits[0]):
                qcl.x(qubits[0])
