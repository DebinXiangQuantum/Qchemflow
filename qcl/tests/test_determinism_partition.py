import pytest

import qcl


@qcl.workflow
def sampled(molecule, active, cpu, qpu, partition=None):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sample(qcl.kernels.hf(space), shots=128, seed=7,
                      partition=partition).to(qpu)


def _inputs():
    return qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), qcl.Active(2, 2)


def test_deterministic_compile_accepts_seeded_local_simulator(monkeypatch, tmp_path):
    cluster = qcl.Cluster.local()
    simulator = qcl.Device("cpu-sim", "qpu", qubits=4,
                           simulator_host=cluster.cpus[0].id,
                           simulator_method="statevector")
    plan = sampled.compile(*_inputs(), cluster.cpus[0], simulator,
                           deterministic=True)
    assert plan.estimate()["shots"] == 128
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    assert plan.run().counts == plan.run().counts


def test_deterministic_cpu_plan_repeats_and_has_distinct_identity(monkeypatch):
    import subprocess

    @qcl.workflow
    def rhf(molecule, cpu):
        return qcl.scf.rhf(molecule).to(cpu).energy

    molecule = _inputs()[0]
    cpu = qcl.Cluster.local().cpus[0]
    plan = rhf.compile(molecule, cpu, deterministic=True)
    assert plan.hash != rhf.compile(molecule, cpu).hash
    original = subprocess.run
    environments = []

    def record_environment(*args, **kwargs):
        environments.append(kwargs["env"])
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_environment)
    first = plan.run()
    second = plan.run()
    assert first.value == second.value
    assert all(environment["OMP_NUM_THREADS"] == "1" and
               environment["OPENBLAS_NUM_THREADS"] == "1"
               for environment in environments)


def test_deterministic_gpu_numerics_report_unavailable_implementation():
    @qcl.workflow
    def rhf(molecule, gpu):
        return qcl.scf.rhf(molecule).to(gpu).energy

    with pytest.raises(qcl.CompileError, match="QCL-E207"):
        rhf.compile(_inputs()[0], qcl.Device("gpu0", "gpu"), deterministic=True)


def test_deterministic_gpu_backed_simulator_reports_unavailable_implementation():
    simulator = qcl.Device("sim0", "qpu", qubits=4,
                           simulator_host="gpu0", simulator_method="statevector")
    with pytest.raises(qcl.CompileError, match="QCL-E207"):
        sampled.compile(*_inputs(), qcl.Device("cpu0", "cpu"), simulator,
                        deterministic=True)


def test_deterministic_compile_rejects_unseedable_cloud_measurement():
    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    with pytest.raises(qcl.CompileError, match="QCL-E207"):
        sampled.compile(*_inputs(), qcl.Cluster.local().cpus[0], cloud,
                        deterministic=True)


def test_circuit_cut_cannot_return_raw_bitstrings():
    cluster = qcl.Cluster.local()
    with pytest.raises(qcl.CompileError, match="QCL-E104"):
        sampled.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0],
                        qcl.CircuitCut(max_cuts=1))


def test_teleport_without_quantum_link_reports_e208():
    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device(f"sim{index}", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector")
            for index in range(2)]
    with pytest.raises(qcl.CompileError, match="QCL-E208"):
        sampled.compile(*_inputs(), cpu, qpus, qcl.Teleport())


def test_quantum_links_are_available_to_partition_checker(tmp_path):
    config = tmp_path / "cluster.toml"
    config.write_text("""
[[cpu]]
id = "cpu0"
threads = 2

[[qpu]]
id = "sim0"
qubits = 4
simulator = {method = "statevector", host = "cpu0"}

[[qpu]]
id = "sim1"
qubits = 4
simulator = {method = "statevector", host = "cpu0"}

[[quantum_link]]
between = ["sim0", "sim1"]
bell_pairs_per_s = 100
fidelity = 0.99
""")
    cluster = qcl.Cluster.load(config)
    assert cluster.qpus[0].quantum_links == (("sim0", "sim1"),)
    assert sampled.compile(*_inputs(), cluster.cpus[0], cluster.qpus[0]).estimate()[
        "shots"] == 128
    with pytest.raises(qcl.CompileError, match="QCL-E203") as failure:
        sampled.compile(*_inputs(), cluster.cpus[0], cluster.qpus, qcl.Teleport())
    assert all(item["code"] != "QCL-E208" for item in failure.value.diagnostics)


def test_circuit_cut_estimate_reports_missing_implementation():
    @qcl.workflow
    def cut(molecule, active, cpu, qpus):
        space = qcl.active_space(qcl.scf.hf(molecule).to(cpu), active).to(cpu)
        operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
        return qcl.estimate(operator, qcl.kernels.hf(space), n_groups=5,
                            shots_per_group=128,
                            partition=qcl.CircuitCut(max_cuts=1)).to(qpus)

    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device(f"sim{index}", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector")
            for index in range(2)]
    with pytest.raises(qcl.CompileError, match="QCL-E203"):
        cut.compile(*_inputs(), cpu, qpus)
