import pytest

import qcl

H2 = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
CPU = qcl.Device("cpu0", "cpu")


@qcl.workflow
def scaled_energy(molecule, cpu):
    energy = qcl.scf.hf(molecule).to(cpu).energy
    return 2 * energy - energy / 2, -energy


def test_estimate_scalar_arithmetic_preserves_units_and_value():
    plan = scaled_energy.compile(H2, CPU)
    assert [node["op"] for node in plan.graph_json()["nodes"]].count(
        "estimate.scale") == 3
    combined, negated = plan.run()
    assert combined.value == pytest.approx(-1.6751389610947, abs=1e-8)
    assert negated.value == pytest.approx(1.1167593073964, abs=1e-8)
    assert combined.unit == negated.unit == "hartree"


@qcl.workflow
def invalid_scale(molecule, cpu, factor):
    return qcl.scf.hf(molecule).to(cpu).energy * factor


@pytest.mark.parametrize("factor", [0.0, float("inf"), "2", True])
def test_invalid_scalar_arithmetic_is_diagnosed(factor):
    if factor == 0.0:
        plan = invalid_scale.compile(H2, CPU, factor)
        assert plan.run().value == 0
    else:
        with pytest.raises(qcl.CompileError, match="QCL-E101"):
            invalid_scale.compile(H2, CPU, factor)


@qcl.workflow
def pauli_energy(molecule, cpu, qpu, scale):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    energy = qcl.estimate(operator, qcl.kernels.hf(space), n_groups=5,
                          shots_per_group=128, seed=7).to(qpu)
    return scale * energy - energy if scale else energy


@qcl.workflow
def pauli_statistics(molecule, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    energy = qcl.estimate(operator, qcl.kernels.hf(space), n_groups=5,
                          shots_per_group=128, seed=7).to(qpu)
    doubled = 2 * energy
    return (qcl.mean([energy, doubled]), qcl.compare(doubled, energy),
            qcl.rank([energy, 1.02 * energy]))


def test_shared_samples_keep_covariance(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    cpu = qcl.Device("cpu0", "cpu")
    simulator = qcl.Device("sim-cpu", "qpu", qubits=20,
                           simulator_host="cpu0", simulator_method="statevector")
    baseline = pauli_energy.compile(H2, cpu, simulator, 0).run()
    combined = pauli_energy.compile(H2, cpu, simulator, 2).run()
    assert combined.value == pytest.approx(baseline.value, abs=1e-10)
    assert combined.variance == pytest.approx(baseline.variance, abs=1e-12)
    averaged, compared, ranked = pauli_statistics.compile(H2, cpu, simulator).run()
    assert averaged.value == pytest.approx(1.5 * baseline.value, abs=1e-10)
    assert averaged.variance == pytest.approx(2.25 * baseline.variance, abs=1e-12)
    assert compared.combined_sigma == pytest.approx(
        baseline.variance ** 0.5, abs=1e-10)
    assert ranked.unresolved == ()
