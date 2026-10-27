import pytest

import qcl


@qcl.workflow
def h2_samples(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sample(qcl.kernels.hf(space), shots=128, seed=7).to(qpu)


def _plan(budget, qpu=None):
    cluster = qcl.Cluster.local()
    return h2_samples.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                              qcl.Active(2, 2), cluster.cpus[0],
                              qpu or cluster.qpus[0], budget=budget)


def test_program_can_lower_but_not_raise_shot_limit(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        _plan(qcl.Budget(qpu_shots=64)).run()
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        _plan(qcl.Budget(qpu_shots=256)).run()
    valid = _plan(qcl.Budget(qpu_shots=128))
    assert valid.run().shots == 128
    assert valid.hash != _plan(None).hash


def test_program_money_limit_applies_only_to_metered_cloud(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        _plan(qcl.Budget(qpu_cost=0.01)).run()

    cloud = qcl.Device("mq0", "qpu", qubits=24, site="remote",
                       billing="metered", provider="lqcloud", backend="MQ02")
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        _plan(qcl.Budget(qpu_cost=0.01), cloud).run()

    forecast = _plan(None, cloud).estimate()["qpu_cost"]
    assert forecast > 0
    permitted = _plan(qcl.Budget(qpu_cost=forecast + 1), cloud)
    assert permitted._check_qpu_policy()["max_cost"] == pytest.approx(forecast + 1)

    policy.write_text("[budget]\nqpu_shots = 128\n"
                      "[cloud_qpu]\nqpu_cost = 10\ncurrency = 'CNY'\n")
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        _plan(qcl.Budget(qpu_cost=11), cloud)._check_qpu_policy()


@pytest.mark.parametrize("kwargs", [
    {"qpu_shots": -1}, {"qpu_shots": True}, {"qpu_s": float("nan")},
    {"qpu_cost": -1}, {"wall_s": 0},
])
def test_program_budget_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        qcl.Budget(**kwargs)


def test_operator_wall_time_caps_cpu_and_program_budget(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nwall_time = '20s'\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))

    @qcl.workflow
    def energy(molecule, cpu):
        return qcl.scf.hf(molecule).to(cpu).energy

    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    plan = energy.compile(molecule, cpu, budget=qcl.Budget(wall_s=21))
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        plan.run()

    accepted = energy.compile(molecule, cpu, budget=qcl.Budget(wall_s=10))
    assert accepted.run().value == pytest.approx(-1.1167593073964, abs=1e-8)

    policy.write_text("[budget]\nwall_time = 'bogus'\n")
    with pytest.raises(qcl.CompileError, match="QCL-E301"):
        accepted.run()
