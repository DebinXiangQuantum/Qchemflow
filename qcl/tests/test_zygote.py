"""The pre-warmed worker (qcl._zygote) behaves like a cold worker process."""

import subprocess

import pytest

import qcl
from qcl import _zygote

H2 = qcl.Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")
H4 = qcl.Molecule("H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7", basis="sto-3g")
# Several seconds of FCI (CAS(8,12), 245025 determinants) to interrupt.
H8 = qcl.Molecule("; ".join(f"H 0 0 {0.9 * index:.1f}" for index in range(8)), basis="6-31g")


@qcl.workflow
def ground(molecule, active, cpu):
    space = qcl.active_space(qcl.scf.hf(molecule).to(cpu), active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy


def _plan(molecule=H2, active=None):
    return ground.compile(molecule, active or qcl.Active(2, 2), qcl.Cluster.local().cpus[0])


def _cold_runs(monkeypatch):
    calls = []
    original = subprocess.run

    def record(*args, **kwargs):
        if "qcl._worker" in list(args[0]):
            calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    return calls


def test_warm_and_cold_workers_agree_and_the_warm_one_spawns_no_interpreter(monkeypatch):
    calls = _cold_runs(monkeypatch)
    plan = _plan()
    warm = plan.run()
    assert calls == [] and plan.cpu_core_s > 0
    monkeypatch.setenv("QCL_WARM_WORKER", "0")
    cold = plan.run()
    assert len(calls) == 1
    assert warm.value == pytest.approx(cold.value, abs=1e-12)
    def stable(provenance):
        return {key: value for key, value in provenance.items() if key != "timestamp"}

    assert stable(warm.provenance) == stable(cold.provenance)


def test_a_changed_environment_reaches_the_warm_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMFLOW_AIZYNTH_HOME", str(tmp_path))

    @qcl.workflow
    def routes(cpu):
        return qcl.synthesis.routes("CCO", time_limit_s=5).to(cpu)

    with pytest.raises(RuntimeError, match="CF-E080.*AiZynthFinder"):
        routes.compile(qcl.Cluster.local().cpus[0]).run()


def test_the_warm_worker_enforces_the_wall_time_limit():
    plan = _plan(H8, qcl.Active(8, 12))
    with pytest.raises(RuntimeError, match="wall-time limit"):
        plan.run(timeout_s=0.05)
    # The zygote survives a killed worker.
    assert _plan().run().value < -1.1


def test_the_warm_worker_enforces_the_cpu_core_time_limit():
    plan = _plan(H8, qcl.Active(8, 12))
    with pytest.raises(RuntimeError, match="CPU core-time limit"):
        plan.run(cpu_core_s=0.01)


def test_worker_failures_keep_their_diagnostics():
    @qcl.workflow
    def broken(cpu):
        space = qcl.active_space(qcl.scf.hf(H4).to(cpu), qcl.Active(2, 2)).to(cpu)
        return qcl.embedding.ewf(space.hamiltonian, space.sector, bath="full").to(cpu)

    with pytest.raises(RuntimeError, match=r"(?s)QCL-E503: worker failed.*CF-E020"):
        broken.compile(qcl.Cluster.local().cpus[0]).run()


def test_disabled_or_busy_zygotes_fall_back_to_cold_workers(monkeypatch):
    monkeypatch.setenv("QCL_WARM_WORKER", "0")
    assert _zygote.run("{}", 1.0) is None
