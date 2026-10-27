import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("qcl")

import qcl
from chemflow.agent.qcl import QclAgent, run_cli
from chemflow.agent.task import TaskGate, TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError
from chemflow.signing import TaskSigner, generate_keypair


@qcl.workflow
def energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy


@qcl.workflow
def sci_energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sci.solve(space.hamiltonian, space.sector,
                         select_cutoff=1e-10, total_spin=0).to(cpu).energy


@qcl.workflow
def native_fci_energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector,
                         engine="native").to(cpu).energy


@qcl.workflow
def sampled_energy(molecule, active, cpu, simulator):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    bits = qcl.sample(qcl.kernels.hf(space), shots=128, seed=7).to(simulator)
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu)


@qcl.workflow
def hf_energy(molecule, cpu):
    return qcl.scf.hf(molecule).to(cpu).energy


@qcl.workflow
def pbe_energy(molecule, cpu, xc="pbe"):
    return qcl.dft.rks(molecule, xc=xc).to(cpu).energy


@qcl.workflow
def uhf_energy(molecule, cpu):
    return qcl.scf.uhf(molecule).to(cpu).energy


@qcl.workflow
def uks_energy(molecule, cpu, xc="pbe"):
    return qcl.dft.uks(molecule, xc=xc).to(cpu).energy


@qcl.workflow
def rohf_energy(molecule, cpu):
    return qcl.scf.rohf(molecule).to(cpu).energy


@qcl.workflow
def vqe_energy(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.vqe.sampled(space.hamiltonian, space.sector,
                           max_evaluations=3, shots_per_group=128,
                           n_groups=5, seed=7).to(qpu)


@qcl.workflow
def pauli_energy(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.estimate(operator, qcl.kernels.hf(space),
                        n_groups=5, shots_per_group=128, seed=7).to(qpu)


@qcl.workflow
def distributed_pauli_energy(molecule, active, cpu, qpus):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.estimate(operator, qcl.kernels.hf(space),
                        n_groups=5, shots_per_group=128, seed=7).to(qpus)


def _agent(budget=None):
    store = TaskStore("qcl-test-key")
    spec = TaskSpec.model_validate({
        "question": "H2 ground energy",
        "target": {
            "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"},
            "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
        },
        "gates": [], "budget": budget or {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_compiles_and_runs_qcl_against_signed_task():
    agent = _agent()
    review = agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    assert review.explanation["operations"][-1]["op"] == "fci.solve"
    assert review.estimate["qpu_cost"] == 0
    result = agent.run(review)
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-9)
    assert result.provenance["implementation"] == "fci.solve/chemflow-pyscf"
    assert result.provenance["task_record_hash"] == agent.record.hash
    assert result.provenance["plan_hash"] == review.plan_hash


def test_agent_accepts_native_ci_against_signed_exact_gate():
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-9,
                           unit="hartree"),)})
    store = TaskStore("qcl-native-ci-agent")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(native_fci_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    assert review.explanation["operations"][-1]["implementation"] == "fci.solve/qcl-rust"
    result = agent.run(review)
    assert result.provenance["signed_gates"][0]["passed"]
    assert result.provenance["implementation"] == "fci.solve/qcl-rust"


def test_agent_selects_a_gate_passing_qcl_candidate():
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 60, "cpu_core_s": 60}})
    store = TaskStore("qcl-candidate-screen")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    reviews = {
        "hf": agent.compile(hf_energy, molecule, cpu),
        "native": agent.compile(native_fci_energy, molecule, qcl.Active(2, 2), cpu),
    }
    selection = agent.select(reviews)
    assert selection.candidate == "native"
    assert selection.result.value == pytest.approx(-1.1372838344885, abs=1e-9)
    assert [item.passed for item in selection.outcomes] == [False, True]
    assert all(item.wall_s > 0 and item.cpu_core_s > 0 for item in selection.outcomes)
    assert reviews["hf"].reference_hash == reviews["native"].reference_hash
    assert reviews["hf"].reference_plan.cpu_core_s > 0
    assert reviews["native"].reference_plan.cpu_core_s is None
    assert all(item.compute_s > 0 for item in selection.outcomes)


def test_agent_rejects_reference_output_changed_after_review():
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),)})
    store = TaskStore("qcl-reference-output-integrity")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(native_fci_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    original_hash = review.reference_plan.hash
    review.reference_plan._output = qcl.Estimate(
        -1.1372838344885, "hartree", 0, {})
    assert review.reference_plan.hash == original_hash
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.run(review)


def test_agent_candidate_screen_requires_a_signed_numeric_gate():
    agent = _agent()
    review = agent.compile(energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.select({"candidate": review})


def test_agent_candidate_screen_rejects_invalid_remaining_limits(monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 30, "cpu_core_s": 30}})
    store = TaskStore("qcl-screen-limits")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(hf_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    monkeypatch.setattr(agent, "run", lambda *_args, **_kwargs: pytest.fail(
        "candidate executed with invalid remaining limits"))
    for limits in ({"cpu_core_s": float("nan")}, {"wall_s": 31}):
        with pytest.raises(AdapterError, match="CF-E070"):
            agent.select({"hf": review}, limits=limits)


def test_agent_candidate_screen_returns_revise_when_no_plan_passes():
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),)})
    store = TaskStore("qcl-no-passing-candidate")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(hf_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    selection = agent.select({"hf": review})
    assert selection.decision == "revise"
    assert selection.candidate is None
    assert selection.result is None
    assert not selection.outcomes[0].passed


def test_agent_candidate_screen_refuses_aggregate_qpu_shots_before_running(monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 30, "qpu_shots": 128, "qpu_s": 10},
        "allowed_effects": ("pure", "simulator")})
    store = TaskStore("qcl-screen-shots")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    review = agent.compile(sampled_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    monkeypatch.setattr(agent, "run", lambda *_args, **_kwargs: pytest.fail(
        "candidate executed before aggregate budget check"))
    with pytest.raises(AdapterError, match="CF-E070"):
        agent.select({"one": review, "two": review})


def test_agent_runs_sci_candidate_against_signed_fci_gate():
    store = TaskStore("qcl-sci-key")
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),)})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(sci_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    result = agent.run(review)
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)
    assert result.provenance["signed_gates"][0]["passed"]


def test_agent_rejects_different_question_before_run():
    agent = _agent()
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.80", "sto-3g"),
                      qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Active(2, 1), qcl.Device("cpu0", "cpu"))


def test_agent_rejects_zero_signed_gpu_time_before_execution():
    store = TaskStore("qcl-gpu-time-key")
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "gpu_s": 0}})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E070"):
        agent.compile(hf_energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Device("gpu0", "gpu"))


def test_agent_enforces_signed_cpu_core_time():
    store = TaskStore("qcl-cpu-time-key")
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "cpu_core_s": 100}})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(hf_energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    result = agent.run(review)
    assert result.value == pytest.approx(-1.1167593073964255, abs=1e-8)
    assert 0 < review.plan.cpu_core_s < 100


def test_agent_rejects_exhausted_cpu_core_time():
    store = TaskStore("qcl-cpu-time-exhausted-key")
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "cpu_core_s": 0.000001}})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(hf_energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    with pytest.raises(RuntimeError, match="CPU core-time"):
        agent.run(review)
    assert review.plan.cpu_core_s > 0


def test_qcl_agent_cli_accounts_for_cpu_core_time(tmp_path):
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "cpu_core_s": 100}})
    private, public = generate_keypair(b"cpu-time-test")
    signed = TaskSigner(private, b"cpu-time-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "public.pem", "output"))
    task.write_text(signed.model_dump_json())
    key.write_bytes(public)
    args = Namespace(task=task, public_key=key, program=None, cluster=None,
                     device=None, out=out, dry_run=False)
    run_cli(args)
    summary = json.loads((out / "summary.json").read_text())
    assert 0 < summary["spent"]["cpu_core_s"] < spec.budget["cpu_core_s"]
    ledger = out / f"{signed.content_hash}.ledger.json"
    spent = json.loads(ledger.read_text())
    spent["cpu_core_s"] = spec.budget["cpu_core_s"]
    ledger.write_text(json.dumps(spent))
    with pytest.raises(AdapterError, match="CF-E070.*cpu_core_s"):
        run_cli(args)
    assert json.loads(ledger.read_text())["cpu_core_s"] == spec.budget["cpu_core_s"]


def test_qcl_agent_cli_records_cpu_core_time_after_failure(tmp_path):
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "cpu_core_s": 1e-6}})
    private, public = generate_keypair(b"cpu-time-failure-test")
    signed = TaskSigner(private, b"cpu-time-failure-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "public.pem", "output"))
    task.write_text(signed.model_dump_json())
    key.write_bytes(public)
    with pytest.raises(RuntimeError, match="CPU core-time"):
        run_cli(Namespace(task=task, public_key=key, program=None, cluster=None,
                          device=None, out=out, dry_run=False))
    ledger = out / f"{signed.content_hash}.ledger.json"
    assert json.loads(ledger.read_text())["cpu_core_s"] > 0


def test_agent_shares_cpu_core_time_with_reference_plan():
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-7,
                           unit="hartree"),),
        "budget": {"wall_s": 30, "cpu_core_s": 100},
    })
    store = TaskStore("qcl-cpu-reference-key")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    result = agent.run(review)
    assert result.provenance["signed_gates"][0]["passed"]
    assert review.reference_plan.cpu_core_s > 0
    assert review.plan.cpu_core_s > 0
    assert review.reference_plan.cpu_core_s + review.plan.cpu_core_s < 100


def test_agent_rejects_derived_zero_as_ground_energy():
    @qcl.workflow
    def zero(molecule, cpu):
        result = qcl.scf.hf(molecule).to(cpu).energy
        return result - result

    with pytest.raises(AdapterError, match="CF-E120"):
        _agent().compile(zero, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                         qcl.Device("cpu0", "cpu"))


def test_qcl_agent_does_not_silently_ignore_signed_gates():
    store = TaskStore("qcl-gate-key")
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="relation_at_least", relation="equivalent"),)})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))


@pytest.mark.parametrize("threshold,passed", [(0.03, True), (0.01, False)])
def test_qcl_agent_evaluates_signed_absolute_error_gate(threshold, passed):
    store = TaskStore("qcl-gate-key")
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=threshold,
                           unit="hartree"),)})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    review = agent.compile(hf_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"))
    result = agent.run(review)
    gate = result.provenance["signed_gates"][0]
    assert gate["passed"] is passed
    assert gate["error"] == pytest.approx(0.0205245271, abs=1e-7)


@pytest.mark.parametrize("threshold,decision", [(0.03, "gate_passed"),
                                                 (0.01, "gate_failed")])
def test_qcl_agent_cli_reports_signed_gate_decision(tmp_path, threshold, decision):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=threshold,
                           unit="hartree"),)})
    private, public = generate_keypair(b"test-password")
    record = TaskSigner(private, b"test-password").freeze(spec, "reviewer")
    task_path = tmp_path / "task.json"
    key_path = tmp_path / "public.pem"
    program_path = tmp_path / "program.py"
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public)
    program_path.write_text("""import qcl

@qcl.workflow
def energy(molecule, cpu):
    return qcl.scf.hf(molecule).to(cpu).energy

def build(task_spec, cluster):
    target = task_spec.target
    return energy, (qcl.Molecule(target.molecule['atoms'],
                                target.electronic_model['basis']), cluster.cpus[0]), {}
""")
    out = tmp_path / "out"
    run_cli(Namespace(task=task_path, public_key=key_path, program=program_path,
                      cluster=None, out=out, dry_run=False))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["decision"] == decision
    assert summary["signed_gates"][0]["passed"] is (decision == "gate_passed")
    assert json.loads((out / "qcl-review.json").read_text())["reference_plan_hash"]


def test_qcl_agent_cli_builds_signed_ground_task_without_program(tmp_path, monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-8,
                           unit="hartree"),)})
    private, public = generate_keypair(b"test-password")
    record = TaskSigner(private, b"test-password").freeze(spec, "reviewer")
    task_path = tmp_path / "task.json"
    key_path = tmp_path / "public.pem"
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public)
    out = tmp_path / "out"

    from chemflow.cli import main

    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task_path),
                                   "--public-key", str(key_path),
                                   "--engine", "qcl", "--out", str(out)])
    main()
    summary = json.loads((out / "summary.json").read_text())
    review = json.loads((out / "qcl-review.json").read_text())
    assert summary["decision"] == "gate_passed"
    assert summary["record_hash"] == record.hash
    assert summary["spent"]["wall_s"] > 0
    assert "QCL result" in (out / "report.md").read_text()
    assert (out / "qcl-profile.json").exists()
    error_budget = json.loads((out / "qcl-error-budget.json").read_text())
    assert error_budget["statistical"]["variance"] == 0
    assert error_budget["systematic"]["upper_bound"] is None
    assert error_budget["systematic"]["max_observed_absolute_error"] == pytest.approx(
        summary["signed_gates"][0]["error"])
    assert error_budget["systematic"]["signed_gate_observations"][0][
        "reference_plan_hash"] == review["reference_plan_hash"]
    assert summary["paths"]["error_budget"] == str(out / "qcl-error-budget.json")
    with sqlite3.connect(out / "results.sqlite") as connection:
        row = connection.execute("SELECT record_hash, plan_hash, decision, result "
                                 "FROM qcl_runs WHERE id=?",
                                 (summary["result_id"],)).fetchone()
    assert row[:3] == (record.hash, summary["plan_hash"], "gate_passed")
    assert json.loads(row[3])["value"] == pytest.approx(-1.1372838344885, abs=1e-9)
    assert summary["paths"]["results"] == str(out / "results.sqlite")
    assert (out / f"{record.content_hash}.ledger.json").exists()
    assert [item["op"] for item in review["explanation"]["operations"]] == [
        "scf.rhf", "active_space", "fci.solve"]
    assert json.loads((out / "qcl-result.json").read_text())["value"] == pytest.approx(
        -1.1372838344885, abs=1e-9)


def test_qcl_agent_cli_screens_ground_energy_candidates(tmp_path, monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 90, "cpu_core_s": 90}})
    private, public = generate_keypair(b"qcl-screen-test")
    record = TaskSigner(private, b"qcl-screen-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "key.pem", "out"))
    task.write_text(record.model_dump_json())
    key.write_bytes(public)
    from chemflow.cli import main

    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task), "--public-key", str(key),
                                   "--engine", "qcl", "--screen", "--out", str(out)])
    main()
    summary = json.loads((out / "summary.json").read_text())
    screen = json.loads((out / "qcl-screen.json").read_text())
    assert summary["decision"] == "stop"
    assert summary["candidate"] in ("pyscf_fci", "native_ci")
    assert summary["record_hash"] == record.hash
    assert screen["hf"]["passed"] is False
    assert screen[summary["candidate"]]["passed"] is True
    assert summary["spent"]["cpu_core_s"] > 0
    assert json.loads((out / "qcl-result.json").read_text())["value"] == pytest.approx(
        -1.1372838344885, abs=1e-9)
    assert (out / "results.sqlite").exists()
    assert summary["paths"]["error_budget"] == str(out / "qcl-error-budget.json")
    budget = json.loads((out / "qcl-error-budget.json").read_text())
    assert budget["systematic"]["signed_gate_observations"][0][
        "reference_plan_hash"] == json.loads((out / "qcl-review.json").read_text())["candidates"][
            summary["candidate"]]["reference_plan_hash"]
    first_cpu = summary["spent"]["cpu_core_s"]
    main()
    repeated = json.loads((out / "summary.json").read_text())
    assert repeated["spent"]["cpu_core_s"] > first_cpu


def test_qcl_agent_cli_screens_authored_qcl_candidates(tmp_path, monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 60, "cpu_core_s": 60}})
    private, public = generate_keypair(b"authored-qcl-screen")
    record = TaskSigner(private, b"authored-qcl-screen").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "key.pem", "out"))
    task.write_text(record.model_dump_json())
    key.write_bytes(public)
    programs = []
    for name, solver in (("control", "hf"), ("candidate", "native")):
        path = tmp_path / f"{name}.py"
        path.write_text("import qcl\n"
                        "@qcl.workflow\n"
                        "def ground(molecule, cpu):\n"
                        "    mean_field = qcl.scf.hf(molecule).to(cpu)\n"
                        + ("    return mean_field.energy\n" if solver == "hf" else
                           "    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)\n"
                           "    return qcl.fci.solve(space.hamiltonian, space.sector, "
                           "engine='native').to(cpu).energy\n")
                        + "def build(task_spec, cluster):\n"
                        "    target = task_spec.target\n"
                        "    molecule = qcl.Molecule(target.molecule['atoms'], 'sto-3g')\n"
                        "    return ground, (molecule, cluster.cpus[0]), {}\n")
        programs.append(path)
    from chemflow.cli import main

    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task), "--public-key", str(key),
                                   "--engine", "qcl", "--screen",
                                   "--candidate-program", str(programs[0]),
                                   "--candidate-program", str(programs[1]),
                                   "--out", str(out)])
    main()
    summary = json.loads((out / "summary.json").read_text())
    screen = json.loads((out / "qcl-screen.json").read_text())
    assert summary["candidate"] == "candidate"
    assert screen["control"]["passed"] is False
    assert screen["candidate"]["passed"] is True


def test_qcl_agent_cli_reserves_screened_simulator_shots(tmp_path, monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 90, "cpu_core_s": 90,
                   "qpu_shots": 128, "qpu_s": 10},
        "allowed_effects": ("pure", "simulator")})
    private, public = generate_keypair(b"qcl-screen-sim")
    record = TaskSigner(private, b"qcl-screen-sim").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "key.pem", "out"))
    task.write_text(record.model_dump_json())
    key.write_bytes(public)
    program = tmp_path / "simulator.py"
    program.write_text("""import qcl

@qcl.workflow
def ground(molecule, cpu, simulator):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(2, 2)).to(cpu)
    bits = qcl.sample(qcl.kernels.hf(space), shots=128, seed=7).to(simulator)
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu)

def build(task_spec, cluster):
    molecule = qcl.Molecule(task_spec.target.molecule['atoms'], 'sto-3g')
    return ground, (molecule, cluster.cpus[0], cluster.qpus[0]), {}
""")
    monkeypatch.setenv("QCL_POLICY_FILE", str(
        Path(__file__).resolve().parents[1] / "qcl/apps/local-policy.toml"))
    from chemflow.cli import main

    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task), "--public-key", str(key),
                                   "--engine", "qcl", "--screen",
                                   "--candidate-program", str(program),
                                   "--out", str(out)])
    main()
    ledger_path = out / f"{record.content_hash}.ledger.json"
    assert json.loads(ledger_path.read_text())["qpu_shots"] == 128
    with pytest.raises(SystemExit):
        main()
    assert json.loads(ledger_path.read_text())["qpu_shots"] == 128


def test_qcl_cloud_cost_reservation_accumulates_across_runs(tmp_path):
    from chemflow.agent.qcl import _reserve_cloud_cost
    from chemflow.runtime.budget import BudgetExceeded, BudgetLedger
    from chemflow.types import CostRecord

    path = tmp_path / "ledger.json"
    signed_budget = {"wall_s": 30, "qpu_cost": 5}
    first = BudgetLedger({"wall_s": 30}, path)
    _reserve_cloud_cost(first, signed_budget, {"currency": "CNY", "qpu_cost": 3})
    first.record(CostRecord())
    second = BudgetLedger({"wall_s": 30}, path)
    assert second.spent["qpu_cost"] == 3
    with pytest.raises(BudgetExceeded, match="CF-E070"):
        _reserve_cloud_cost(second, signed_budget,
                            {"currency": "CNY", "qpu_cost": 3})
    assert second.spent["qpu_cost"] == 3


def test_qcl_agent_screen_rejects_mixed_cloud_currencies(monkeypatch):
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-6,
                           unit="hartree"),),
        "budget": {"wall_s": 30}})
    store = TaskStore("qcl-mixed-cloud-currency")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    hf = agent.compile(hf_energy, molecule, cpu)
    exact = agent.compile(native_fci_energy, molecule, qcl.Active(2, 2), cpu)
    hf.estimate.update(qpu_cost=1.0, currency="CNY")
    exact.estimate.update(qpu_cost=1.0, currency="USD")
    monkeypatch.setattr(agent, "run", lambda *_args, **_kwargs: pytest.fail(
        "mixed-currency screen started execution"))
    with pytest.raises(AdapterError, match="CF-E070"):
        agent.select({"hf": hf, "exact": exact})


def test_qcl_agent_periodic_target_checks_signed_system_and_mesh():
    from chemflow.agent.qcl import _builtin_program

    system = {
        "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
        "atoms": "C 0 0 0; C 0.8917 0.8917 0.8917",
        "basis": "gth-szv", "pseudo": "gth-pade", "unit": "angstrom",
    }

    @qcl.workflow
    def periodic_energy(system, device, kmesh=(1, 1, 1)):
        return qcl.periodic.scf(system, kmesh=kmesh).to(device)

    store = TaskStore("qcl-periodic-key")
    spec = TaskSpec.model_validate({
        "question": "Periodic cell energy",
        "target": {"kind": "periodic_energy", "system": system,
                   "kmesh": [1, 1, 1], "method": "rhf", "exxdiv": "ewald"},
        "gates": [], "budget": {"wall_s": 180}, "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    workflow, args, kwargs = _builtin_program(spec, cluster)
    review = agent.compile(workflow, *args, **kwargs)
    assert review.plan_hash
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(periodic_energy, system, cluster.cpus[0], (1, 1, 2))
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(periodic_energy, {**system, "pseudo": "gth-blyp"},
                      cluster.cpus[0])


def test_qcl_agent_spectral_target_checks_identity_and_l1_gate(tmp_path):
    from chemflow.agent.qcl import _builtin_program

    model = {"U": 2.0, "eps_bath": [0.2], "t_bath": [0.5], "unit": "hartree"}
    sector = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
    omega = {"start": -4.0, "stop": 4.0, "n": 201, "unit": "hartree"}
    store = TaskStore("qcl-spectral-key")
    spec = TaskSpec.model_validate({
        "question": "Impurity spectral density",
        "target": {"kind": "spectral_function", "model": {"impurity_model": model},
                   "sector": sector, "orbital": 0, "spin": "alpha",
                   "omega": omega, "eta": 0.1},
        "gates": [{"kind": "l1_error_vs_exact", "threshold": 1e-8,
                   "unit": "dimensionless"}],
        "budget": {"wall_s": 60}, "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    workflow, args, kwargs = _builtin_program(spec, cluster)
    review = agent.compile(workflow, *args, **kwargs)
    result = agent.run(review)
    assert result.provenance["signed_gates"][0]["passed"]
    assert result.provenance["signed_gates"][0]["error"] < 1e-8

    @qcl.workflow
    def altered(model, sector, device):
        return qcl.dynamics.spectral_exact(
            model, sector, orbital=0, spin="alpha", omega=omega, eta=0.2).to(device)

    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(altered, model, sector, cluster.cpus[0])

    private, public = generate_keypair(b"spectral-test")
    signed = TaskSigner(private, b"spectral-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "public.pem", "output"))
    task.write_text(signed.model_dump_json())
    key.write_bytes(public)
    run_cli(Namespace(task=task, public_key=key, program=None, cluster=None,
                      device=None, out=out, dry_run=False))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["decision"] == "gate_passed"
    assert "201 frequencies" in (out / "report.md").read_text()
    assert len(json.loads((out / "qcl-result.json").read_text())["spectral"]) == 201


@pytest.mark.parametrize("model,molecule,operation", [
    ({"basis": "sto-3g", "reference": "UHF"},
     {"atoms": "Li 0 0 0", "spin": 1}, "scf.uhf"),
    ({"basis": "sto-3g", "reference": "UHF", "dft_functional": "pbe"},
     {"atoms": "Li 0 0 0", "spin": 1}, "dft.uks"),
    ({"basis": "sto-3g", "reference": "RHF", "dft_functional": "pbe"},
     {"atoms": "H 0 0 0; H 0 0 0.74"}, "dft.rks"),
])
def test_builtin_agent_matches_signed_mean_field(model, molecule, operation):
    from chemflow.agent.qcl import _builtin_program

    spec = _agent().record.spec.model_copy(update={
        "target": _agent().record.spec.target.model_copy(update={
            "molecule": molecule, "electronic_model": model,
            "active": "auto", "sector": {}}),
    })
    store = TaskStore("qcl-builtin-key")
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    assert [item["op"] for item in review.explanation["operations"]] == [operation]
    assert agent.run(review).value < 0


def test_agent_rejects_unsupported_signed_electronic_model():
    store = TaskStore("qcl-test-key")
    spec = _agent().record.spec.model_copy(update={
        "target": _agent().record.spec.target.model_copy(update={
            "electronic_model": {"basis": "sto-3g", "dft_functional": "pbe"}})})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))


def test_agent_checks_signed_dft_functional_and_hf_lineage():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        _agent().compile(pbe_energy, molecule, cpu)

    store = TaskStore("qcl-dft-key")
    original = _agent().record.spec
    spec = original.model_copy(update={
        "target": original.target.model_copy(update={
            "electronic_model": {"basis": "sto-3g", "dft_functional": "pbe"}})})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(hf_energy, molecule, cpu)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(pbe_energy, molecule, cpu, xc="b3lyp")
    result = agent.run(agent.compile(pbe_energy, molecule, cpu))
    assert result.value == pytest.approx(-1.1520727951590923, abs=1e-8)
    assert result.provenance["task_record_hash"] == agent.record.hash
    if qcl.Cluster.local().gpus:
        pytest.importorskip("gpu4pyscf")
        gpu = qcl.Cluster.local().gpus[0]
        gpu_result = agent.run(agent.compile(pbe_energy, molecule, gpu))
        assert gpu_result.value == pytest.approx(result.value, abs=1e-8)
        assert gpu_result.provenance["implementation"] == "dft.rks/gpu4pyscf"


def test_agent_checks_signed_unrestricted_methods():
    molecule = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
    cpu = qcl.Device("cpu0", "cpu")
    original = _agent().record.spec
    target = original.target.model_copy(update={
        "molecule": {"atoms": "H 0 0 0", "spin": 1},
        "active": {"n_elec": 1, "n_orb": 1},
        "sector": {"n_alpha": 1, "n_beta": 0, "n_orb": 1},
        "electronic_model": {"basis": "sto-3g", "reference": "UHF"},
    })
    store = TaskStore("qcl-unrestricted-key")
    agent = QclAgent(store.freeze(original.model_copy(update={"target": target}),
                                  "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(hf_energy, molecule, cpu)
    result = agent.run(agent.compile(uhf_energy, molecule, cpu))
    assert result.value == pytest.approx(-0.4665818495572752, abs=1e-9)

    rohf_target = target.model_copy(update={
        "electronic_model": {"basis": "sto-3g", "reference": "ROHF"}})
    rohf_agent = QclAgent(store.freeze(original.model_copy(update={"target": rohf_target}),
                                       "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        rohf_agent.compile(uhf_energy, molecule, cpu)
    assert rohf_agent.run(rohf_agent.compile(rohf_energy, molecule, cpu)).value == (
        pytest.approx(-0.4665818495572752, abs=1e-9))

    dft_target = target.model_copy(update={
        "electronic_model": {"basis": "sto-3g", "reference": "UHF",
                             "dft_functional": "pbe"}})
    dft_agent = QclAgent(store.freeze(original.model_copy(update={"target": dft_target}),
                                      "reviewer"), store)
    with pytest.raises(AdapterError, match="CF-E120"):
        dft_agent.compile(uhf_energy, molecule, cpu)
    with pytest.raises(AdapterError, match="CF-E120"):
        dft_agent.compile(uks_energy, molecule, cpu, xc="b3lyp")
    dft_result = dft_agent.run(dft_agent.compile(uks_energy, molecule, cpu))
    assert dft_result.value == pytest.approx(-0.46437566325051144, abs=1e-8)
    if qcl.Cluster.local().gpus:
        pytest.importorskip("gpu4pyscf")
        gpu_result = dft_agent.run(dft_agent.compile(
            uks_energy, molecule, qcl.Cluster.local().gpus[0]))
        assert gpu_result.value == pytest.approx(dft_result.value, abs=1e-8)
        assert gpu_result.provenance["implementation"] == "dft.uks/gpu4pyscf"


def test_agent_requires_signed_simulator_effect():
    agent = _agent({"wall_s": 30, "qpu_shots": 128})
    cluster = qcl.Cluster.local()
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(sampled_energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])


def test_signed_cloud_cost_budget_is_checked_only_for_metered_qpus():
    agent = _agent({"wall_s": 30, "qpu_shots": 128, "qpu_s": 10,
                    "qpu_cost": 5})
    forecast = {"shots": 128, "qpu_s": 1.0, "qpu_cost": 6.0,
                "currency": "CNY"}
    with pytest.raises(AdapterError, match="CF-E070"):
        agent._check_budget(forecast)
    with pytest.raises(AdapterError, match="CF-E070"):
        agent._check_budget({**forecast, "qpu_cost": None})
    with pytest.raises(AdapterError, match="CF-E070"):
        agent._check_budget({**forecast, "qpu_cost": None, "currency": None})
    agent._check_budget({**forecast, "qpu_cost": 4.0})
    agent._check_budget({**forecast, "qpu_cost": 0.0, "currency": None})


@qcl.workflow
def atomization(species, cpu, hydrogen_coefficient):
    def ground(molecule, active):
        mean_field = (qcl.scf.rohf(molecule).to(cpu) if molecule.spin else
                      qcl.scf.hf(molecule).to(cpu))
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy

    molecule, atom = species
    return hydrogen_coefficient * ground(atom, qcl.Active(1, 1)) - ground(
        molecule, qcl.Active(2, 2))


@qcl.workflow
def atomization_hf(species, cpu):
    molecule, atom = species
    return (2 * qcl.scf.hf(atom).to(cpu).energy -
            qcl.scf.hf(molecule).to(cpu).energy)


def test_qcl_agent_checks_signed_reaction_stoichiometry():
    store = TaskStore("reaction-key")
    spec = TaskSpec.model_validate({
        "question": "H2 atomization energy",
        "target": {"kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
                   "species": [
                       {"name": "H2", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                        "coefficient": -1, "active": {"n_elec": 2, "n_orb": 2},
                        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}},
                       {"name": "H", "molecule": {"atoms": "H 0 0 0", "spin": 1},
                        "coefficient": 2, "active": {"n_elec": 1, "n_orb": 1},
                        "sector": {"n_alpha": 1, "n_beta": 0, "n_orb": 1}},
                   ]},
        "budget": {"wall_s": 60},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.01,
                   "unit": "hartree"}], "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    species = (qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
               qcl.Molecule("H 0 0 0", "sto-3g", spin=1))
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(atomization, species, qcl.Device("cpu0", "cpu"), 3)
    review = agent.compile(atomization, species, qcl.Device("cpu0", "cpu"), 2)
    result = agent.run(review)
    assert result.value == pytest.approx(0.20412013537395, abs=1e-8)
    assert result.provenance["signed_gates"][0]["passed"] is True
    assert result.provenance["task_record_hash"] == agent.record.hash
    hf_review = agent.compile(atomization_hf, species, qcl.Device("cpu0", "cpu"))
    hf_result = agent.run(hf_review)
    assert hf_result.provenance["signed_gates"][0]["passed"] is False


@qcl.workflow
def interaction_energy(components, cpu, wrong_sign=False):
    energies = [qcl.scf.hf(molecule).to(cpu).energy for molecule in components]
    return (energies[0] + energies[1] - energies[2] if wrong_sign else
            energies[0] - energies[1] - energies[2])


def test_qcl_agent_checks_signed_counterpoise_interaction():
    from chemflow.domains.drug import component_specs

    store = TaskStore("interaction-key")
    target = {"kind": "interaction_energy", "complex": {
        "atoms": "H 0 0 0; H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"},
        "fragments": [[0, 1], [2, 3]],
        "electronic_model": {"basis": "sto-3g"}, "counterpoise": True}
    spec = TaskSpec.model_validate({
        "question": "H2 dimer interaction", "target": target,
        "budget": {"wall_s": 90}, "gates": [
            {"kind": "abs_error_vs_exact", "threshold": 0.01,
             "unit": "hartree"}], "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    components = [qcl.Molecule(item["atoms"], "sto-3g") for item in
                  component_specs(target["complex"], target["fragments"],
                                  target["electronic_model"], True)]
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(interaction_energy, components, cpu, wrong_sign=True)
    review = agent.compile(interaction_energy, components, cpu)
    result = agent.run(review)
    assert result.unit == "hartree"
    assert result.provenance["signed_gates"][0]["passed"] is True
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_runs_signed_qpu_simulator_sqd_task(tmp_path, monkeypatch):
    store = TaskStore("qcl-test-key")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "budget": {"wall_s": 30, "qpu_shots": 128},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    review = agent.compile(sampled_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = agent.run(review)
    assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)
    assert result.provenance["sample_source"] == "quantum_simulator"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_runs_signed_simulator_vqe_with_exact_gate(tmp_path, monkeypatch):
    store = TaskStore("qcl-vqe-key")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=0.5,
                           unit="hartree"),),
        "budget": {"wall_s": 60, "qpu_shots": 1920},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    review = agent.compile(vqe_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1920\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = agent.run(review)
    assert result.provenance["signed_gates"][0]["passed"] is True
    assert result.provenance["sample_source"] == "quantum_simulator"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_runs_signed_pauli_estimate_with_exact_gate(tmp_path, monkeypatch):
    store = TaskStore("qcl-pauli-key")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=0.5,
                           unit="hartree"),),
        "budget": {"wall_s": 60, "qpu_shots": 640},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster.local()
    review = agent.compile(pauli_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = agent.run(review)
    assert result.provenance["signed_gates"][0]["passed"] is True
    assert result.provenance["sample_source"] == "quantum_simulator"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_runs_signed_multi_qpu_pauli_estimate(tmp_path, monkeypatch):
    store = TaskStore("qcl-pauli-multi-key")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "budget": {"wall_s": 60, "qpu_shots": 640},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cpu = qcl.Cluster.local().cpus[0]
    qpus = [qcl.Device(f"sim{index}", "qpu", qubits=4,
                       simulator_host="cpu0", simulator_method="statevector")
            for index in range(2)]
    review = agent.compile(distributed_pauli_energy,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), cpu, qpus)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 640\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    result = agent.run(review)
    assert result.provenance["device"] == "multi"
    assert result.provenance["sample_shots"] == 640
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_changed_review_before_execution():
    agent = _agent({"wall_s": 30})
    review = agent.compile(energy, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.run(type(review)(review.plan, "different", review.explanation,
                               review.estimate, review.record_hash))


def test_qcl_agent_command_runs_signed_program(tmp_path, monkeypatch):
    spec = _agent().record.spec
    private, public = generate_keypair(b"test-password")
    record = TaskSigner(private, b"test-password").freeze(spec, "reviewer")
    task_path = tmp_path / "task.json"
    key_path = tmp_path / "public.pem"
    program_path = tmp_path / "program.py"
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public)
    program_path.write_text("""import qcl

@qcl.workflow
def energy(molecule, active, cpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy

def build(task_spec, cluster):
    target = task_spec.target
    molecule = qcl.Molecule(target.molecule['atoms'], target.electronic_model['basis'])
    active = qcl.Active(target.active['n_elec'], target.active['n_orb'])
    return energy, (molecule, active, cluster.cpus[0]), {}
""")
    out = tmp_path / "out"
    run_cli(Namespace(task=task_path, public_key=key_path,
                      program=program_path, cluster=None, out=out, dry_run=False))
    assert (out / "qcl-review.json").exists()
    assert '"value": -1.137283834' in (out / "qcl-result.json").read_text()
    assert json.loads((out / "summary.json").read_text())["decision"] == "executed"
    assert [json.loads(line)["state"] for line in
            (out / "journal.jsonl").read_text().splitlines()] == [
                "QCL_REVIEW", "QCL_EXECUTED"]

    from chemflow.cli import main

    integrated = tmp_path / "integrated"
    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task_path),
                                   "--public-key", str(key_path),
                                   "--program", str(program_path),
                                   "--out", str(integrated), "--dry-run"])
    main()
    assert (integrated / "qcl-review.json").exists()
    assert not (integrated / "qcl-result.json").exists()
    assert json.loads((integrated / "summary.json").read_text())["decision"] == "reviewed"


@qcl.workflow
def frontier_quantity(molecule, cpu, quantity):
    mean_field = qcl.scf.rhf(molecule).to(cpu)
    if quantity == "gap":
        return mean_field.lumo - mean_field.homo
    return getattr(mean_field, quantity)


def _frontier_agent(quantity="gap", method="hf"):
    store = TaskStore("qcl-frontier-key")
    spec = TaskSpec.model_validate({
        "question": "H2 frontier orbital energy",
        "target": {"kind": "frontier_orbitals",
                   "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                   "electronic_model": {"basis": "sto-3g"},
                   "level": {"method": method,
                             **({"functional": "pbe"} if method == "dft" else {})},
                   "quantity": quantity},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_frontier_gap():
    agent = _frontier_agent()
    review = agent.compile(frontier_quantity,
                           qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                           qcl.Device("cpu0", "cpu"), "gap")
    result = agent.run(review)
    assert result.value == pytest.approx(1.24969735174578, abs=1e-9)
    assert result.unit == "hartree"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_frontier_quantity_mismatch_and_unsupported_method():
    agent = _frontier_agent("homo")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(frontier_quantity,
                      qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                      qcl.Device("cpu0", "cpu"), "lumo")
    with pytest.raises(AdapterError, match="CF-E120"):
        _frontier_agent("gap", "dft").compile(
            frontier_quantity, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
            qcl.Device("cpu0", "cpu"), "gap")


@qcl.workflow
def solvation_quantity(molecule, cpu, solvent, reverse=False):
    eri = qcl.int2e(molecule).to(cpu)
    gas = qcl.scf.rhf(molecule, eri=eri).to(cpu).energy
    liquid = qcl.scf.rhf(molecule, eri=eri, solvent=solvent).to(cpu).energy
    return gas - liquid if reverse else liquid - gas


def _solvation_agent(solvent_model="pcm", eps=78.355, method="hf", functional=None,
                     spin=0):
    store = TaskStore("qcl-pcm-key")
    spec = TaskSpec.model_validate({
        "question": "H2 water solvation free energy",
        "target": {"kind": "solvation_free_energy",
                   "molecule": {"atoms": "H 0 0 0" if spin else
                                "H 0 0 0; H 0 0 0.74", **({"spin": spin} if spin else {})},
                   "electronic_model": {"basis": "sto-3g",
                                        **({"reference": "UHF"} if spin else {}),
                                        **({"dft_functional": functional}
                                           if functional else {})},
                   "level": {"method": method,
                             **({"functional": functional} if functional else {})},
                   "solvent": {"model": solvent_model, "name": "water", "eps": eps}},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_pcm_solvation_difference():
    agent = _solvation_agent()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    review = agent.compile(solvation_quantity, molecule, qcl.Device("cpu0", "cpu"),
                           qcl.PCM("water", eps=78.355))
    result = agent.run(review)
    assert result.value < 0
    assert result.provenance["task_record_hash"] == agent.record.hash


@qcl.workflow
def dft_solvation_quantity(molecule, device, solvent, xc="pbe", reverse=False):
    method = qcl.dft.uks if molecule.spin else qcl.dft.rks
    gas = method(molecule, xc=xc).to(device).energy
    liquid = method(molecule, xc=xc, solvent=solvent).to(device).energy
    return gas - liquid if reverse else liquid - gas


@pytest.mark.parametrize("solvent", [qcl.PCM("water"), qcl.SMD("water")])
def test_agent_runs_signed_dft_solvation(solvent):
    agent = _solvation_agent("smd" if isinstance(solvent, qcl.SMD) else "pcm",
                             None if isinstance(solvent, qcl.SMD) else solvent.eps,
                             method="dft", functional="pbe")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    review = agent.compile(dft_solvation_quantity, molecule, cpu, solvent)
    assert [item["op"] for item in review.explanation["operations"]] == [
        "dft.rks", "dft.rks"]
    if isinstance(solvent, qcl.SMD):
        pytest.importorskip("pyscf.solvent.smd")
    try:
        result = agent.run(review)
    except RuntimeError as error:
        if isinstance(solvent, qcl.SMD) and "SMD module is not available" in str(error):
            pytest.skip("PySCF was built without SMD")
        raise
    assert result.provenance["operands"][0]["method"]["functional"] == "pbe"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_unsigned_dft_solvation_changes():
    agent = _solvation_agent(method="dft", functional="pbe")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(dft_solvation_quantity, molecule, cpu, qcl.PCM("water"), xc="b3lyp")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(dft_solvation_quantity, molecule, cpu, qcl.PCM("water"), reverse=True)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(dft_solvation_quantity, molecule, cpu, qcl.PCM("toluene"))


def test_agent_runs_signed_open_shell_dft_solvation():
    agent = _solvation_agent(method="dft", functional="pbe", spin=1)
    molecule = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
    cpu = qcl.Device("cpu0", "cpu")
    result = agent.run(agent.compile(dft_solvation_quantity, molecule, cpu,
                                     qcl.PCM("water")))
    assert result.provenance["operands"][0]["method"]["family"] == "UKS"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_restricted_reference_for_open_shell_solvation():
    agent = _solvation_agent(method="dft", functional="pbe", spin=1)
    molecule = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)

    @qcl.workflow
    def restricted(mol, device):
        gas = qcl.dft.rks(mol, xc="pbe").to(device).energy
        liquid = qcl.dft.rks(mol, xc="pbe", solvent=qcl.PCM("water")).to(device).energy
        return liquid - gas

    with pytest.raises(qcl.CompileError, match="QCL-E101"):
        agent.compile(restricted, molecule, qcl.Device("cpu0", "cpu"))


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("solvent", [qcl.PCM("water"), qcl.SMD("water")])
def test_agent_runs_signed_open_shell_dft_solvation_on_gpu(solvent, ordinal):
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    agent = _solvation_agent("smd" if isinstance(solvent, qcl.SMD) else "pcm",
                             None if isinstance(solvent, qcl.SMD) else solvent.eps,
                             method="dft", functional="pbe", spin=1)
    molecule = qcl.Molecule("H 0 0 0", "sto-3g", spin=1)
    result = agent.run(agent.compile(dft_solvation_quantity,
                                     molecule, cluster.gpus[ordinal], solvent))
    assert result.provenance["operands"][0]["method"]["family"] == "UKS"
    assert result.provenance["operands"][0]["device"] == f"gpu{ordinal}"
    assert result.provenance["task_record_hash"] == agent.record.hash


@pytest.mark.parametrize("device_kind", ["cpu", "gpu"])
def test_agent_builtin_signed_dft_solvation(device_kind):
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent(method="dft", functional="pbe")
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device("gpu0", "gpu")
    cluster = qcl.Cluster((cpu,), (gpu,), ())
    workflow, args, kwargs = _builtin_program(
        agent.record.spec, cluster, device_id=gpu.id if device_kind == "gpu" else cpu.id)
    review = agent.compile(workflow, *args, **kwargs)
    assert [item["op"] for item in review.explanation["operations"]] == [
        "dft.rks", "dft.rks"]
    assert review.explanation["operations"][-1]["implementation"].endswith("-pcm")


@pytest.mark.parametrize("ordinal", [0, 1])
def test_agent_builtin_signed_open_shell_dft_solvation_on_gpu(ordinal):
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent(method="dft", functional="pbe", spin=1)
    gpu_workflow, gpu_args, gpu_kwargs = _builtin_program(
        agent.record.spec, cluster, device_id=cluster.gpus[ordinal].id)
    cpu_workflow, cpu_args, cpu_kwargs = _builtin_program(
        agent.record.spec, cluster)
    gpu = agent.run(agent.compile(gpu_workflow, *gpu_args, **gpu_kwargs))
    cpu = agent.run(agent.compile(cpu_workflow, *cpu_args, **cpu_kwargs))
    assert gpu.value == pytest.approx(cpu.value, abs=1e-8)
    assert gpu.provenance["operands"][0]["method"]["family"] == "UKS"


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("solvent", [qcl.PCM("water"), qcl.SMD("water")])
def test_agent_runs_signed_dft_solvation_on_gpu(solvent, ordinal):
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if len(cluster.gpus) <= ordinal:
        pytest.skip("Requested CUDA GPU unavailable")
    agent = _solvation_agent("smd" if isinstance(solvent, qcl.SMD) else "pcm",
                             None if isinstance(solvent, qcl.SMD) else solvent.eps,
                             method="dft", functional="pbe")
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    gpu = agent.run(agent.compile(dft_solvation_quantity,
                                  molecule, cluster.gpus[ordinal], solvent))
    cpu = agent.run(agent.compile(dft_solvation_quantity,
                                  molecule, cluster.cpus[0], solvent))
    assert gpu.value == pytest.approx(cpu.value, abs=1e-8)
    assert gpu.provenance["task_record_hash"] == agent.record.hash
    assert gpu.provenance["operands"][0]["device"] == f"gpu{ordinal}"


def test_agent_compiles_builtin_gpu_pcm_solvation_without_eri():
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent()
    gpu = qcl.Device("gpu0", "gpu")
    cluster = qcl.Cluster((qcl.Device("cpu0", "cpu"),), (gpu,), ())
    workflow, positional, keyword = _builtin_program(agent.record.spec, cluster,
                                                     device_id="gpu0")
    review = agent.compile(workflow, *positional, **keyword)
    operations = review.explanation["operations"]
    assert [item["op"] for item in operations] == ["scf.rhf", "scf.rhf"]
    assert [item["device"] for item in operations] == ["gpu0", "gpu0"]
    assert operations[1]["implementation"] == "scf.rhf/gpu4pyscf-pcm"


@pytest.mark.parametrize("device_kind", ["cpu", "gpu"])
def test_agent_compiles_builtin_signed_smd_solvation(device_kind):
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent("smd", None)
    cpu = qcl.Device("cpu0", "cpu")
    gpu = qcl.Device("gpu0", "gpu")
    cluster = qcl.Cluster((cpu,), (gpu,), ())
    device_id = gpu.id if device_kind == "gpu" else cpu.id
    workflow, positional, keyword = _builtin_program(
        agent.record.spec, cluster, device_id=device_id)
    review = agent.compile(workflow, *positional, **keyword)
    assert review.explanation["operations"][-1]["implementation"] == (
        f"scf.rhf/{'gpu4pyscf' if device_kind == 'gpu' else 'chemflow-pyscf'}-smd")


def test_agent_runs_signed_smd_solvation_on_gpu():
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("CUDA GPU unavailable")
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent("smd", None)
    gpu_workflow, gpu_args, gpu_kwargs = _builtin_program(
        agent.record.spec, cluster, device_id=cluster.gpus[-1].id)
    cpu_workflow, cpu_args, cpu_kwargs = _builtin_program(agent.record.spec, cluster)
    result = agent.run(agent.compile(gpu_workflow, *gpu_args, **gpu_kwargs))
    reference = agent.run(agent.compile(cpu_workflow, *cpu_args, **cpu_kwargs))
    assert result.value == pytest.approx(reference.value, abs=1e-8)
    assert result.provenance["operands"][0]["method"]["solvent_model"] == "SMD"


def test_agent_runs_builtin_gpu_pcm_solvation_against_cpu():
    pytest.importorskip("gpu4pyscf")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("CUDA GPU unavailable")
    from chemflow.agent.qcl import _builtin_program

    agent = _solvation_agent()
    gpu = cluster.gpus[-1]
    gpu_workflow, gpu_args, gpu_kwargs = _builtin_program(
        agent.record.spec, cluster, device_id=gpu.id)
    cpu_workflow, cpu_args, cpu_kwargs = _builtin_program(agent.record.spec, cluster)
    result = agent.run(agent.compile(gpu_workflow, *gpu_args, **gpu_kwargs))
    reference = agent.run(agent.compile(cpu_workflow, *cpu_args, **cpu_kwargs))
    assert result.value == pytest.approx(reference.value, abs=1e-8)
    assert result.provenance["operands"][0]["implementation"] == (
        "scf.rhf/gpu4pyscf-pcm")
    assert result.provenance["operands"][0]["device"] == gpu.id


def test_agent_rejects_gpu_pcm_with_wrong_signed_solvent():
    @qcl.workflow
    def wrong_solvent(molecule, device):
        gas = qcl.scf.rhf(molecule).to(device).energy
        liquid = qcl.scf.rhf(molecule, solvent=qcl.PCM("toluene")).to(device).energy
        return liquid - gas

    with pytest.raises(AdapterError, match="CF-E120"):
        _solvation_agent().compile(
            wrong_solvent, qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
            qcl.Device("gpu0", "gpu"))


def test_agent_rejects_wrong_pcm_model_eps_or_difference_sign():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        _solvation_agent().compile(solvation_quantity, molecule, cpu,
                                   qcl.PCM("water", eps=32.613))
    with pytest.raises(AdapterError, match="CF-E120"):
        _solvation_agent().compile(solvation_quantity, molecule, cpu,
                                   qcl.PCM("water", eps=78.355), reverse=True)
    with pytest.raises(AdapterError, match="CF-E120"):
        _solvation_agent("smd", None).compile(solvation_quantity, molecule, cpu,
                                               qcl.PCM("water", eps=78.355))


@qcl.workflow
def metabolic_quantity(molecule, cpu, level="hf"):
    return qcl.metabolism.sites(molecule, level=level).to(cpu)


def _metabolic_agent(level="hf"):
    store = TaskStore("qcl-metabolic-key")
    spec = TaskSpec.model_validate({
        "question": "Methane C-H site scores",
        "target": {"kind": "metabolic_sites",
                   "molecule": {"atoms": "C 0 0 0; H 0.6276 0.6276 0.6276; "
                                         "H -0.6276 -0.6276 0.6276; "
                                         "H -0.6276 0.6276 -0.6276; "
                                         "H 0.6276 -0.6276 -0.6276"},
                   "electronic_model": {"basis": "sto-3g"},
                   "level": {"method": level}},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_metabolic_sites():
    agent = _metabolic_agent()
    result = agent.run(agent.compile(
        metabolic_quantity,
        qcl.Molecule(agent.record.spec.target.molecule["atoms"], "sto-3g"),
        qcl.Cluster.local().cpus[0]))
    assert len(result.sites) == 4
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_wrong_metabolic_level():
    agent = _metabolic_agent("mp2")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(metabolic_quantity,
                      qcl.Molecule(agent.record.spec.target.molecule["atoms"], "sto-3g"),
                      qcl.Cluster.local().cpus[0], "hf")


@qcl.workflow
def binding_quantity(receptor, ligand, cpu, **options):
    return qcl.binding.mmgbsa(receptor, ligand, **options).to(cpu)


def _binding_agent():
    store = TaskStore("qcl-binding-key")
    spec = TaskSpec.model_validate({
        "question": "Model-defined MM/GBSA binding energy",
        "target": {"kind": "binding_free_energy",
                   "receptor": {"pdb_text": "PDB", "chains": ["A"]},
                   "ligand": {"source": {"smiles": "C"}},
                   "frames": 0, "minimize": False, "seed": 7,
                   "forcefield": "gaff-2.11", "ph": 7.0},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_accepts_signed_binding_protocol_and_rejects_input_change():
    agent = _binding_agent()
    target = agent.record.spec.target
    cpu = qcl.Cluster.local().cpus[0]
    review = agent.compile(binding_quantity, target.receptor, target.ligand, cpu,
                           frames=0, minimize=False, seed=7, forcefield="gaff-2.11", ph=7.0)
    assert review.explanation["operations"][0]["op"] == "binding.mmgbsa"
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(binding_quantity, target.receptor, {"source": {"smiles": "CC"}},
                      cpu, frames=0, minimize=False, seed=7, forcefield="gaff-2.11", ph=7.0)


@qcl.workflow
def barrier_quantity(reactant, guess, product, active, cpu, reverse=False,
                     thermo=None, rate_constant=False):
    minimum = qcl.optimize(reactant).to(cpu)
    transition = qcl.engines.sella.saddle(guess, reactant, product).to(cpu)

    def energy(molecule):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        electronic = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        if thermo is None:
            return electronic
        modes = qcl.frequencies(molecule).to(cpu)
        return electronic + qcl.thermo.rrho(
            molecule, modes, temperature_K=thermo["temperature_K"],
            pressure_Pa=thermo["pressure_Pa"], quantity=thermo["quantity"]).to(cpu)

    barrier = (energy(minimum) - energy(transition) if reverse else
               energy(transition) - energy(minimum))
    if rate_constant:
        return barrier, qcl.thermo.eyring(barrier, thermo["temperature_K"]).to(cpu)
    return barrier


def _barrier_agent(thermo=None, rate_constant=False):
    store = TaskStore("qcl-barrier-key")
    reactant = {"atoms": "C 0 0 0; N 0 0 1.16; H 0 0 -1.07"}
    transition = {"atoms": "C 0 0 0; N 0 0 1.2; H 1.1 0 .5"}
    product = {"atoms": "C 0 0 0; N 0 0 1.17; H 0 0 2.17"}
    spec = TaskSpec.model_validate({
        "question": "HCN to HNC barrier",
        "target": {"kind": "reaction_barrier", "product": product,
                   "electronic_model": {"basis": "sto-3g"}, "thermo": thermo,
                   "rate_constant": rate_constant,
                   "species": [
                       {"name": "r", "molecule": reactant, "coefficient": -1,
                        "role": "minimum", "active": {"n_elec": 2, "n_orb": 2},
                        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}},
                       {"name": "ts", "molecule": transition, "coefficient": 1,
                        "role": "ts", "active": {"n_elec": 2, "n_orb": 2},
                        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}}]},
        "gates": [], "budget": {"wall_s": 1800}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_accepts_signed_electronic_barrier_and_rejects_reversal():
    agent = _barrier_agent()
    target = agent.record.spec.target
    args = (qcl.Molecule(target.species[0].molecule["atoms"], "sto-3g"),
            qcl.Molecule(target.species[1].molecule["atoms"], "sto-3g"),
            qcl.Molecule(target.product["atoms"], "sto-3g"),
            qcl.Active(2, 2), qcl.Cluster.local().cpus[0])
    review = agent.compile(barrier_quantity, *args)
    assert review.explanation["operations"][1]["op"] == "sella.saddle"
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(barrier_quantity, *args, reverse=True)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(barrier_quantity, *args[:3], qcl.Active(2, 3), args[-1])


def test_agent_accepts_signed_gibbs_barrier_and_rate():
    agent = _barrier_agent({"quantity": "gibbs", "temperature_K": 298.15,
                            "pressure_Pa": 101325.0}, rate_constant=True)
    target = agent.record.spec.target
    args = (qcl.Molecule(target.species[0].molecule["atoms"], "sto-3g"),
            qcl.Molecule(target.species[1].molecule["atoms"], "sto-3g"),
            qcl.Molecule(target.product["atoms"], "sto-3g"),
            qcl.Active(2, 2), qcl.Cluster.local().cpus[0])
    conditions = target.thermo.model_dump()
    review = agent.compile(barrier_quantity, *args, thermo=conditions, rate_constant=True)
    assert [operation["op"] for operation in review.explanation["operations"]][-1] == "thermo.eyring"
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(barrier_quantity, *args,
                      thermo={**conditions, "temperature_K": 310.0}, rate_constant=True)


@qcl.workflow
def lattice_quantity(lattice, sector, cpu, quantity="energy"):
    operator = qcl.models.hubbard(lattice, t=1, U=4)
    ground = qcl.ed.ground_state(operator, sector=sector).to(cpu)
    if quantity == "energy":
        return ground.energy
    return qcl.expect(ground, qcl.ops.onsite_pair(0, 1)).to(cpu)


def _lattice_agent(kind="lattice_ground_energy"):
    store = TaskStore("qcl-lattice-key")
    target = {"kind": kind,
              "lattice": {"model": "hubbard", "geometry": "chain", "n_sites": 2,
                          "t": 1.0, "U": 4.0},
              "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}}
    if kind == "pairing_correlation":
        target["site_pair"] = [0, 1]
    spec = TaskSpec.model_validate({"question": kind, "target": target, "gates": [],
                                    "budget": {"wall_s": 30}, "decision": "test"})
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_lattice_energy_and_pairing():
    args = (qcl.lattice.chain(2), qcl.Sector(n=2, sz=0),
            qcl.Device("cpu0", "cpu"))
    energy_agent = _lattice_agent()
    energy = energy_agent.run(energy_agent.compile(lattice_quantity, *args))
    assert energy.value == pytest.approx(2 - 8 ** 0.5, abs=1e-10)
    pair_agent = _lattice_agent("pairing_correlation")
    pair = pair_agent.run(pair_agent.compile(lattice_quantity, *args, quantity="pair"))
    assert pair.value == pytest.approx((1 - 1 / 2 ** 0.5) / 4, abs=1e-10)
    assert pair.provenance["task_record_hash"] == pair_agent.record.hash


def test_agent_rejects_lattice_question_mismatch():
    args = (qcl.lattice.chain(2), qcl.Sector(n=2, sz=0),
            qcl.Device("cpu0", "cpu"))
    with pytest.raises(AdapterError, match="CF-E120"):
        _lattice_agent().compile(lattice_quantity, *args, quantity="pair")
    with pytest.raises(AdapterError, match="CF-E120"):
        _lattice_agent("pairing_correlation").compile(lattice_quantity, *args)
    with pytest.raises(AdapterError, match="CF-E120"):
        _lattice_agent().compile(lattice_quantity, qcl.lattice.chain(3),
                                 *args[1:])


@qcl.workflow
def solubility_quantity(smiles, cpu, model, melting_point=None):
    if model == "esol":
        return qcl.solubility.esol(smiles).to(cpu)
    return qcl.solubility.gse(smiles, melting_point).to(cpu)


def _solubility_agent(model="esol"):
    store = TaskStore("qcl-solubility-key")
    target = {"kind": "aqueous_solubility", "molecule_smiles": "CCO", "model": model}
    if model == "gse":
        target["melting_point_C"] = 30.0
    spec = TaskSpec.model_validate({"question": "Ethanol solubility", "target": target,
                                    "gates": [], "budget": {"wall_s": 30},
                                    "decision": "test"})
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_aqueous_solubility():
    agent = _solubility_agent()
    cpu = qcl.Device("cpu0", "cpu")
    result = agent.run(agent.compile(solubility_quantity, "CCO", cpu, "esol"))
    assert result.unit == "log10(mol/L)"
    assert result.provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_wrong_signed_solubility_model_or_molecule():
    agent = _solubility_agent("gse")
    cpu = qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(solubility_quantity, "CCO", cpu, "esol")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(solubility_quantity, "CCN", cpu, "gse", 30.0)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(solubility_quantity, "CCO", cpu, "gse", 31.0)


@qcl.workflow
def charge_difference(molecule, active, cpu, kind):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    base = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
    if kind == "ionization_energy":
        ion = qcl.fci.solve(space.hamiltonian,
                            space.sector.add_electrons(-1)).to(cpu).energy
        return ion - base
    added = qcl.fci.solve(space.hamiltonian,
                          space.sector.add_electrons(1)).to(cpu).energy
    return base - added


def _charge_agent(kind="ionization_energy", final_spin=0.5):
    store = TaskStore("qcl-charge-key")
    spec = TaskSpec.model_validate({
        "question": kind,
        "target": {"kind": kind,
                   "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                   "electronic_model": {"basis": "sto-3g"},
                   "final_spin": final_spin},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_ip_and_ea():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    active, cpu = qcl.Active(2, 2), qcl.Device("cpu0", "cpu")
    ip_agent = _charge_agent()
    ip = ip_agent.run(ip_agent.compile(charge_difference, molecule, active, cpu,
                                       "ionization_energy"))
    assert ip.value == pytest.approx(0.5990783869236054, abs=1e-9)
    ea_agent = _charge_agent("electron_affinity")
    ea = ea_agent.run(ea_agent.compile(charge_difference, molecule, active, cpu,
                                       "electron_affinity"))
    assert ea.value == pytest.approx(-0.6916680190063282, abs=1e-9)


def test_agent_rejects_wrong_charge_difference_or_final_spin():
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    active, cpu = qcl.Active(2, 2), qcl.Device("cpu0", "cpu")
    with pytest.raises(AdapterError, match="CF-E120"):
        _charge_agent().compile(charge_difference, molecule, active, cpu,
                                "electron_affinity")
    with pytest.raises(AdapterError, match="CF-E120"):
        _charge_agent(final_spin=1.5).compile(charge_difference, molecule, active,
                                               cpu, "ionization_energy")


@qcl.workflow
def ranking_quantity(molecules, cpu):
    values = qcl.map(lambda molecule, lane: qcl.scf.rhf(molecule).to(lane).energy,
                     molecules).to(cpu)
    return qcl.rank(values)


def _ranking_agent():
    store = TaskStore("qcl-ranking-key")
    spec = TaskSpec.model_validate({
        "question": "Rank H2 geometries",
        "target": {"kind": "ranking", "electronic_model": {"basis": "sto-3g"},
                   "items": [{"name": "stretched", "molecule":
                              {"atoms": "H 0 0 0; H 0 0 1.5"}},
                             {"name": "equilibrium", "molecule":
                              {"atoms": "H 0 0 0; H 0 0 0.74"}}]},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_ranking():
    agent = _ranking_agent()
    molecules = [qcl.Molecule("H 0 0 0; H 0 0 1.5", "sto-3g"),
                 qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")]
    result = agent.run(agent.compile(ranking_quantity, molecules,
                                     qcl.Device("cpu0", "cpu")))
    assert result.order == (1, 0)
    assert result.provenance["task_record_hash"] == agent.record.hash
    assert all(value.provenance["task_record_hash"] == agent.record.hash and
               "timestamp" in value.provenance for value in result.values)


def test_agent_rejects_swapped_signed_ranking_items():
    agent = _ranking_agent()
    molecules = [qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                 qcl.Molecule("H 0 0 0; H 0 0 1.5", "sto-3g")]
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(ranking_quantity, molecules, qcl.Device("cpu0", "cpu"))


@qcl.workflow
def bde_collection(molecule, bonds, cpu, reverse=False):
    def energy(species, device):
        mean_field = qcl.scf.hf(species).to(device)
        space = qcl.active_space(mean_field, qcl.Active(max_orbitals=2)).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    parent = energy(molecule, cpu)

    def point(bond, lane):
        first, second = molecule.cleave(bond)
        fragments = energy(first, lane) + energy(second, lane)
        return parent - fragments if reverse else fragments - parent

    return qcl.map(point, bonds).to([cpu])


def _bde_agent():
    store = TaskStore("qcl-bde-key")
    spec = TaskSpec.model_validate({
        "question": "H2 bond dissociation",
        "target": {"kind": "bond_dissociation",
                   "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                   "electronic_model": {"basis": "sto-3g"},
                   "bonds": [[0, 1]]},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_bond_dissociation_collection():
    agent = _bde_agent()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    result = agent.run(agent.compile(bde_collection, molecule, [(0, 1)],
                                     qcl.Device("cpu0", "cpu")))
    assert len(result) == 1 and result[0].value > 0
    assert result[0].provenance["task_record_hash"] == agent.record.hash


def test_agent_rejects_reversed_bde_sign():
    agent = _bde_agent()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(bde_collection, molecule, [(0, 1)],
                      qcl.Device("cpu0", "cpu"), reverse=True)


@qcl.workflow
def interaction_collection(molecule, separations, cpu):
    def energy(species, device):
        mean_field = qcl.scf.hf(species).to(device)
        space = qcl.active_space(mean_field, qcl.Active(max_orbitals=4)).to(device)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(device).energy

    def point(distance, lane):
        geometry = molecule.set_separation([0, 1], [2, 3], distance)
        return (energy(geometry, lane) - energy(geometry.ghost([2, 3]), lane) -
                energy(geometry.ghost([0, 1]), lane))

    return qcl.map(point, separations).to([cpu])


def _interaction_agent():
    store = TaskStore("qcl-interaction-key")
    spec = TaskSpec.model_validate({
        "question": "H2 dimer interaction scan",
        "target": {"kind": "interaction_scan",
                   "complex": {"atoms":
                               "H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74"},
                   "electronic_model": {"basis": "sto-3g"},
                   "fragments": [[0, 1], [2, 3]],
                   "separations": [2.0, 3.0], "counterpoise": True},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_interaction_scan():
    agent = _interaction_agent()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", "sto-3g")
    result = agent.run(agent.compile(interaction_collection, molecule, [2.0, 3.0],
                                     qcl.Device("cpu0", "cpu")))
    assert len(result) == 2
    assert all(item.provenance["task_record_hash"] == agent.record.hash
               for item in result)


def test_agent_rejects_wrong_interaction_scan_grid():
    agent = _interaction_agent()
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", "sto-3g")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(interaction_collection, molecule, [3.0, 2.0],
                      qcl.Device("cpu0", "cpu"))


@qcl.workflow
def pes_collection(molecule, angles, cpu):
    return qcl.map(lambda angle, lane: qcl.scf.hf(
        molecule.set_dihedral(0, 1, 2, 3, angle)).to(lane).energy,
        angles).to([cpu])


def _pes_agent():
    store = TaskStore("qcl-pes-key")
    spec = TaskSpec.model_validate({
        "question": "Rigid peroxide torsion scan",
        "target": {"kind": "pes_scan",
                   "molecule": {"atoms":
                                "H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1"},
                   "electronic_model": {"basis": "sto-3g"},
                   "coordinate": {"type": "dihedral", "atoms": [0, 1, 2, 3],
                                  "values": [60.0, 120.0]}, "mode": "rigid"},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    return QclAgent(store.freeze(spec, "reviewer"), store)


def test_agent_runs_signed_rigid_pes_scan():
    agent = _pes_agent()
    molecule = qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g")
    result = agent.run(agent.compile(pes_collection, molecule, [60.0, 120.0],
                                     qcl.Device("cpu0", "cpu")))
    assert len(result) == 2
    assert all(item.provenance["task_record_hash"] == agent.record.hash
               for item in result)


def test_agent_rejects_wrong_rigid_pes_grid():
    agent = _pes_agent()
    molecule = qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g")
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(pes_collection, molecule, [120.0, 60.0],
                      qcl.Device("cpu0", "cpu"))


@pytest.mark.parametrize("factory,operation", [
    (_frontier_agent, "scf.rhf"),
    (_lattice_agent, "ed.ground_state"),
    (lambda: _lattice_agent("pairing_correlation"), "expect"),
    (_solubility_agent, "solubility.esol"),
    (lambda: _solubility_agent("gse"), "solubility.gse"),
])
def test_builtin_qcl_agent_compiles_and_runs_signed_targets(factory, operation):
    from chemflow.agent.qcl import _builtin_program

    agent = factory()
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    assert review.explanation["operations"][-1]["op"] == operation
    result = agent.run(review)
    assert result.provenance["task_record_hash"] == agent.record.hash


@pytest.mark.parametrize("kind,expected", [
    ("ionization_energy", 0.5990783869236054),
    ("electron_affinity", -0.6916680190063282),
])
def test_builtin_qcl_agent_charge_differences(kind, expected):
    from chemflow.agent.qcl import _builtin_program

    agent = _charge_agent(kind)
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    result = agent.run(agent.compile(workflow, *args, **kwargs))
    assert result.value == pytest.approx(expected, abs=1e-9)


def test_builtin_qcl_agent_reaction_energy_uses_signed_stoichiometry():
    from chemflow.agent.qcl import _builtin_program

    store = TaskStore("qcl-builtin-reaction")
    spec = TaskSpec.model_validate({
        "question": "H2 atomization",
        "target": {"kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
                   "species": [
                       {"name": "H2", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                        "coefficient": -1, "active": {"n_elec": 2, "n_orb": 2},
                        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}},
                       {"name": "H", "molecule": {"atoms": "H 0 0 0", "spin": 1},
                        "coefficient": 2, "active": {"n_elec": 1, "n_orb": 1},
                        "sector": {"n_alpha": 1, "n_beta": 0, "n_orb": 1}},
                   ]},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 1e-8,
                   "unit": "hartree"}], "budget": {"wall_s": 60},
        "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(spec, qcl.Cluster.local())
    result = agent.run(agent.compile(workflow, *args, **kwargs))
    assert result.value == pytest.approx(0.20412013537395, abs=1e-8)
    assert result.provenance["signed_gates"][0]["passed"] is True


@pytest.mark.parametrize("factory,expected_count", [
    (_ranking_agent, None),
    (_bde_agent, 1),
    (_interaction_agent, 2),
    (_pes_agent, 2),
])
def test_builtin_qcl_agent_signed_collections(factory, expected_count):
    from chemflow.agent.qcl import _builtin_program

    agent = factory()
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    result = agent.run(agent.compile(workflow, *args, **kwargs))
    if expected_count is None:
        assert result.order == (1, 0)
        values = result.values
    else:
        assert len(result) == expected_count
        values = result
    assert all(item.provenance["task_record_hash"] == agent.record.hash
               for item in values)


@pytest.mark.parametrize("factory,expected_operation", [
    (_metabolic_agent, "metabolism.sites"),
    (_solvation_agent, "scf.rhf"),
])
def test_builtin_qcl_agent_metabolism_and_solvation(factory, expected_operation):
    from chemflow.agent.qcl import _builtin_program

    agent = factory()
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    assert review.explanation["operations"][-1]["op"] == expected_operation
    result = agent.run(review)
    assert result.provenance["task_record_hash"] == agent.record.hash


@pytest.mark.parametrize("solvent_model", ["alpb", "gbsa"])
def test_builtin_qcl_agent_gfn2_solvation(solvent_model):
    from chemflow.agent.qcl import _builtin_program

    store = TaskStore("qcl-gfn2-solvation")
    spec = TaskSpec.model_validate({
        "question": "H2 water solvation",
        "target": {"kind": "solvation_free_energy",
                   "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                   "electronic_model": {"basis": "sto-3g"},
                   "level": {"method": "gfn2"},
                   "solvent": {"model": solvent_model, "name": "water"}},
        "gates": [], "budget": {"wall_s": 30}, "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    result = agent.run(review)
    assert result.provenance["task_record_hash"] == agent.record.hash
    assert result.provenance["operands"][0]["method"]["solvent"] == {
        "model": solvent_model, "name": "water"}

    @qcl.workflow
    def reversed_solvation(molecule, cpu):
        gas = qcl.engines.tblite.gfn2.energy(molecule).to(cpu)
        liquid = qcl.engines.tblite.gfn2.energy(
            molecule, solvent=qcl.ALPB("water") if solvent_model == "alpb"
            else qcl.GBSA("water")).to(cpu)
        return gas - liquid

    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(reversed_solvation, *args)


def test_builtin_qcl_agent_binding_plan_preserves_signed_protocol():
    from chemflow.agent.qcl import _builtin_program

    agent = _binding_agent()
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    assert review.explanation["operations"][0]["op"] == "binding.mmgbsa"


@pytest.mark.parametrize("thermo,rate", [
    (None, False),
    ({"quantity": "gibbs", "temperature_K": 298.15,
      "pressure_Pa": 101325.0}, True),
])
def test_builtin_qcl_agent_barrier_preserves_signed_path(thermo, rate):
    from chemflow.agent.qcl import _builtin_program

    agent = _barrier_agent(thermo, rate_constant=rate)
    workflow, args, kwargs = _builtin_program(agent.record.spec, qcl.Cluster.local())
    review = agent.compile(workflow, *args, **kwargs)
    operations = [item["op"] for item in review.explanation["operations"]]
    assert operations[:2] == ["geometry.optimize", "sella.saddle"]
    assert ("thermo.eyring" in operations) is rate


def test_builtin_qcl_agent_counterpoise_interaction():
    from chemflow.agent.qcl import _builtin_program

    store = TaskStore("qcl-builtin-interaction")
    spec = TaskSpec.model_validate({
        "question": "H2 dimer interaction",
        "target": {"kind": "interaction_energy",
                   "complex": {"atoms": "H 0 0 0; H 0 0 0.74; "
                                         "H 0 0 3.0; H 0 0 3.74"},
                   "fragments": [[0, 1], [2, 3]],
                   "electronic_model": {"basis": "sto-3g"},
                   "counterpoise": True},
        "budget": {"wall_s": 90}, "gates": [
            {"kind": "abs_error_vs_exact", "threshold": 0.01,
             "unit": "hartree"}], "decision": "test",
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(spec, qcl.Cluster.local())
    result = agent.run(agent.compile(workflow, *args, **kwargs))
    assert result.provenance["signed_gates"][0]["passed"] is True


def test_builtin_qcl_agent_gpu_plan_has_explicit_scf_and_fci_copies():
    from chemflow.agent.qcl import _builtin_program

    store = TaskStore("qcl-builtin-gpu")
    spec = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-7,
                           unit="hartree"),)})
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    cluster = qcl.Cluster(cpus=(qcl.Device("cpu0", "cpu"),),
                          gpus=(qcl.Device("gpu0", "gpu"),), qpus=())
    workflow, args, kwargs = _builtin_program(spec, cluster, device_id="gpu0")
    review = agent.compile(workflow, *args, **kwargs)
    operations = review.explanation["operations"]
    assert [(item["op"], item["device"]) for item in operations] == [
        ("scf.rhf", "gpu0"), ("active_space", "cpu0"), ("fci.solve", "gpu0")]
    copies = [node for node in review.plan.graph_json()["nodes"]
              if node["op"] == "copy"]
    assert len(copies) >= 3


def test_builtin_qcl_agent_runs_gpu_ground_and_lattice_when_available():
    pytest.importorskip("cupy")
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA device")
    from chemflow.agent.qcl import _builtin_program

    store = TaskStore("qcl-builtin-gpu-fci")
    gated = _agent().record.spec.model_copy(update={
        "gates": (TaskGate(kind="abs_error_vs_exact", threshold=1e-7,
                           unit="hartree"),)})
    fci_agent = QclAgent(store.freeze(gated, "reviewer"), store)
    for agent in (_agent(), fci_agent, _lattice_agent()):
        workflow, args, kwargs = _builtin_program(
            agent.record.spec, cluster, device_id=cluster.gpus[0].id)
        review = agent.compile(workflow, *args, **kwargs)
        result = agent.run(review)
        assert result.provenance["task_record_hash"] == agent.record.hash
        assert any(item["device"] == cluster.gpus[0].id
                   for item in review.explanation["operations"])
        if agent is fci_agent:
            assert result.value == pytest.approx(-1.1372838344885, abs=1e-8)
            assert result.provenance["signed_gates"][0]["passed"]


def test_qcl_agent_cli_records_signed_gpu_time_when_available(tmp_path):
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        pytest.skip("No CUDA GPU is available")
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 60, "gpu_s": 30}})
    private, public = generate_keypair(b"gpu-time-test")
    signed = TaskSigner(private, b"gpu-time-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "public.pem", "output"))
    task.write_text(signed.model_dump_json())
    key.write_bytes(public)
    run_cli(Namespace(task=task, public_key=key, program=None, cluster=None,
                      device=cluster.gpus[0].id, out=out, dry_run=False))
    summary = json.loads((out / "summary.json").read_text())
    assert 0 < summary["spent"]["gpu_s"] <= summary["spent"]["wall_s"]
    assert summary["spent"]["gpu_s"] < spec.budget["gpu_s"]


def test_qcl_agent_accounts_for_gpu_backed_qpu_simulator(tmp_path, monkeypatch):
    cluster = qcl.Cluster.local()
    simulator = next((device for device in cluster.qpus
                      if (device.simulator_host or "").startswith("gpu")), None)
    if simulator is None:
        pytest.skip("No GPU-backed QPU simulator is available")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "budget": {"wall_s": 60, "qpu_shots": 128, "gpu_s": 30},
    })
    private, public = generate_keypair(b"gpu-simulator-test")
    signed = TaskSigner(private, b"gpu-simulator-test").freeze(spec, "reviewer")
    task, key, out = (tmp_path / name for name in ("task.json", "public.pem", "output"))
    task.write_text(signed.model_dump_json())
    key.write_bytes(public)
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    run_cli(Namespace(task=task, public_key=key, program=None, cluster=None,
                      device=simulator.id, out=out, dry_run=False))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["spent"]["qpu_shots"] == 128
    assert 0 < summary["spent"]["gpu_s"] <= summary["spent"]["wall_s"]


def test_builtin_qcl_agent_qpu_sampling_respects_signed_task(tmp_path, monkeypatch):
    from chemflow.agent.qcl import _builtin_program

    cluster = qcl.Cluster.local()
    store = TaskStore("qcl-builtin-qpu")
    spec = _agent().record.spec.model_copy(update={
        "allowed_effects": ("pure", "simulator"),
        "budget": {"wall_s": 60, "qpu_shots": 128},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(
        spec, cluster, device_id=cluster.qpus[0].id)
    policy = tmp_path / "qcl-policy.toml"
    policy.write_text("[budget]\nqpu_shots = 128\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    review = agent.compile(workflow, *args, **kwargs)
    assert review.estimate["qpu_cost"] == 0
    assert [item["op"] for item in review.explanation["operations"]] == [
        "scf.hf", "active_space", "sample", "sqd.diagonalize"]
    result = agent.run(review)
    assert result.provenance["task_record_hash"] == agent.record.hash
    assert result.provenance["sample_source"] == "quantum_simulator"

    private, public = generate_keypair(b"test-password")
    record = TaskSigner(private, b"test-password").freeze(spec, "reviewer")
    task_path, key_path, out = (tmp_path / name for name in
                                ("task.json", "public.pem", "agent-output"))
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public)
    from chemflow.cli import main

    monkeypatch.setattr("sys.argv", ["chemflow", "agent", "run",
                                   "--task", str(task_path),
                                   "--public-key", str(key_path),
                                   "--engine", "qcl", "--device", cluster.qpus[0].id,
                                   "--out", str(out)])
    main()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["decision"] == "executed"
    assert summary["estimate"]["qpu_cost"] == 0
    assert (out / "qcl-result.json").exists()
    assert summary["spent"]["qpu_shots"] == 128
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert json.loads((out / f"{record.content_hash}.ledger.json").read_text())[
        "qpu_shots"] == 128


def test_builtin_qcl_agent_qpu_rejects_unsigned_resources_and_effects():
    from chemflow.agent.qcl import _builtin_program

    cluster = qcl.Cluster.local()
    device_id = cluster.qpus[0].id
    with pytest.raises(ValueError, match="CF-E070"):
        _builtin_program(_agent().record.spec, cluster, device_id=device_id)

    store = TaskStore("qcl-builtin-qpu-effects")
    spec = _agent().record.spec.model_copy(update={
        "budget": {"wall_s": 30, "qpu_shots": 128},
    })
    agent = QclAgent(store.freeze(spec, "reviewer"), store)
    workflow, args, kwargs = _builtin_program(spec, cluster, device_id=device_id)
    with pytest.raises(AdapterError, match="CF-E120"):
        agent.compile(workflow, *args, **kwargs)
