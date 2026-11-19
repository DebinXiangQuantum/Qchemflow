"""Reviewer regression tests for M3a (written by Claude; implementers must not edit)."""
import json

import pytest

from chemflow import build
from chemflow.agent.memory import ResultsDB
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task

N2 = "N 0 0 0; N 0 0 1.10"


def _spec(active, sector) -> TaskSpec:
    return TaskSpec.model_validate({
        "question": "Which method meets the N2 singlet gate?",
        "target": {"operator": "ground_energy", "molecule": {"atoms": N2},
                   "electronic_model": {"basis": "6-31g"}, "active": active,
                   "sector": sector, "target_spin": 0},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "budget": {"wall_s": 300, "cpu_core_s": 300, "sim_shots": 4096},
        "allowed_effects": ["pure", "simulator"],
        "decision": "choose a method"})


def _run(spec: TaskSpec, tmp_path):
    store = TaskStore("reviewer-secret")
    agent = Orchestrator(store, lambda draft: store.freeze(draft, "reviewer"), tmp_path,
                         max_rounds=1)
    run = agent.run(spec)
    entries = [json.loads(line) for line in run.journal.read_text().splitlines()]
    return run, entries, ResultsDB(tmp_path / "results.sqlite")


def test_grading_reference_never_competes_as_a_control(tmp_path) -> None:
    run, entries, memory = _run(
        _spec({"n_elec": 8, "n_orb": 10}, {"n_alpha": 4, "n_beta": 4, "n_orb": 10}), tmp_path)
    for entry in entries:
        if entry["state"] == "CRITIQUE":
            assert all(item["relation"] != "equivalent"
                       for item in entry["observation"]["controls"])
    # CCSD(T) meets the 1.6 mHa gate on N2 CAS(8,10) (0.93 mHa) far cheaper than SQD-2048,
    # so the meaningful decision is "use classical" naming an approximate control.
    assert run.decision.decision == "stop"
    assert "use classical" in run.decision.reason
    chosen = memory.query(node_id=run.decision.candidate)
    assert chosen and all(row["relation"] != "equivalent" for row in chosen)


def test_closed_shell_coupled_cluster_accepts_a_singlet_target() -> None:
    with build.program("cc_singlet") as b:
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"))
        energy = b.ground_energy(ham, b.sector(n_alpha=1, n_beta=1, n_orb=2, s=0), s=0)
        b.refine(energy, "ccsd_t")
        b.bind(energy, "ccsd_t")
    program = b.to_ir()
    result = Executor(molecular_registry()).run(program, task=freeze_task(program))[energy.id]
    assert result.context.keys.get("target_spin") in ("0", "0.0")
    assert result.relation == "approximate"
    with build.program("cc_triplet") as b:
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"))
        energy = b.ground_energy(ham, b.sector(n_alpha=1, n_beta=1, n_orb=2, s=1), s=1)
        b.refine(energy, "ccsd_t")
        b.bind(energy, "ccsd_t")
    program = b.to_ir()
    with pytest.raises(CheckError) as info:
        Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert "CF-E020" in {d.code for d in info.value.diagnostics}


def test_auto_active_space_is_chosen_by_the_modeler(tmp_path) -> None:
    _, entries, _ = _run(_spec("auto", {}), tmp_path)
    model = next(entry for entry in entries if entry["state"] == "MODEL")
    controls = next(entry for entry in entries if entry["state"] == "CONTROLS")
    chosen = model["decision"]["active"]
    assert controls["observation"]["active"] == chosen
    assert controls["observation"].get("fallback") in (False, None)
    assert chosen["n_orb"] <= 10 and 2 <= chosen["n_elec"] <= 2 * chosen["n_orb"]
    assert (14 - chosen["n_elec"]) % 2 == 0


def test_orchestrator_executes_only_through_registered_tools(tmp_path) -> None:
    spec = TaskSpec.model_validate({
        "question": "H2 smoke", "target": {
            "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "budget": {"wall_s": 60}, "allowed_effects": ["pure", "simulator"],
        "decision": "smoke"})
    store = TaskStore("reviewer-secret")
    record = store.freeze(spec, "reviewer")
    registered = {item["name"] for role in ("planner", "modeler", "designer", "critic",
                                            "analyst", "orchestrator")
                  for item in make_tools(molecular_registry(), Executor(molecular_registry()),
                                         record, store).schemas(role)}
    agent = Orchestrator(store, lambda draft: store.freeze(draft, "reviewer"), tmp_path,
                         max_rounds=1)
    run = agent.run(spec, diagnostics={"mp2_noons": [2.0, 0.0],
                                       "suggested_active": {"n_elec": 2, "n_orb": 2}})
    calls = [call for line in run.journal.read_text().splitlines()
             for call in json.loads(line)["tool_calls"]]
    assert calls
    assert {call["name"] for call in calls} <= registered
