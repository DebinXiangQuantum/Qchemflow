"""Reviewer tests for R130 (written by Claude; implementers must not edit).

User requirement (2026-10-14): "there contains some state vector simulator. Please ensure that
the simulator is only used for verification, and cannot be used for the algorithm deployed in
hybrid quantum-classical nodes."

What the reviewer found:
- `vqe_statevector` and `adapt_statevector` compute exact energies from a simulated state, and
  were offered as candidate templates. glm-5.3 proposed 13–17 VQE candidates per run.
- The exact `expectation` op and the `ideal_energy_projective_sampler` produce results no
  hardware can.
- Only `sample_configurations` has a QPU implementation (lqcloud MQ02), and SQD uses it.

Contract:
- `ImplementationSpec` gains `execution_role`, one of "classical", "qpu", "emulator" or
  "verification_only", and `emulates` (an op name or None).
  - cpu_numeric and gpu_numeric default to "classical"; qpu means "qpu".
  - Every quantum_simulator or fault_tolerant_projection implementation must declare
    "emulator" or "verification_only".
  - An emulator stands in for an op that has a QPU implementation in
    `chemflow.backends.QPU_CATALOG` (op → QPU implementation ids). Its `emulates` names that
    op, and its observable semantics match the QPU's (samples of a prepared circuit).
- `check(program, registry, task=None, purpose=...)`, with purpose "screen" (the default),
  "deploy" or "production":
  - "deploy": CF-E140 for every verification-only binding in the dependency closure of a
    target that is not one of the plan's references. References, and the verification
    graphs behind certificates, falsification, validity maps and `algorithm.verify`, may use
    any simulator.
  - "production": additionally CF-E140 for every emulator binding, since production quantum
    steps must be bound to QPU implementations.
- `plan.candidates` refuses templates with no deployable implementation (vqe today). The
  error says the statevector implementation is verification-only. `design.submit` and the
  orchestrator's candidate path check with purpose "deploy". `run` at scale "production"
  checks with purpose "production".
- The scripted designer proposes only deployable candidates.
- Candidate items gain `deployable` and `deployment` (node → QPU implementation id from the
  catalog, for each emulated quantum step). The critic never escalates a candidate whose
  `deployable` is false.
"""
from __future__ import annotations

import json
from argparse import Namespace

import pytest
import test_review_m4a as m4a

from chemflow.registry import ImplementationSpec  # noqa: F401  (field contract below)

VERIFICATION_ONLY = ("vqe_statevector", "adapt_statevector", "expectation",
                     "ideal_energy_projective_sampler")


def _all_packs():
    from chemflow.domains import load_registry, pack_names

    return load_registry(pack_names())


def test_r130_every_simulator_declares_its_role() -> None:
    from chemflow.backends import QPU_CATALOG
    from chemflow.types.results import BackendKind

    registry = _all_packs()
    assert QPU_CATALOG.get("sample_configurations")
    for identifier, implementation in registry.implementations.items():
        role = implementation.execution_role
        kind = implementation.backend_kind
        if kind in (BackendKind.cpu_numeric, BackendKind.gpu_numeric):
            assert role == "classical", identifier
        elif kind == BackendKind.qpu:
            assert role == "qpu", identifier
        else:
            assert role in ("emulator", "verification_only"), identifier
        if role == "emulator":
            assert implementation.emulates in QPU_CATALOG, identifier
            assert implementation.emulates == implementation.target, identifier
    for identifier in VERIFICATION_ONLY:
        assert registry.implementations[identifier].execution_role == "verification_only"
    sampler = registry.implementations["sample_configurations"]
    assert sampler.execution_role == "emulator"


def _vqe_plan(as_reference: bool):
    from chemflow import build
    from chemflow.ops.molecular import molecular_registry

    with build.program("vqe") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                          builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        vqe = builder.ground_energy(hamiltonian, sector, s=0)
        builder.refine(vqe, "vqe")
        builder.bind(vqe, "vqe_statevector")
        sci = builder.ground_energy(hamiltonian, sector, s=0)
        builder.refine(sci, "sci")
        builder.bind(sci, "sci_pyscf")
    program = builder.to_ir()
    program.targets = [vqe.id, sci.id]
    if as_reference:
        program.references = [vqe.id]
    return program, molecular_registry(), vqe.id


def test_r130_designs_may_not_contain_verification_only_steps() -> None:
    from chemflow.check.static import check

    program, registry, vqe = _vqe_plan(as_reference=False)
    issues = check(program, registry, purpose="deploy")
    assert any(issue.code == "CF-E140" and issue.node == vqe for issue in issues)
    assert not any(issue.code == "CF-E140" for issue in check(program, registry))  # screening
    reference, registry, _ = _vqe_plan(as_reference=True)
    assert not any(issue.code == "CF-E140"
                   for issue in check(reference, registry, purpose="deploy"))


def test_r130_the_designer_tools_refuse_verification_only_candidates() -> None:
    tools = m4a._tools()
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base_id = tools.call("modeler", "plan.base", {"plan_id": empty,
                                                  "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]
    with pytest.raises(Exception) as caught:
        tools.call("designer", "plan.candidates", {"plan_id": base_id,
                                                   "candidates": [{"template": "vqe"}]})
    assert "verification" in str(caught.value).lower()
    base = tools.plans.get(base_id)
    hamiltonian = next(node.id for node in base.nodes if node.op == "hamiltonian")
    sector = next(node.id for node in base.nodes if node.op == "sector")
    node = {"id": "vqe0", "op": "chemflow.ground_energy",
            "inputs": [{"port": "hamiltonian", "source": hamiltonian},
                       {"port": "sector", "source": sector}],
            "params": [{"key": "s", "value_json": "0"}], "effects": "pure",
            "refinement": "vqe", "binding": "vqe_statevector"}
    built = tools.call("designer", "plan.add_node", {"plan_id": base_id, "node": node,
                                                     "target": True})["data"]
    tools.design_base_plan_id = base_id
    with pytest.raises(Exception) as caught:
        tools.call("designer", "design.submit", {"plan_id": built["plan_id"]})
    assert "CF-E140" in str(caught.value)


def _sampling_tools():
    from chemflow.agent.task import TaskSpec, TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    spec = TaskSpec.model_validate({
        "question": "q", "target": {
            "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "budget": {"sim_shots": 100000}, "allowed_effects": ["pure", "simulator"],
        "decision": "d"})
    store = TaskStore("reviewer")
    registry = molecular_registry()
    return make_tools(registry, Executor(registry), store.freeze(spec, "reviewer"), store)


def test_r130_production_runs_need_qpu_bindings() -> None:
    tools = _sampling_tools()
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base_id = tools.call("modeler", "plan.base", {"plan_id": empty,
                                                  "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]
    built = tools.call("designer", "plan.candidates", {"plan_id": base_id, "candidates": [
        {"template": "sqd", "shots": 1024}]})["data"]
    screened = tools.call("designer", "run", {"plan_id": built["plan_id"], "scale": "screen"})
    assert screened["data"]["results"]
    with pytest.raises(Exception) as caught:
        tools.call("designer", "run", {"plan_id": built["plan_id"], "scale": "production"})
    assert "CF-E140" in str(caught.value) and "qpu" in str(caught.value).lower()


def test_r130_scripted_candidates_are_deployable(tmp_path) -> None:
    from chemflow.backends import QPU_CATALOG
    from chemflow.eval import runner

    out = tmp_path / "out"
    runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="scripted",
                         model=None, roles="designer", jobs=1, out=out))
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    candidates = [item for entry in entries if entry["state"] == "CRITIQUE"
                  for item in entry["observation"]["candidates"]]
    assert candidates
    assert {item["method"] for item in candidates} <= {"sci", "sqd"}
    for item in candidates:
        assert item["deployable"] is True, item["id"]
        if item["method"] == "sqd":
            assert item["deployment"] and all(
                implementation in QPU_CATALOG["sample_configurations"]
                for implementation in item["deployment"].values())


def test_r130_the_critic_never_escalates_an_undeployable_candidate() -> None:
    from chemflow.agent.critic import review

    candidate = {"id": "c", "method": "sqd", "relation": "variational_upper_bound",
                 "relation_status": "proved", "evidence": True, "gate_passed": True,
                 "error": 1e-6, "cost": {"wall_s": 0.0}, "selection_work": 1}
    control = {"id": "n6", "method": "ccsd_t", "relation": "approximate",
               "relation_status": "proved", "evidence": True, "gate_passed": False,
               "error": 1e-2, "cost": {"wall_s": 1.0}, "selection_work": 10 ** 6}
    assert review([candidate | {"deployable": False}], [control]).decision != "escalate"
    assert review([candidate | {"deployable": True}], [control]).decision == "escalate"
