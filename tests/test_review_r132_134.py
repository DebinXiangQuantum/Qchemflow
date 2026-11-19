"""Reviewer tests for R132-R134 (written by Claude; implementers must not edit).

Found by the live glm-5.3 benchmark on the final V1/R130 code (2026-10-14, d290d30, full
suite). glm-5.3 used the new verification tools: 58 `algorithm.prove`, 55
`algorithm.falsify` and 32 `algorithm.validity` calls. Most of those calls failed:
- R133: falsify (and validity) failed on 20 tasks with "CF-E070: qpu_s budget exceeded". Every
  suite task signs `qpu_s: 0, qpu_shots: 0`. The family engine charged unknown or positive
  QPU amounts for purely classical or emulated verification work, and R99 forbids that on an
  exhausted dimension. The V1b/V1c reviewer tests used budgets without QPU keys, so they
  missed it.
- R132: falsify on 6 reaction tasks raised "Reaction active argument must equal the sum of
  declared species spaces" instead of working or refusing cleanly.
- R134: 32 validity calls failed with "requires a signed validity domain". The tool was
  offered for tasks that declare none.

Contract:
- R133: verification tools charge the task ledger only for resources they actually use.
  Classical and emulator work uses no QPU time or shots, and known zeros are recorded as
  zero (D-R2 known-zero defaults). With `qpu_s: 0, qpu_shots: 0` signed, validity and
  falsify run, and the ledger's QPU spend stays 0.
- R132: reaction-energy targets support the `geometry_scale` axis, which scales every
  species geometry about its own first atom, both as a signed validity axis and as the
  default falsification family. The species active spaces stay as signed.
- R134: `algorithm.validity` appears in a role's schemas only when the frozen task declares a
  validity domain. Direct calls keep their structured refusal.
"""
from __future__ import annotations

import test_review_v1c as v1c

NO_QPU = {"wall_s": 1800, "sim_shots": 200000, "qpu_s": 0, "qpu_shots": 0}


def test_r133_verification_tools_run_under_a_zero_qpu_budget() -> None:
    tools, base = v1c._tools(v1c._h4() | {"budget": NO_QPU})
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}, {"template": "sqd", "shots": 2048}]})["data"]
    for node in built["ids"].values():
        falsified = tools.call("critic", "algorithm.falsify", {"plan_id": built["plan_id"],
                                                               "node_id": node})["data"]
        assert falsified["verdict"] == "clean", node
        mapped = tools.call("critic", "algorithm.validity", {"plan_id": built["plan_id"],
                                                             "node_id": node})["data"]
        assert mapped["task_point"]["status"] in ("certified", "sampled_valid"), node
    ledger = tools.ledger if hasattr(tools, "ledger") else None
    if ledger is not None:
        assert (ledger.spent.get("qpu_s") or 0) == 0 and (ledger.spent.get("qpu_shots") or 0) == 0


def test_r132_reaction_targets_have_a_geometry_family() -> None:
    from chemflow.agent.task import TaskSpec, TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.domains import load_registry, pack_for_target
    from chemflow.eval.runner import load_suite
    from chemflow.runtime import Executor

    task = next(item for item in load_suite("core").tasks if item.id == "h2-atomization")
    spec_dict = task.spec.model_dump(mode="json") | {"validity": [
        {"axis": "geometry_scale", "range": [0.8, 2.0], "points": 5, "resolution": 0.1}]}
    spec = TaskSpec.model_validate(spec_dict)
    store = TaskStore("reviewer")
    registry = load_registry(["molecular", pack_for_target(spec.target.kind).name])
    tools = make_tools(registry, Executor(registry), store.freeze(spec, "reviewer"), store)
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty,
                                               "active": {"n_elec": 3, "n_orb": 3}})["data"]
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    node = next(iter(built["ids"].values()))
    falsified = tools.call("critic", "algorithm.falsify", {"plan_id": built["plan_id"],
                                                           "node_id": node})["data"]
    assert falsified["verdict"] == "clean" and len(falsified["instances"]) >= 5
    mapped = tools.call("critic", "algorithm.validity", {"plan_id": built["plan_id"],
                                                         "node_id": node})["data"]
    assert mapped["axes"][0]["axis"] == "geometry_scale"
    assert mapped["task_point"]["status"] in ("certified", "sampled_valid")


def test_r134_validity_is_offered_only_with_a_signed_domain() -> None:
    without, _ = v1c._tools(v1c._h4(validity=False))
    with_domain, _ = v1c._tools(v1c._h4())
    for role in ("designer", "critic"):
        assert "algorithm.validity" not in {item["name"] for item in without.schemas(role)}
        assert "algorithm.validity" in {item["name"] for item in with_domain.schemas(role)}
