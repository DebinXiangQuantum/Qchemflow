"""Reviewer tests for R135-R136 (written by Claude; implementers must not edit).

Found by the glm-5.3 confirmation run on 4e01666 (2026-10-14).

- R135: `algorithm.falsify` failed outright in 6 tasks (16 calls) with "SCF did not converge".
  Some instance of the family, usually an extreme geometry, has no converged mean-field
  reference, and one such instance aborted the whole search. A single unevaluable instance
  must not hide the instances that could be checked, and it must not be mistaken for a
  counterexample either.
- R136: `plan.add_node` crashed (CF-E080 "unhashable type: 'dict'") when glm-5.3 gave a
  `component_energy` node a dict-valued `method` parameter. `infer_relations` used the
  parameter as a dict key. A malformed parameter must produce a checker diagnostic, not a
  crash.

Contract:
- R135: in falsification and validity mapping, an instance whose model setup or exact
  reference cannot be computed is recorded under `unevaluable`: `[{"instance", "reason"}]`.
  The search continues.
  - Falsify: unevaluable instances are not counterexamples, and the verdict is computed over
    the evaluated instances.
  - Validity maps: such points have status "unknown" with a `reason`, and so do the regions
    they bound.
  - Only if every instance is unevaluable does the tool fail, with a structured error.
- R136: `check()` and `infer_relations()` never raise on malformed node parameters. A
  `component_energy` whose `method` is not a known method string yields an error diagnostic
  naming the parameter.
"""
from __future__ import annotations

import json

import test_review_v1c as v1c

from chemflow.diagnostics import AdapterError


def _stretched_scf_fails(monkeypatch, limit=6.6):
    from chemflow.ops import molecular

    original = molecular.build_molecule

    def build(atoms, *args, **kwargs):
        coordinates = [float(value) for atom in str(atoms).split(";")
                       for value in atom.split()[1:]]
        if coordinates and max(abs(value) for value in coordinates) > limit:
            raise AdapterError("CF-E080", None, "SCF did not converge", "test")
        return original(atoms, *args, **kwargs)

    monkeypatch.setattr(molecular, "build_molecule", build)


def test_r135_unevaluable_instances_do_not_abort_falsification(monkeypatch) -> None:
    _stretched_scf_fails(monkeypatch)  # H4 spacing beyond 2.2 angstrom (last atom beyond 6.6)
    tools, base = v1c._tools(v1c._h4())
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    node = next(iter(built["ids"].values()))
    report = tools.call("critic", "algorithm.falsify", {"plan_id": built["plan_id"],
                                                        "node_id": node})["data"]
    assert report["verdict"] == "clean" and not report["counterexamples"]
    assert report["unevaluable"]
    assert all(item["instance"]["geometry_scale"] > 2.2 - 1e-9 for item in report["unevaluable"])
    assert all("SCF" in item["reason"] for item in report["unevaluable"])
    evaluated = [item["geometry_scale"] for item in report["instances"]]
    assert any(value <= 2.2 for value in evaluated)


def test_r135_validity_maps_mark_unevaluable_points_unknown(monkeypatch) -> None:
    _stretched_scf_fails(monkeypatch)
    tools, base = v1c._tools(v1c._h4())
    node = v1c._method(tools, base["plan_id"], "ccsd_t")
    mapped = tools.call("critic", "algorithm.validity", {"plan_id": base["plan_id"],
                                                         "node_id": node})["data"]
    axis = mapped["axes"][0]
    stretched = [point for point in axis["points"] if point["value"] > 2.2 + 1e-9]
    assert stretched and all(point["status"] == "unknown" and point.get("reason")
                             for point in stretched)
    assert v1c._region(axis, 2.5)["status"] == "unknown"
    assert v1c._region(axis, 1.0)["status"] in ("sampled_valid", "certified")
    assert v1c._region(axis, 2.0)["status"] == "invalid"  # still found where evaluable


def test_r136_malformed_parameters_are_diagnosed_not_crashed() -> None:
    from chemflow.agent.task import TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.check.static import check, infer_relations
    from chemflow.domains import load_registry, pack_for_target
    from chemflow.eval.runner import load_suite
    from chemflow.runtime import Executor

    task = next(item for item in load_suite("full").tasks if item.id == "h2-dimer-no-cp")
    store = TaskStore("reviewer")
    registry = load_registry(["molecular", pack_for_target(task.spec.target.kind).name])
    tools = make_tools(registry, Executor(registry), store.freeze(task.spec, "reviewer"), store)
    plan_id = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    node = {"id": "t1", "op": "chemflow.component_energy", "inputs": [], "effects": "pure",
            "refinement": None, "binding": None, "params": [
                {"key": "molecule", "value_json": json.dumps({"atoms": "H 0 0 0; H 0 0 0.74"})},
                {"key": "electronic_model", "value_json": json.dumps({"basis": "sto-3g"})},
                {"key": "method", "value_json": json.dumps({"template": "sci"})}]}
    added = tools.call("designer", "plan.add_node", {"plan_id": plan_id, "node": node,
                                                     "target": True})["data"]
    assert any("method" in item["message"].lower() for item in added["diagnostics"])
    plan = tools.plans.get(added["plan_id"])
    infer_relations(plan, registry)  # must not raise
    assert check(plan, registry)
