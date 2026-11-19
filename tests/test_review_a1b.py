"""Reviewer regression tests for A1b (written by Claude; implementers must not edit).

The design loop: an algorithm the agent assembled from primitives is verified on random
instances of the problem family against the exact reference for the *target* (not for the
design's own parameters), and only a passing, server-recorded report can register it as an
experimental design that later tasks may use. A label alone never makes a design right.

Contract (A1b brief): `chemflow.design.{DesignStore, verify_design, register_design,
instantiate_design}`, `dynamics.random_instance`, tools `algorithm.verify`,
`algorithm.register`, `algorithm.list` (designer may verify and register; others may list).
Revised for A1c: the accuracy bar is declared in the program (`accept` gate) or the frozen
task, never passed by the caller (`a1a.ACCURACY`, `a1a._accept`).
"""
from __future__ import annotations

import pytest
import test_review_a1a as a1a

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.runtime import Executor

TARGET = {"kind": "spectral_function", "model": {"impurity_model": a1a.AIM},
          "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "orbital": 0, "spin": "alpha",
          "omega": a1a.OMEGA, "eta": a1a.ETA}
BAR = a1a.ACCURACY["threshold"]


def _registry():
    from chemflow.domains import load_registry

    return load_registry(a1a.PACKS)


def _exact_plan():
    program, _, response = a1a._designed_plan(full=True)
    return a1a._accept(program, response), response


def _wrong_orbital_plan():
    """Exact machinery, wrong quantity: the impurity is orbital 0, this asks for orbital 1."""
    program, response = _exact_plan()
    node = next(item for item in program.nodes if item.id == response)
    node.params["orbital"] = 1
    return program, response


def test_dynamics_pack_draws_random_instances_of_the_problem_family() -> None:
    import numpy as np

    from chemflow.domains import get_pack

    rng = np.random.default_rng(3)
    draws = [get_pack("dynamics").random_instance("impurity_model", rng, n_bath=3)
             for _ in range(3)]
    assert len({str(draw) for draw in draws}) == 3
    assert all(len(draw["eps_bath"]) == len(draw["t_bath"]) == 3 for draw in draws)


def test_verification_passes_an_exact_design_and_records_evidence() -> None:
    from chemflow.design import DesignStore, verify_design

    store = DesignStore()
    program, response = _exact_plan()
    report = verify_design(program, response, target=TARGET, registry=_registry(),
                           store=store, instances=3, seed=11)
    assert report.passed and report.inferred_relation == "equivalent"
    assert len(report.instances) == 3
    assert len({str(item["params"]) for item in report.instances}) == 3
    assert all(item["metric"] <= 1e-8 for item in report.instances)  # what "equivalent" means
    assert {key: report.accuracy[key] for key in a1a.ACCURACY} == a1a.ACCURACY
    assert report.accuracy["source"] == "program"
    assert store.get(report.id) == report  # recorded server-side


def test_verification_rejects_a_design_that_computes_the_wrong_quantity() -> None:
    from chemflow.design import DesignStore, verify_design

    store = DesignStore()
    program, response = _wrong_orbital_plan()
    report = verify_design(program, response, target=TARGET, registry=_registry(),
                           store=store, instances=3, seed=11)
    assert not report.passed
    assert max(item["metric"] for item in report.instances) > BAR


def test_sampled_design_reports_metrics_resources_and_an_honest_verdict() -> None:
    from chemflow.design import DesignStore, verify_design

    store = DesignStore()
    program, response = a1a._enriched_plan()
    report = verify_design(a1a._accept(program, response), response, target=TARGET,
                           registry=_registry(), store=store, instances=2, seed=5)
    assert report.inferred_relation == "approximate"
    assert report.accuracy["threshold"] == BAR
    assert report.passed == all(item["metric"] <= BAR for item in report.instances)
    assert report.resources["sim_shots"] >= 2 * 20000
    assert report.resources["qubits"] == 8  # prepared circuits on 2 x 4 spin orbitals


def test_only_passing_server_recorded_reports_can_register_a_design() -> None:
    from chemflow.design import DesignStore, register_design, verify_design

    registry, store = _registry(), DesignStore()
    wrong, wrong_target = _wrong_orbital_plan()
    failing = verify_design(wrong, wrong_target, target=TARGET, registry=registry,
                            store=store, instances=2, seed=11)
    with pytest.raises((ValueError, PermissionError)):
        register_design(registry, store, failing.id, name="wrong_orbital")
    with pytest.raises((ValueError, PermissionError, KeyError)):
        register_design(registry, store, "not-a-recorded-report", name="forged")
    program, response = _exact_plan()
    passing = verify_design(program, response, target=TARGET, registry=registry, store=store,
                            instances=2, seed=11)
    entry = register_design(registry, store, passing.id, name="full_lehmann")
    assert entry["id"] == "design:full_lehmann" and entry["status"] == "experimental"
    assert entry["evidence"] == passing.id
    assert entry["accuracy"] == passing.accuracy  # the bar it met travels with it
    assert any("design:full_lehmann" in str(item) for item in registry.search("design:"))


def test_a_registered_design_runs_on_a_new_instance_and_is_marked_experimental() -> None:
    from chemflow.design import DesignStore, instantiate_design, register_design, verify_design

    registry, store = _registry(), DesignStore()
    program, response = _exact_plan()
    report = verify_design(program, response, target=TARGET, registry=registry, store=store,
                           instances=2, seed=11)
    register_design(registry, store, report.id, name="full_lehmann")
    other = {"U": 3.0, "eps_bath": [-1.0, 0.1, 1.4], "t_bath": [0.4, 0.7, 0.3],
             "unit": "hartree"}
    instance, target = instantiate_design(store, "full_lehmann", model_params=other)
    result = Executor(registry).run(instance)[target]
    _, omega, oracle = a1a._oracle(other, 2, 2)
    assert a1a._l1(omega, result.spectral, oracle) < 1e-8
    assert result.diagnostics["design"] == "design:full_lehmann"
    assert result.diagnostics["design_status"] == "experimental"
    assert result.diagnostics["design_evidence"] == report.id


def test_designers_verify_and_register_through_the_tools() -> None:
    from chemflow.agent.tools import make_tools
    from chemflow.domains import get_pack

    spec = TaskSpec.model_validate({
        "question": "AIM impurity spectral function", "target": TARGET,
        "gates": [a1a.ACCURACY],
        "budget": {"wall_s": 900}, "allowed_effects": ["pure", "simulator"], "decision": "d"})
    store = TaskStore("reviewer")
    record = store.freeze(spec, "reviewer")
    registry = _registry()
    tools = make_tools(registry, Executor(registry), record, store)
    base, _ = get_pack("dynamics").base_plan(record)
    designed, _, response = a1a._designed_plan(full=True)
    plan = base.model_copy(deep=True)
    known = {node.id for node in plan.nodes}
    plan.nodes.extend(node for node in designed.nodes if node.id not in known)
    plan_id = tools.plans.put(plan)
    verified = tools.call("designer", "algorithm.verify", {
        "plan_id": plan_id, "target_node": response, "instances": 2, "seed": 11})["data"]
    assert verified["passed"] and verified["inferred_relation"] == "equivalent"
    assert {key: verified["accuracy"][key] for key in a1a.ACCURACY} == a1a.ACCURACY
    assert verified["accuracy"]["source"] == "task"  # the frozen task's gate, not the agent
    with pytest.raises(PermissionError):
        tools.call("analyst", "algorithm.register", {"report_id": verified["report_id"],
                                                     "name": "full_lehmann"})
    with pytest.raises(Exception):  # noqa: B017 - unknown report ids are refused
        tools.call("designer", "algorithm.register", {"report_id": "made-up",
                                                      "name": "forged"})
    registered = tools.call("designer", "algorithm.register", {
        "report_id": verified["report_id"], "name": "full_lehmann"})["data"]
    assert registered["status"] == "experimental"
    listed = tools.call("analyst", "algorithm.list", {})["data"]
    assert any(item["id"] == "design:full_lehmann" for item in listed)
