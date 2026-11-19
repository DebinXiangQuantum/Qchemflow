"""Reviewer regression tests for A1c (written by Claude; implementers must not edit).

The accuracy bar belongs to the problem, so it is written in the program (an `accept` gate on
the target node, with no reference: the verifier brings the independent exact one) or, under a
task, it is the frozen task's gate. Verification reads it from there; no caller or agent passes
a tolerance. Reports and registered designs record the bar they met, and a design is not used
where a stricter bar is required until it is verified at that bar. User decision 2026-10-12,
prompted by a test threshold borrowed from a different algorithm.
"""
from __future__ import annotations

import pytest
import test_review_a1a as a1a
import test_review_a1b as a1b
from pydantic import ValidationError

from chemflow.agent.task import TaskGate, TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.runtime import Executor

STRICT = {**a1a.ACCURACY, "threshold": 1e-3}
OTHER = {"U": 3.0, "eps_bath": [-1.0, 0.1, 1.4], "t_bath": [0.4, 0.7, 0.3], "unit": "hartree"}


def _verify(program, node, **kwargs):
    from chemflow.design import DesignStore, verify_design

    store = kwargs.pop("store", None) or DesignStore()
    return verify_design(program, node, target=a1b.TARGET, registry=kwargs.pop("registry", None)
                         or a1b._registry(), store=store, instances=2, seed=11, **kwargs)


def test_verification_reads_the_bar_from_the_program_and_nowhere_else() -> None:
    from chemflow.ir import AcceptanceGate
    from chemflow.units import Q

    bare, _, response = a1a._designed_plan(full=True)
    with pytest.raises(ValueError):  # no declared bar: nothing to verify against
        _verify(bare, response)
    program, _, response = a1a._designed_plan(full=True)
    with pytest.raises(TypeError):  # callers cannot supply one
        _verify(a1a._accept(program, response), response, tolerance=10.0)
    own_reference, _, response = a1a._designed_plan(full=True)
    own_reference.gates.append(AcceptanceGate(node=response, l1_error_le=Q(0.1, "dimensionless"),
                                              reference=response))
    with pytest.raises(ValueError):  # the candidate may not choose its own reference
        _verify(own_reference, response)
    wrong_measure, _, response = a1a._designed_plan(full=True)
    a1a._accept(wrong_measure, response,
                {"kind": "abs_error_vs_exact", "threshold": 0.1, "unit": "hartree"})
    with pytest.raises(ValueError):  # an energy bar says nothing about a spectral function
        _verify(wrong_measure, response)


def test_the_declared_bar_decides_the_verdict_and_is_recorded() -> None:
    program, response = a1a._enriched_plan()
    loose = _verify(a1a._accept(program, response, {**a1a.ACCURACY, "threshold": 10.0}),
                    response)
    program, response = a1a._enriched_plan()
    strict = _verify(a1a._accept(program, response, STRICT), response)
    assert loose.passed and not strict.passed  # same design, same instances, different bars
    assert [item["metric"] for item in loose.instances] == [
        item["metric"] for item in strict.instances]
    assert loose.accuracy["threshold"] == 10.0 and strict.accuracy["threshold"] == 1e-3
    assert loose.accuracy["source"] == strict.accuracy["source"] == "program"
    wrong, response = a1b._wrong_orbital_plan()
    assert _verify(wrong, response).accuracy["threshold"] == a1a.ACCURACY["threshold"]


def test_l1_gates_are_dimensionless() -> None:
    assert TaskGate.model_validate(a1a.ACCURACY).unit == "dimensionless"
    with pytest.raises(ValidationError):  # the L1 integral of A(omega) d omega has no unit
        TaskGate.model_validate({**a1a.ACCURACY, "unit": "hartree"})


def test_the_task_gate_is_the_bar_for_agents() -> None:
    from chemflow.agent.tools import make_tools
    from chemflow.domains import get_pack

    def tools_for(gates):
        spec = TaskSpec.model_validate({
            "question": "AIM impurity spectral function", "target": a1b.TARGET, "gates": gates,
            "budget": {"wall_s": 900}, "allowed_effects": ["pure", "simulator"],
            "decision": "d"})
        store = TaskStore("reviewer")
        record = store.freeze(spec, "reviewer")
        registry = a1b._registry()
        tools = make_tools(registry, Executor(registry), record, store)
        base, _ = get_pack("dynamics").base_plan(record)
        designed, _, response = a1a._designed_plan(full=True)
        plan = base.model_copy(deep=True)
        known = {node.id for node in plan.nodes}
        plan.nodes.extend(node for node in designed.nodes if node.id not in known)
        return tools, tools.plans.put(plan), response

    tools, plan_id, response = tools_for([a1a.ACCURACY])
    request = {"plan_id": plan_id, "target_node": response, "instances": 2, "seed": 11}
    with pytest.raises(Exception):  # noqa: B017 - the agent cannot pass a tolerance
        tools.call("designer", "algorithm.verify", {**request, "tolerance": 10.0})
    tools, plan_id, response = tools_for([{"kind": "relation_at_least",
                                           "relation": "approximate"}])
    with pytest.raises(Exception):  # noqa: B017 - a task without an accuracy gate has no bar
        tools.call("designer", "algorithm.verify", {**request, "plan_id": plan_id})


def test_r98_a_required_relation_is_part_of_the_bar_wherever_it_is_declared() -> None:
    """Reviewer finding R98: the task path ignored the task's relation_at_least gate."""
    from chemflow.ir import AcceptanceGate

    loose = {**a1a.ACCURACY, "threshold": 10.0}  # every sampled design meets the number
    store = TaskStore("reviewer")
    record = store.freeze(TaskSpec.model_validate({
        "question": "q", "target": a1b.TARGET, "decision": "d",
        "gates": [loose, {"kind": "relation_at_least", "relation": "equivalent"}]}), "reviewer")
    program, response = a1a._enriched_plan()
    by_task = _verify(program, response, task=record)
    program, response = a1a._enriched_plan()
    a1a._accept(program, response, loose)
    program.gates.append(AcceptanceGate(node=response, relation="equivalent"))
    by_program = _verify(program, response)
    for report in (by_task, by_program):
        assert report.inferred_relation == "approximate"
        assert report.accuracy["relation"] == "equivalent" and not report.passed


def test_a_design_is_not_used_at_a_stricter_bar_than_it_met() -> None:
    from chemflow.design import DesignStore, instantiate_design, register_design
    from chemflow.ir import AcceptanceGate
    from chemflow.runtime.task import freeze_task
    from chemflow.units import Q

    registry, store = a1b._registry(), DesignStore()
    program, response = a1b._exact_plan()
    met = _verify(program, response, registry=registry, store=store)
    register_design(registry, store, met.id, name="full_lehmann")

    def instance(threshold):
        plan, target = instantiate_design(store, "full_lehmann", model_params=OTHER)
        if threshold is not None:
            plan.gates.append(AcceptanceGate(node=target, l1_error_le=Q(threshold,
                                                                        "dimensionless")))
        return plan, target

    def run(plan):  # gates are frozen with a task before anything runs (CF-E110)
        return Executor(registry).run(plan, freeze_task(plan))

    plan, target = instance(None)
    result = run(plan)[target]
    assert result.diagnostics["design_accuracy"] == met.accuracy
    run(instance(a1a.ACCURACY["threshold"])[0])  # the bar it met: allowed
    with pytest.raises((AdapterError, CheckError, ValueError)):  # stricter: re-verify first
        run(instance(STRICT["threshold"])[0])
    program, response = a1b._exact_plan()
    program.gates.clear()
    stricter = _verify(a1a._accept(program, response, STRICT), response, registry=registry,
                       store=store)
    assert stricter.passed
    register_design(registry, store, stricter.id, name="full_lehmann_strict")
    plan, target = instantiate_design(store, "full_lehmann_strict", model_params=OTHER)
    plan.gates.append(AcceptanceGate(node=target, l1_error_le=Q(1e-3, "dimensionless")))
    run(plan)
