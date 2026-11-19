import copy

import pytest
from pydantic import ValidationError

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.domains import load_registry
from chemflow.ids import content_hash
from chemflow.runtime import Executor


def specification(kind="lattice"):
    sector = {"n_alpha": 1, "n_beta": 1, "n_orb": 2}
    targets = {
        "lattice": {"kind": "lattice_ground_energy", "lattice": {
            "model": "hubbard", "n_sites": 2, "t": 1., "U": 2., "unit": "hartree"},
            "sector": sector},
        "impurity": {"kind": "spectral_function", "model": {"impurity_model": {
            "U": 2., "eps_bath": [0.], "t_bath": [.5], "unit": "hartree"}},
            "sector": sector, "orbital": 0, "spin": "alpha",
            "omega": {"start": -4., "stop": 4., "n": 21, "unit": "hartree"}, "eta": .1},
        "molecular": {"molecule": {"atoms": "H 0 0 0; H 0 0 .74"},
            "electronic_model": {"basis": "sto-3g"}, "sector": sector,
            "active": {"n_elec": 2, "n_orb": 2}, "target_spin": 0}}
    spectral = kind == "impurity"
    axis = {"lattice": "hubbard_u", "impurity": "impurity_u",
            "molecular": "geometry_scale"}[kind]
    return {"question": "Family regression", "target": targets[kind],
            "gates": [{"kind": "l1_error_vs_exact" if spectral else "abs_error_vs_exact",
                       "threshold": .0016, "unit": "dimensionless" if spectral else "hartree"}],
            "budget": {"wall_s": 300}, "allowed_effects": ["pure", "simulator"],
            "decision": "d", "validity": [{"axis": axis,
                "range": [.7, 2.5] if kind == "molecular" else [0., 4.],
                "points": 3, "resolution": .1}]}


def toolset(spec, directory=None):
    store = TaskStore("test")
    record = store.freeze(TaskSpec.model_validate(spec), "human")
    registry = load_registry(["molecular", "materials", "algorithms", "dynamics"])
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store, plan_directory=directory)
    empty = tools.call("designer", "plan.new", {"identifier": "family"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty})["data"]
    return tools, base, executor


@pytest.mark.parametrize("kind", ["lattice", "impurity"])
def test_axes_use_exact_route_charge_budget_and_preserve_signature(kind, monkeypatch):
    tools, base, _ = toolset(specification(kind))
    search = tools.family_search
    original = search.record.model_dump()
    monkeypatch.setattr(search.store, "freeze", lambda *args: pytest.fail("Must not re-sign"))
    node = tools.plans.get(base["plan_id"]).references[0]
    result = tools.call("critic", "algorithm.validity", {
        "plan_id": base["plan_id"], "node_id": node})["data"]
    assert result["task_point"]["status"] == "sampled_valid"
    assert search.ledger.spent["wall_s"] > 0
    assert search.record.model_dump() == original
    for point in result["axes"][0]["points"]:
        assert point["error"] == pytest.approx(0., abs=1e-9)
        assert point["derived_from"] == search.record.content_hash
        if kind == "lattice":
            expected = (point["value"] - (point["value"] ** 2 + 16) ** .5) / 2
            assert point["exact"] == pytest.approx(expected)


@pytest.mark.parametrize("change", [
    {"points": 2}, {"points": 34}, {"points": 3.5}, {"points": True},
    {"range": [2., 2.]}, {"range": [4., 0.]}, {"range": [3., 4.]},
    {"range": [0., float("inf")]}, {"resolution": 0.}, {"resolution": 4.},
    {"resolution": float("nan")}, {"axis": "geometry_scale"}, {"extra": 1},
    {"range": [False, 4.]}, {"resolution": ".1"}])
def test_domain_validation(change):
    spec = specification()
    spec["validity"][0].update(change)
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(spec)


def test_signed_fields_are_not_dropped():
    spec = specification()
    original = TaskSpec.model_validate(spec)
    changed = copy.deepcopy(spec)
    changed["validity"][0]["resolution"] = .05
    assert content_hash(original) != content_hash(TaskSpec.model_validate(changed))
    for payload in (spec | {"unknown": 1},
                    spec | {"target": spec["target"] | {"unknown": 1}}):
        with pytest.raises(ValidationError):
            TaskSpec.model_validate(payload)


@pytest.mark.parametrize("resolution", [.1, 1e-30])
def test_bisection_brackets_boundary(monkeypatch, resolution):
    spec = specification()
    spec["validity"][0]["resolution"] = resolution
    tools, base, _ = toolset(spec)
    search = tools.family_search
    node = tools.plans.get(base["plan_id"]).references[0]

    def evaluate(candidate, reference, axis, value):
        return {"value": value, "error": 0 if value <= 2.71 else .01,
                "result": 0., "exact": 0., "certified_bound": None, "failures": []}

    monkeypatch.setattr(search, "evaluate", evaluate)
    if resolution < 1e-20:
        with pytest.raises(ValueError, match="floating-point spacing"):
            search.validity(tools.plans.get(base["plan_id"]), node)
        return
    result = search.validity(tools.plans.get(base["plan_id"]), node)
    gap = next(region for region in result["axes"][0]["regions"]
               if region["status"] == "unknown")
    assert gap["lo"] <= 2.71 <= gap["hi"]
    assert gap["hi"] - gap["lo"] <= .1
    assert len(result["axes"][0]["points"]) > 3


@pytest.mark.parametrize("plan_id", ["null", "", None])
def test_r129_nullable_decline(plan_id):
    tools, _, _ = toolset(specification())
    result = tools.call("designer", "design.decline", {
        "reason": "No supported proposal", "plan_id": plan_id})["data"]
    assert result


def test_full_ritz_state_has_zero_variance():
    spec = specification()
    tools, _, _ = toolset(spec)
    with build.program("ritz-family") as builder:
        ham = builder.node("lattice_model", **spec["target"]["lattice"])
        sector = builder.sector(**spec["target"]["sector"])
        basis = builder.node("full_subspace", sector=sector)
        ritz = builder.node("rayleigh_ritz", hamiltonian=ham, subspace=basis)
        builder.ir.targets = [ritz.id]
    plan_id = tools.plans.put(builder.to_ir())
    result = tools.call("critic", "algorithm.validity", {
        "plan_id": plan_id, "node_id": ritz.id})["data"]
    assert all(point["certified_bound"] < 1e-10 for point in result["axes"][0]["points"])
    assert result["task_point"]["status"] == "certified"


@pytest.mark.parametrize("method", ["sci", "sqd", "vqe"])
def test_variational_state_bounds(method):
    spec = specification("molecular")
    spec["target"]["target_spin"] = None
    tools, base, executor = toolset(spec)
    if method == "vqe":
        from chemflow.ir.schema import Node

        plan = tools.plans.get(base["plan_id"])
        reference = next(item for item in plan.nodes if item.id == plan.references[0])
        node = "verification_vqe"
        plan.nodes.append(Node(id=node, op="ground_energy", inputs=reference.inputs,
                               params=reference.params, refinement="vqe",
                               binding="vqe_statevector"))
        plan.targets.append(node)
        built = {"plan_id": tools.plans.put(plan)}
    else:
        built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"],
            "candidates": [{"template": method}]})["data"]
        node = next(iter(built["ids"].values()))
    result = tools.call("critic", "algorithm.validity", {
        "plan_id": built["plan_id"], "node_id": node})["data"]
    for point in result["axes"][0]["points"]:
        assert point["certified_bound"] is not None
        assert point["certified_bound"] + 1e-8 >= abs(point["error"])
    assert executor.store.memory
