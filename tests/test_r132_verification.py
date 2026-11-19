import numpy as np
import pytest
from pyscf.gto import format_atom
from test_v1c_domains import specification, toolset

from chemflow.agent.families import default_axes, instantiate
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.domains import load_registry
from chemflow.eval.runner import load_suite
from chemflow.runtime import Executor
from chemflow.runtime.budget import BudgetExceeded, BudgetLedger, resource_defaults
from chemflow.types import CostRecord

NO_QPU = {"wall_s": 1800, "sim_shots": 200000, "qpu_s": 0, "qpu_shots": 0}


@pytest.mark.parametrize("kind,method", [
    ("molecular", "reference"), ("lattice", "reference"), ("impurity", "reference"),
    ("molecular", "sqd"), ("lattice", "sqd")])
def test_verification_records_known_zero_qpu(kind, method, monkeypatch):
    tools, base, _ = toolset(specification(kind) | {"budget": NO_QPU})
    if method == "reference":
        node = tools.plans.get(base["plan_id"]).references[0]
    else:
        base = tools.call("designer", "plan.candidates", {
            "plan_id": base["plan_id"], "candidates": [
                {"template": "sqd", "shots": 2048}]})["data"]
        node = next(iter(base["ids"].values()))
    ledger = tools.family_search.ledger
    reservations, records = [], []
    reserve, record = ledger.reserve, ledger.record

    def reserve_cost(cost):
        reservations.append(cost)
        return reserve(cost)

    def record_cost(cost):
        records.append(cost)
        return record(cost)

    monkeypatch.setattr(ledger, "reserve", reserve_cost)
    monkeypatch.setattr(ledger, "record", record_cost)
    args = {"plan_id": base["plan_id"], "node_id": node}
    proof = tools.call("critic", "algorithm.prove", args)["data"]
    assert proof["kernel_verdict"]["valid"]
    falsified = tools.call("critic", "algorithm.falsify", args)["data"]
    assert falsified["verdict"] == "clean"
    mapped = tools.call("critic", "algorithm.validity", args)["data"]
    assert mapped["task_point"]["status"] in ("certified", "sampled_valid")
    if kind == "impurity":
        verified = tools.call("designer", "algorithm.verify", {
            "plan_id": base["plan_id"], "target_node": node,
            "instances": 3, "seed": 1729})["data"]
        assert verified["passed"]
    assert reservations and records
    assert all(cost.qpu_s == 0 and cost.qpu_shots == 0
               for cost in reservations + records)
    assert ledger.spent["qpu_s"] == ledger.spent["qpu_shots"] == 0
    assert ledger.spent["wall_s"] > 0
    if method == "sqd":
        assert ledger.spent["sim_shots"] > 0


def reaction_tools(signed):
    task = next(task for task in load_suite("core").tasks if task.id == "h2-atomization")
    spec = task.spec.model_dump(mode="json") | {"budget": NO_QPU}
    for index, species in enumerate(spec["target"]["species"]):
        atoms = format_atom(species["molecule"]["atoms"], unit="Bohr")
        species["molecule"]["atoms"] = "; ".join(
            symbol + " " + " ".join(str(value + index + 2) for value in position)
            for symbol, position in atoms)
    spec["validity"] = ([{"axis": "geometry_scale", "range": [.8, 2.],
                          "points": 3, "resolution": .1}] if signed else [])
    store = TaskStore("test")
    record = store.freeze(TaskSpec.model_validate(spec), "human")
    registry = load_registry(["molecular", "reactions"])
    tools = make_tools(registry, Executor(registry), record, store)
    empty = tools.call("designer", "plan.new", {"identifier": "reaction"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {
        "plan_id": empty, "active": {"n_elec": 3, "n_orb": 3}})["data"]
    built = tools.call("designer", "plan.candidates", {
        "plan_id": base["plan_id"], "candidates": [
            {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    return tools, built, record


@pytest.mark.parametrize("signed", [False, True])
def test_reaction_geometry_family_preserves_each_signed_species(signed):
    tools, built, record = reaction_tools(signed)
    original = record.model_dump()
    node = next(iter(built["ids"].values()))
    plan = tools.plans.get(built["plan_id"])
    candidate, reference = tools.family_search.prepare(plan, node)
    expected_spaces = [species.active for species in record.spec.target.species]
    for program in (candidate, reference):
        scaled = instantiate(program, "geometry_scale", 1.7)
        assert [item.params["active"] for item in scaled.nodes
                if item.op == "hamiltonian"] == expected_spaces
        for before, after in zip(program.nodes, scaled.nodes, strict=True):
            if before.op != "molecule":
                assert before == after
                continue
            original_atoms = format_atom(before.params["atoms"], unit="Bohr")
            scaled_atoms = format_atom(after.params["atoms"], unit="Bohr")
            positions = np.asarray([position for _, position in original_atoms])
            actual = np.asarray([position for _, position in scaled_atoms])
            np.testing.assert_allclose(actual, positions[0] + 1.7 * (positions - positions[0]))
            assert [symbol for symbol, _ in original_atoms] == [symbol for symbol, _ in scaled_atoms]
            assert {key: value for key, value in before.params.items() if key != "atoms"} == {
                key: value for key, value in after.params.items() if key != "atoms"}
    args = {"plan_id": built["plan_id"], "node_id": node}
    falsified = tools.call("critic", "algorithm.falsify", args)["data"]
    assert falsified["verdict"] == "clean"
    assert len(falsified["instances"]) >= 5
    assert falsified["domain"][0]["axis"] == "geometry_scale"
    if signed:
        mapped = tools.call("critic", "algorithm.validity", args)["data"]
        assert mapped["task_point"]["status"] in ("certified", "sampled_valid")
    else:
        with pytest.raises(ValueError, match="requires a signed validity domain"):
            tools.call("critic", "algorithm.validity", args)
    assert record.model_dump() == original


@pytest.mark.parametrize("kind", ["cpu_numeric", "quantum_simulator"])
def test_known_zero_defaults_do_not_exempt_consumed_resources(kind):
    ledger = BudgetLedger({"wall_s": 0, "qpu_s": 0, "qpu_shots": 0})
    with pytest.raises(BudgetExceeded, match="wall_s"):
        ledger.reserve(CostRecord(**resource_defaults({}, kind)))
    ledger = BudgetLedger({"qpu_s": 0, "qpu_shots": 0})
    ledger.reserve(CostRecord(**resource_defaults({}, kind)))
    for amount in (None, 1):
        with pytest.raises(BudgetExceeded, match="qpu_s"):
            ledger.reserve(CostRecord(**resource_defaults({"qpu_s": amount}, "qpu")))


def test_interaction_family_remains_explicitly_unsupported():
    task = next(task for task in load_suite("core").tasks
                if task.spec.target.kind == "interaction_energy")
    with pytest.raises(ValueError, match="No falsification family"):
        default_axes(task.spec.target)
