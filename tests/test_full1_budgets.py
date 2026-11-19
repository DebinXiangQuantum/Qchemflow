import json

import pytest
from test_m3a_task import spec

from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.diagnostics import AdapterError
from chemflow.ir.schema import Node
from chemflow.ops.molecular import molecular_registry
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ExecutionResult, Executor
from chemflow.types import Samples


def task(budget, *, paid=False):
    data = spec().model_dump()
    data["budget"] = budget
    if paid:
        data.update(allowed_effects=("pure", "simulator"))
    store = TaskStore("budget")
    return store, store.freeze(TaskSpec.model_validate(data), "human")


def sampling_plan(record, shots=8, count=1, binding="sample_configurations"):
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    hamiltonian = next(node.id for node in plan.nodes if node.op == "hamiltonian")
    sector = next(node.id for node in plan.nodes if node.op == "sector")
    plan.nodes.append(Node(id="prep", op="prepare_state",
                           inputs={"hamiltonian": hamiltonian, "sector": sector}))
    for index in range(count):
        plan.nodes.append(Node(id=f"samples{index}", op="sample_configurations", binding=binding,
                               inputs={"prep": "prep"}, params={"shots": shots, "seed": index}))
    return plan


def test_actual_device_spend_blocks_next_submission():
    store, record = task({"qpu_s": 20, "wall_s": 600}, paid=True)
    registry = molecular_registry()
    calls = []

    def paid(store, prep, shots, seed):
        calls.append(shots)
        return ExecutionResult(Samples(counts={"5": shots}, shots=shots,
            encoding_id=prep.encoding_id, basis="computational", source="qpu", seed=seed),
            cost={"qpu_s": 15.0})

    registry.register(ImplementationSpec(id="paid", version="1", target="sample_configurations",
        backend_kind="qpu", effects="random", adapter=paid, params=frozenset({"shots", "seed"}),
        resource_model=lambda **kwargs: {"qpu_s": 10.0}))
    plan = sampling_plan(record, count=2, binding="paid")
    computed = Executor(registry).run_plan(plan, record, store)
    assert "samples0" in computed
    assert "CF-E070" in computed["refused"]["samples1"]
    assert calls == [8]


def test_whole_run_prediction_is_refused_before_any_adapter():
    store, record = task({"sim_shots": 8, "wall_s": 600})
    registry = molecular_registry()
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store)
    plan = sampling_plan(record, shots=5, count=2)
    with pytest.raises(AdapterError, match="CF-E070"):
        tools.call("designer", "run", {"plan_id": tools.plans.put(plan)})
    assert executor.cache == {}
    assert executor.store.memory == {}


def test_task_ledger_survives_tool_registry_restart_and_cache_is_free(tmp_path):
    store, record = task({"sim_shots": 8, "wall_s": 600})
    registry = molecular_registry()
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store, plan_directory=tmp_path)
    identifier = tools.plans.put(sampling_plan(record))
    tools.call("designer", "run", {"plan_id": identifier})
    ledger_path = next(tmp_path.glob("*.ledger.json"))
    spent = json.loads(ledger_path.read_text())
    assert spent["sim_shots"] == 8
    base, _ = Orchestrator._base_plan(record, record.spec.target.active)
    cached = tools.call("designer", "run", {"plan_id": tools.plans.put(base)})["data"]
    assert all(cost["wall_s"] == cost["sim_shots"] == cost["qpu_s"] == 0
               for cost in cached["costs"].values())
    assert json.loads(ledger_path.read_text()) == spent
    restarted = make_tools(registry, Executor(registry), record, store, plan_directory=tmp_path)
    with pytest.raises(AdapterError, match="CF-E070"):
        restarted.call("designer", "run", {"plan_id": identifier})
    assert json.loads(ledger_path.read_text()) == spent


def test_measured_overrun_is_persisted_and_reported(tmp_path):
    store, record = task({"sim_shots": 5, "wall_s": 600})
    registry = molecular_registry()
    calls = []

    def underestimate(store, prep, shots, seed):
        calls.append(shots)
        return Samples(counts={"5": 6}, shots=6, encoding_id=prep.encoding_id,
                       basis="computational", source="quantum_simulator", seed=seed)

    registry.implementations["sample_configurations"].adapter = underestimate
    tools = make_tools(registry, Executor(registry), record, store, plan_directory=tmp_path)
    identifier = tools.plans.put(sampling_plan(record, shots=4))
    computed = tools.call("designer", "run", {"plan_id": identifier})["data"]
    assert computed["budget_overruns"] == [
        {"node": "samples0", "dimension": "sim_shots", "limit": 5, "spent": 6}]
    assert json.loads(next(tmp_path.glob("*.ledger.json")).read_text())["sim_shots"] == 6
    restarted = make_tools(registry, Executor(registry), record, store, plan_directory=tmp_path)
    with pytest.raises(AdapterError, match="CF-E070"):
        restarted.call("designer", "run", {"plan_id": identifier})
    assert calls == [4]


def test_mcp_profiles_share_persistent_task_budget(tmp_path):
    from chemflow.mcp import create_tools
    from chemflow.signing import TaskSigner, generate_keypair

    private_key, public_key = generate_keypair(b"human")
    record = TaskSigner(private_key, b"human").freeze(
        spec().model_copy(update={"budget": {"sim_shots": 8, "wall_s": 600}}), "human")
    task_path, key_path = tmp_path / "task.json", tmp_path / "public.pem"
    task_path.write_text(record.model_dump_json())
    key_path.write_bytes(public_key)
    tools = create_tools("designer", str(task_path), public_key=str(key_path))
    identifier = tools.plans.put(sampling_plan(record))
    tools.call("designer", "run", {"plan_id": identifier})
    other = create_tools("full", str(task_path), public_key=str(key_path))
    with pytest.raises(AdapterError, match="CF-E070"):
        other.call("orchestrator", "run", {"plan_id": identifier})
