import pytest
from test_m3a_task import spec

from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def test_permissions_and_schemas():
    store = TaskStore("private-key")
    record = store.freeze(spec(), "human")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    with pytest.raises(ValueError, match="plan_id"):
        tools.call("designer", "run", {"scale": "production"})
    with pytest.raises(PermissionError):
        tools.call("designer", "task.freeze", {})
    schema = tools.schemas("designer")
    assert any(item["name"] == "run" for item in schema)
    assert "private-key" not in str(schema)
    assert "private-key" not in str(tools.call("designer", "plan.new", {}))


def test_screen_results_are_json_and_queryable():
    from chemflow.agent.orchestrator import Orchestrator

    store = TaskStore("private-key")
    record = store.freeze(spec(), "human")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    output = tools.call("designer", "run", {"plan_id": tools.plans.put(plan)})
    assert output["data"]["gates"][0]["passed"]
    assert tools.call("analyst", "results.query", {"task_id": record.content_hash})[
        "data"]
