import json

import pytest
from test_m3a_task import spec

from chemflow.agent.memory import ResultsDB
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.report import ReportLintError
from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


@pytest.fixture
def recorded():
    store = TaskStore("reports")
    record = store.freeze(spec(), "human")
    memory = ResultsDB()
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store, memory)
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    output = tools.call("designer", "run", {"plan_id": tools.plans.put(plan)})["data"]
    return tools, memory, output


def report(tools, headline, markdown="Screening summary."):
    return tools.call("analyst", "report.write", {
        "markdown": markdown, "headlines_json": json.dumps([headline])})["data"]


def test_report_resolves_and_renders_recorded_fields(recorded):
    tools, _, output = recorded
    headline = next(iter(output["headlines"].values()))
    text = report(tools, {"result_id": headline["result_id"]})
    for field in ("node", "value", "relation", "evidence", "gate_verdict", "backend_kind"):
        assert headline[field] in text
    assert report(tools, headline) == text


@pytest.mark.parametrize("field,value", [
    ("node", "invented"), ("value", "-100 hartree"), ("relation", "unrelated"),
    ("evidence", "certified_bound"), ("gate_verdict", "invented"),
    ("cost", {"money": 0}), ("backend_kind", "qpu"), ("advantage_evidence", True),
])
def test_report_rejects_each_forged_field(recorded, field, value):
    tools, _, output = recorded
    identifier = next(iter(output["results"].values()))
    with pytest.raises(ReportLintError):
        report(tools, {"result_id": identifier, field: value})


def test_claims_require_a_recorded_supporting_critic_verdict(recorded):
    tools, memory, output = recorded
    identifier = next(iter(output["results"].values()))
    headline = {"result_id": identifier}
    for verdict in (None, {"decision": "escalate", "reason": "gate passed"}):
        if verdict:
            memory.record_critic(identifier, verdict)
        with pytest.raises(ReportLintError, match="Unsupported claim"):
            report(tools, headline, "Quantum advantage and speedup observed.")
    memory.record_critic(identifier, {"decision": "supported",
                         "supported_claims": ["quantum advantage", "speedup"]})
    assert "Quantum advantage" in report(tools, headline, "Quantum advantage and speedup observed.")
    with pytest.raises(ReportLintError, match="Numeric headline"):
        report(tools, headline, "The value is -100 hartree.")


def test_results_from_another_task_cannot_be_reported(recorded):
    _, memory, output = recorded
    store = TaskStore("other-task")
    record = store.freeze(spec().model_copy(update={"question": "Another task"}), "human")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store, memory)
    with pytest.raises(ReportLintError, match="this task"):
        report(tools, {"result_id": next(iter(output["results"].values()))})
