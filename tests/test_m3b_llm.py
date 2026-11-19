import json

from test_m3a_task import spec

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def policy(tmp_path, role, responses):
    store = TaskStore("secret")
    record = store.freeze(spec(), "human")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    client = FakeLLMClient(responses)
    journal = Journal(tmp_path / "llm.jsonl")
    return LLMPolicy(role, client, tools, record, journal), client, journal


def test_designer_builds_plan_via_tools(tmp_path):
    agent, client, _ = policy(tmp_path, "designer", [])
    base, _ = Orchestrator._base_plan(agent.record, agent.record.spec.target.active)
    plan_id = agent.tools.plans.put(base)
    client.responses = iter([{"tool_calls": [{"name": "plan.candidates", "arguments": {
        "plan_id": plan_id,
        "candidates": [{"template": "sci", "select_cutoff": 0.01}]}}],
        "content": "Plan ready"}])
    result = agent.decide({"cards": [{"template": "sci"}],
                           "diagnostics": {"array": [1, 2, 3]}})
    plan_id = result["outputs"][0]["result"]["data"]["plan_id"]
    assert not agent.tools.call("designer", "check", {"plan_id": plan_id})["data"]
    assert "[1, 2, 3]" not in str(client.messages)


def test_malicious_calls_refused_and_journaled(tmp_path):
    bad = [{"name": "run", "arguments": {"scale": "production"}},
           {"name": "plan.new", "arguments": {"params": {"evidence": "certified"}}},
           {"name": "report.write", "arguments": {"sections": {}}}]
    agent, client, journal = policy(tmp_path, "designer", [
        {"tool_calls": bad, "content": ""}, {"tool_calls": [], "content": "revised"}])
    result = agent.decide({})
    assert len([output for output in result["outputs"] if "error" in output]) == 3
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()
               if json.loads(line)["state"] == "LLM_TOOL"]
    assert len(entries) == 3
    assert all("error" in entry["decision"] for entry in entries)
    assert "Correct structured tool errors" in str(client.messages[-1])


def test_critic_cannot_flip_revision_without_an_answer(tmp_path):
    agent, _, _ = policy(tmp_path, "critic", [
        {"tool_calls": [], "content": "escalate"}])
    result = agent.decide({"candidates": [], "controls": []})
    assert result["verdict"]["decision"] == "revise"
    assert result["verdict"]["candidate"] is None
