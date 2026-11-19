import json

from test_m3a_task import spec

from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore


def run_designer(tmp_path, responses):
    store = TaskStore("test-secret")
    client = FakeLLMClient(responses)
    policy = LLMPolicy("designer", client, None)
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=1, policies={"designer": policy})
    result = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    entries = [json.loads(line) for line in result.journal.read_text().splitlines()]
    return result, entries, client


class DesignerClient(FakeLLMClient):
    def complete(self, messages, tools):
        self.messages.append(list(messages))
        base = json.loads(messages[1]["content"])["observation"]["base_plan_id"]
        return {"tool_calls": [{"id": "candidate-1", "name": "plan.candidates",
                                "arguments": {"plan_id": base, "candidates": [
                                    {"template": "sci", "select_cutoff": 0.01}]}}],
                "content": "screen this plan", "usage": {"input_tokens": 12,
                "output_tokens": 4}, "request_id": "offline-1", "stop_reason": "tool_use"}


def test_llm_designer_screen_and_critic(tmp_path):
    store = TaskStore("test-secret")
    client = DesignerClient([])
    policy = LLMPolicy("designer", client, None)
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=1, policies={"designer": policy})
    result = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    entries = [json.loads(line) for line in result.journal.read_text().splitlines()]
    assert any(entry["state"] == "SCREEN" for entry in entries)
    assert any(entry["state"] == "CRITIQUE" for entry in entries)
    assert any(entry["state"] == "LLM_TOOL" for entry in entries)
    call = next(entry for entry in entries if entry["state"] == "LLM_CALL")
    assert call["decision"]["usage"]["input_tokens"] == 12
    assert call["decision"]["request_id"] == "offline-1"
    assert result.decision.decision in {"stop", "escalate"}


def test_adversarial_designer_refused_and_critic_retains_control(tmp_path):
    bad = [{"id": str(index), "name": name, "arguments": arguments} for index,
           (name, arguments) in enumerate([
               ("report.write", {"sections": {}}), ("run", {"scale": "production"}),
               ("plan.new", {"budget": {"wall_s": 1000}}),
               ("plan.new", {"evidence": "certified"}),
               ("missing.tool", {}),
               ("plan.candidates", {"plan": {"name": "bad", "gates": [{}]},
                                    "candidates": []})])]
    result, entries, _ = run_designer(tmp_path, [
        {"tool_calls": bad, "content": ""}, {"tool_calls": [], "content": "no plan"}])
    assert result.decision.decision == "stop"
    assert any(entry["state"] == "DESIGN_REFUSED" for entry in entries)
    assert any(entry["state"] == "CRITIQUE" for entry in entries)
    errors = [entry for entry in entries if entry["state"] == "LLM_TOOL" and
              "error" in entry["decision"]]
    assert len(errors) == len(bad)
    assert not any(entry["state"] == "SCREEN" for entry in entries)
