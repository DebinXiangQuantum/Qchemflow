import json

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy, _redact
from chemflow.agent.tools import QueryInput, ToolRegistry


def test_journal_call_alias_preserves_redacted_arguments(tmp_path):
    call = {"id": "call-1", "name": "cards.search", "arguments": {"query": "secret"}}
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append(state="LLM_TOOL", role="designer", observation={}, decision={},
                   calls=[_redact(call, "secret")])
    entry = json.loads(journal.path.read_text())
    assert entry["calls"] == entry["tool_calls"]
    assert entry["calls"][0]["id"] == "call-1"
    assert "secret" not in journal.path.read_text()


def test_glm_call_and_unexpected_tool_error_are_redacted(tmp_path, monkeypatch):
    secret = "offline-test-key"
    monkeypatch.setenv("GLM_API_KEY", secret)
    tools = ToolRegistry()

    def broken(inputs):
        raise TypeError(secret)

    tools.register("cards.search", {"designer"}, broken, QueryInput)
    call = {"id": "call-1", "name": "cards.search", "arguments": {"query": secret}}
    client = FakeLLMClient([
        {"tool_calls": [call], "content": "", "usage": {}, "stop_reason": "tool_calls"},
        {"tool_calls": [], "content": "done", "usage": {}, "stop_reason": "stop"}])
    journal = Journal(tmp_path / "glm.jsonl")
    LLMPolicy("designer", client, tools, journal=journal, provider="glm").decide({})
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    entry = next(item for item in entries if item["state"] == "LLM_TOOL")
    assert entry["calls"][0] == {
        "id": "call-1", "name": "cards.search", "arguments": {"query": "[REDACTED]"}}
    assert entry["decision"]["error"]["code"] == "CF-E080"
    assert secret not in journal.path.read_text()
