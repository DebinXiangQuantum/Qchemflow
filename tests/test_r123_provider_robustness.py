import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.tools import QueryInput, ToolRegistry


def test_default_retry_budget_exhaustion(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("chemflow.agent.llm.time.sleep", sleeps.append)
    failure = {"tool_calls": [], "content": "", "error": {
        "code": "1302", "message": "rate limit", "retryable": True}}
    client = FakeLLMClient([failure] * 11)
    journal = Journal(tmp_path / "journal.jsonl")
    policy = LLMPolicy("designer", client, ToolRegistry(), journal=journal, max_turns=1)
    policy.decide({})
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert policy.max_provider_attempts == 10
    assert len(client.messages) == 10
    assert sleeps == [2, 4, 8, 16, 32, 60, 60, 60, 60]
    assert all(messages == client.messages[0] for messages in client.messages)
    assert [entry["decision"]["attempt"] for entry in entries
            if entry["state"] == "LLM_ERROR"] == list(range(1, 11))
    assert [entry["decision"] for entry in entries if entry["state"] == "LLM_FAILED"] == [
        {"code": "1302", "attempts": 10}]


def _tools(executed):
    tools = ToolRegistry()
    tools.register("cards.search", {"designer"},
                   lambda data: executed.append(data.query) or data.query, QueryInput)
    return tools


def _openai_completion(arguments):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 1, "model": "offline",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if arguments else "stop",
                     "message": {"role": "assistant", "content": "" if arguments else "done",
                                 "tool_calls": [
                                     {"id": f"call-{index}", "type": "function", "function": {
                                         "name": "cf_" + b"cards.search".hex(),
                                         "arguments": text}}
                                     for index, text in enumerate(arguments)]}}]})


@pytest.mark.parametrize("arguments", ['{"query": "broken"', '', '{not json}'])
def test_openai_parse_error_and_corrected_call(tmp_path, arguments):
    pytest.importorskip("openai")
    from chemflow.agent.llm_openai import OpenAIClient

    with pytest.raises(json.JSONDecodeError) as caught:
        json.loads(arguments)
    responses = iter([_openai_completion([arguments, '{"query": "sibling"}']),
                      _openai_completion(['{"query": "corrected"}']),
                      _openai_completion([])])
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(responses)

    client = OpenAIClient("offline", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    executed = []
    journal = Journal(tmp_path / "journal.jsonl")
    result = LLMPolicy("designer", client, _tools(executed), journal=journal).decide({})
    assert result["content"] == "done"
    assert executed == ["sibling", "corrected"]
    bad, sibling, corrected = result["outputs"]
    assert str(caught.value) in bad["error"]["message"]
    assert "json" in bad["error"]["message"].lower()
    assert "result" in sibling and "result" in corrected
    second = requests[1]["messages"]
    assistant = next(message for message in second if message["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == arguments
    results = [message for message in second if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == ["call-0", "call-1"]
    assert json.loads(results[0]["content"]) == bad
    assert requests[2]["messages"][:len(second)] == second
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert not any(entry["state"] in {"LLM_ERROR", "LLM_FAILED"} for entry in entries)


def test_parse_marker_uses_tool_error_budget_without_executing(tmp_path):
    response = {"content": "", "tool_calls": [{"id": "bad", "name": "cards.search",
                "arguments": {}, "arguments_error": "Expecting value: line 1 column 1"}]}
    client = FakeLLMClient([deepcopy(response) for _ in range(3)])
    executed = []
    journal = Journal(tmp_path / "journal.jsonl")
    result = LLMPolicy("designer", client, _tools(executed), journal=journal,
                       max_retries=1).decide({})
    assert result["end_reason"] == "consecutive tool errors"
    assert len(client.messages) == 2 and not executed
    assert all("Expecting value" in output["error"]["message"] for output in result["outputs"])
    assert "LLM_FAILED" not in journal.path.read_text()


@pytest.mark.parametrize("invalid_input", ['{"query":', [], None])
def test_claude_malformed_input_is_already_a_tool_error(tmp_path, invalid_input):
    pytest.importorskip("anthropic")
    from chemflow.agent.llm_anthropic import ClaudeClient

    responses = iter([
        SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(), content=[
            SimpleNamespace(type="tool_use", id="bad", name="cards__search",
                            input=invalid_input),
            SimpleNamespace(type="tool_use", id="good", name="cards__search",
                            input={"query": "sibling"})]),
        SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(), content=[
            SimpleNamespace(type="text", text="done")])])
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(responses)

    client = ClaudeClient(client=SimpleNamespace(beta=SimpleNamespace(
        messages=SimpleNamespace(create=create))))
    executed = []
    journal = Journal(tmp_path / "journal.jsonl")
    result = LLMPolicy("designer", client, _tools(executed), journal=journal).decide({})
    assert result["content"] == "done" and executed == ["sibling"]
    assert "error" in result["outputs"][0] and "result" in result["outputs"][1]
    results = requests[1]["messages"][2]["content"]
    assert [item["tool_use_id"] for item in results] == ["bad", "good"]
    assert [item["is_error"] for item in results] == [True, False]
    assert "LLM_FAILED" not in journal.path.read_text()
