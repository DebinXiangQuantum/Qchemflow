import json
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
from anthropic.types.beta import BetaMessage

from chemflow.agent.llm_anthropic import ClaudeClient


def message(blocks, stop="tool_use", details=None):
    payload = {"id": "msg_1", "type": "message", "role": "assistant",
               "model": "claude-opus-5-5", "content": blocks, "stop_reason": stop,
               "stop_details": details, "stop_sequence": None,
               "usage": {"input_tokens": 7, "output_tokens": 9,
                         "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}}
    return BetaMessage.model_validate(payload)


class FakeMessages:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.responses)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_client(responses):
    endpoint = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=endpoint)), endpoint


def test_adapter_transcript_and_contract():
    tool_use = lambda identifier: {"type": "tool_use", "id": identifier,
                                    "name": "plan__new", "input": {"identifier": identifier}}
    client, endpoint = fake_client([message([tool_use("a"), tool_use("b")]),
                                    message([{"type": "text", "text": "done"}], "end_turn")])
    adapter = ClaudeClient(client=client)
    tools = [{"name": "plan.new", "parameters": {"type": "object", "properties": {
        "identifier": {"type": "string"}}, "required": ["identifier"],
        "additionalProperties": False}}, {"name": "check", "parameters": {
            "type": "object", "properties": {}, "required": [], "additionalProperties": False}}]
    conversation = [{"role": "system", "content": "Stable contract"},
                    {"role": "user", "content": "Task"}]
    first = adapter.complete(conversation, tools)
    assert [call["name"] for call in first["tool_calls"]] == ["plan.new"] * 2
    conversation.append({"role": "assistant", "content": first["content"],
                         "tool_calls": first["tool_calls"]})
    conversation.extend({"role": "tool", "tool_call_id": identifier,
                         "content": json.dumps({"ok": identifier}), "is_error": identifier == "b"}
                        for identifier in ("a", "b"))
    adapter.complete(conversation, tools)
    initial, subsequent = endpoint.calls
    assert initial["model"] == "claude-opus-5-5" and initial["max_tokens"] == 16000
    assert "thinking" not in initial and initial["tool_choice"] == {"type": "auto"}
    assert initial["output_config"] == {"effort": "high"}
    assert initial["fallbacks"] == "default" and initial["betas"]
    assert initial["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [tool["name"] for tool in initial["tools"]] == ["check", "plan__new"]
    assert all(tool["strict"] for tool in initial["tools"])
    assert subsequent["messages"][:len(initial["messages"])] == initial["messages"]
    assert subsequent["messages"][-2]["content"] == message([
        tool_use("a"), tool_use("b")]).content
    results = subsequent["messages"][-1]["content"]
    assert [item["tool_use_id"] for item in results] == ["a", "b"]
    assert [item["is_error"] for item in results] == [False, True]
    assert first["usage"]["cache_read_input_tokens"] == 3
    with pytest.raises(ValueError, match="prefix"):
        adapter.complete([{"role": "system", "content": "changed"}], tools)


def test_refusal_and_sdk_errors():
    refusal = message([{"type": "text", "text": "No"}], "refusal",
                      {"type": "refusal", "category": "general_harms", "explanation": "No"})
    client, _ = fake_client([refusal])
    result = ClaudeClient(client=client).complete([
        {"role": "system", "content": "Contract"}, {"role": "user", "content": "Task"}], [])
    assert result["error"]["code"] == "refusal"
    assert result["error"]["category"] == "general_harms"
    httpx = __import__("httpx2")
    request = httpx.Request("POST", "https://example.invalid")
    for exception, code, retryable in (
        (anthropic.BadRequestError("bad", response=httpx.Response(
            400, request=request), body=None), "bad_request", False),
        (anthropic.RateLimitError("slow", response=httpx.Response(
            429, request=request), body=None), "rate_limit", True)):
        client, _ = fake_client([exception])
        result = ClaudeClient(client=client).complete([
            {"role": "system", "content": "Contract"}, {"role": "user", "content": "Task"}], [])
        assert result["error"]["code"] == code
        assert result["error"]["retryable"] is retryable
