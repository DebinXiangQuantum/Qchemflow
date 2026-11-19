"""Offline contract checks against the installed SDK's own response models."""

import json
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
from openai.types.chat import ChatCompletion

from chemflow.agent.llm_openai import OpenAIClient


def response(tool=False):
    return ChatCompletion.model_validate({
        "id": "chatcmpl-offline", "object": "chat.completion", "created": 1,
        "model": "explicit-test-model", "usage": {"prompt_tokens": 8,
        "completion_tokens": 3, "total_tokens": 11},
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop",
                     "message": {"role": "assistant", "content": None if tool else "done",
                                 "tool_calls": [{"id": "call-1", "type": "function",
                                                 "function": {"name": "cf_706c616e2e6e6577",
                                                              "arguments": '{"identifier":"p"}'}}]
                                 if tool else None}}]})


class FakeEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return response(len(self.calls) == 1)


def test_typed_tool_calls_and_append_only_transcript():
    endpoint = FakeEndpoint()
    adapter = OpenAIClient("explicit-test-model", SimpleNamespace(
        chat=SimpleNamespace(completions=endpoint)))
    schema = {"type": "object", "properties": {"identifier": {"type": "string"}},
              "required": ["identifier"], "additionalProperties": False}
    tools = [{"name": "plan.new", "description": "Create a server-side plan handle.",
              "parameters": schema}]
    conversation = [{"role": "system", "content": "Contract"},
                    {"role": "user", "content": "Task"}]
    first = adapter.complete(conversation, tools)
    assert first["tool_calls"] == [{"id": "call-1", "name": "plan.new",
                                    "arguments": {"identifier": "p"}}]
    assert first["usage"] == {"input_tokens": 8, "output_tokens": 3}
    conversation.append({"role": "assistant", "content": first["content"],
                         "tool_calls": first["tool_calls"]})
    conversation.append({"role": "tool", "tool_call_id": "call-1",
                         "content": json.dumps({"ok": True}), "is_error": False})
    assert adapter.complete(conversation, tools)["content"] == "done"
    assert endpoint.calls[0]["tools"][0]["function"]["strict"] is True
    assert endpoint.calls[0]["tool_choice"] == "auto"
    assert endpoint.calls[1]["messages"][:2] == endpoint.calls[0]["messages"]
    assert endpoint.calls[1]["messages"][-1]["role"] == "tool"
    with pytest.raises(ValueError, match="prefix"):
        adapter.complete(conversation[:1], tools)


def test_explicit_model_required():
    with pytest.raises((ValueError, TypeError)):
        OpenAIClient("")
