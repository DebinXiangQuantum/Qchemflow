import json
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("provider", ["openai", "claude"])
def test_three_turn_native_prefixes_with_multiple_tool_results(provider):
    if provider == "openai":
        pytest.importorskip("openai")
        from test_m4a1_openai_adapter import response

        from chemflow.agent.llm_openai import OpenAIClient

        first = response(tool=True)
        first.choices[0].message.tool_calls.append(
            first.choices[0].message.tool_calls[0].model_copy(update={"id": "call-2"}))
        second = response(tool=True)
        second.choices[0].message.tool_calls[0].id = "call-3"
        responses = [first, second, response()]
    else:
        pytest.importorskip("anthropic")
        from test_m4a_claude_adapter import message

        from chemflow.agent.llm_anthropic import ClaudeClient

        def tool(identifier):
            return {"type": "tool_use", "id": identifier, "name": "plan__new",
                    "input": {"identifier": "p"}}

        responses = [message([tool("call-1"), tool("call-2")]),
                     message([tool("call-3")]),
                     message([{"type": "text", "text": "done"}], "end_turn")]
    requests = []
    responses = iter(responses)

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(responses)

    endpoint = SimpleNamespace(create=create)
    adapter = (OpenAIClient("offline", SimpleNamespace(chat=SimpleNamespace(completions=endpoint)))
               if provider == "openai" else
               ClaudeClient(client=SimpleNamespace(beta=SimpleNamespace(messages=endpoint))))
    tools = [{"name": "plan.new", "parameters": {"type": "object", "properties": {
        "identifier": {"type": "string"}}, "required": ["identifier"],
        "additionalProperties": False}}]
    messages = [{"role": "system", "content": "Stable contract"},
                {"role": "user", "content": "Task"}]
    for round_index in range(3):
        result = adapter.complete(messages, tools)
        assert result["error"] is None
        if round_index == 2:
            assert result["content"] == "done"
            assert result["tool_calls"] == []
            break
        messages.append({"role": "assistant", "content": result["content"],
                         "tool_calls": result["tool_calls"]})
        for call in result["tool_calls"]:
            assert call["name"] == "plan.new"
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps({"plan_id": call["id"]}),
                             "is_error": call["id"] == "call-2"})
        if round_index == 0:
            messages.append({"role": "user", "content": "Correct structured tool errors."})
    for previous, current in pairwise(requests):
        assert current["messages"][:len(previous["messages"])] == previous["messages"]
    native = requests[-1]["messages"]
    assert len([item for item in native if item["role"] == "assistant"]) == 2
    if provider == "openai":
        results = [item["tool_call_id"] for item in native if item["role"] == "tool"]
        assert results == ["call-1", "call-2", "call-3"]
        assistants = [item for item in native if item["role"] == "assistant"]
        assert [[call["id"] for call in item["tool_calls"]] for item in assistants] == [
            ["call-1", "call-2"], ["call-3"]]
    else:
        results = [block for item in native if item["role"] == "user" and
                   isinstance(item["content"], list) for block in item["content"]]
        assert [block["tool_use_id"] for block in results] == ["call-1", "call-2", "call-3"]
        assert [block["is_error"] for block in results] == [False, True, False]
        assert requests[0]["system"] == requests[1]["system"] == requests[2]["system"]
    changed = deepcopy(messages)
    changed[1]["content"] = "Rewritten task"
    with pytest.raises(ValueError, match="prefix"):
        adapter.complete(changed, tools)
