"""Reviewer regression tests for R106 (written by Claude; implementers must not edit).

Found by the first live GLM benchmark (2026-10-13). `LLMPolicy.decide` called the model once,
ran its tool calls, and stopped: its loop retried only on tool errors. The model therefore never
saw its own tool results. A model that inspects before acting (glm-5.3 rendered the plan and
searched the cards first) could never propose anything, so every live run fell back to the
classical controls.

Contract: `decide` is an agent loop. After each round of tool calls, the results go back to the
model, and it is called again until it answers without tool calls or `max_turns` model calls
have been made (a constructor parameter, default ≥ 8). Hitting the limit is journalled
(state "LLM_TURN_LIMIT"). Error-correction retries (`max_retries`) still apply to consecutive
failing tool rounds. The final content is the model's last answer.
"""
from __future__ import annotations

import test_review_m4a as m4a

from chemflow.agent.llm import FakeLLMClient, LLMPolicy


def _turn(calls, content=""):
    return {"tool_calls": calls, "content": content,
            "stop_reason": "tool_calls" if calls else "stop", "usage": {}}


def _call(identifier, name, arguments):
    return {"id": identifier, "name": name, "arguments": arguments}


def _policy(client, tmp_path, **kwargs):
    from chemflow.agent.journal import Journal

    journal = Journal(tmp_path / "journal.jsonl")
    tools = m4a._tools()
    return LLMPolicy("designer", client, tools, journal=journal, **kwargs), journal


def test_the_model_sees_its_tool_results_and_continues_until_it_answers(tmp_path) -> None:
    client = FakeLLMClient([
        _turn([_call("c1", "registry.search", {"query": "fci", "kind": "refinement"})],
              "I'll inspect the registry first."),
        _turn([_call("c2", "cards.search", {"query": "sqd"})], "Now the method cards."),
        _turn([], "Proposed: sqd with 4096 shots."),
    ])
    policy, _ = _policy(client, tmp_path)
    result = policy.decide({})
    assert len(client.messages) == 3, "the model was not called again after its tool calls"
    second = client.messages[1]
    assert any(message["role"] == "tool" for message in second)  # it saw the first result
    assert result["content"] == "Proposed: sqd with 4096 shots."
    assert [output["name"] for output in result["outputs"]] == ["registry.search",
                                                               "cards.search"]


def test_the_loop_is_bounded_and_says_so(tmp_path) -> None:
    import json

    client = FakeLLMClient([
        _turn([_call(f"c{index}", "cards.search", {"query": "sqd"})]) for index in range(50)])
    policy, _ = _policy(client, tmp_path, max_turns=4)
    policy.decide({})
    assert len(client.messages) == 4
    states = [json.loads(line)["state"] for line in
              (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert "LLM_TURN_LIMIT" in states
