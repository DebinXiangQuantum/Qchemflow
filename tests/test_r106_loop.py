import json

import pytest
from test_m3b_llm import policy


def turn(name=None, identifier="call", content="", arguments=None):
    return {"content": content, "tool_calls": [] if name is None else [
        {"id": identifier, "name": name,
         "arguments": arguments if arguments is not None else {"query": "sqd"}}],
        "usage": {"input_tokens": 7, "output_tokens": 3}, "request_id": identifier}


def entries(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def test_retries_reset_after_success_and_keep_all_outputs(tmp_path):
    agent, client, journal = policy(tmp_path, "designer", [
        turn("missing", "bad-1"), turn("cards.search", "good-1"),
        turn("missing", "bad-2"), turn("cards.search", "good-2"),
        turn(content="Final answer")])
    agent.max_retries = 1
    result = agent.decide({})
    assert result["content"] == "Final answer"
    assert [item["name"] for item in result["outputs"]] == [
        "missing", "cards.search", "missing", "cards.search"]
    assert len(client.messages) == 5
    for index in (1, 3):
        assert client.messages[index][-1]["content"] == "Correct structured tool errors."
        assert client.messages[index][-2]["is_error"] is True
    for index in (2, 4):
        assert client.messages[index][-1]["role"] == "tool"
        assert client.messages[index][-1]["is_error"] is False
    records = entries(journal)
    calls = [entry["decision"] for entry in records if entry["state"] == "LLM_CALL"]
    assert [call["turn"] for call in calls] == list(range(1, 6))
    assert all(call["usage"] == {"input_tokens": 7, "output_tokens": 3} for call in calls)
    assert all(call["latency_s"] >= 0 for call in calls)
    assert [call["request_id"] for call in calls] == [
        "bad-1", "good-1", "bad-2", "good-2", "call"]
    assert records[-1]["decision"] == {"turns": 5, "end_reason": "text reply"}
    assert not any(entry["state"] == "LLM_TURN_LIMIT" for entry in records)


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_consecutive_failures_exhaust_only_retry_budget(tmp_path, retries):
    agent, client, journal = policy(tmp_path, "designer", [turn("missing")] * 20)
    agent.max_retries = retries
    result = agent.decide({})
    assert len(client.messages) == retries + 1
    assert len(result["outputs"]) == retries + 1
    assert not any(entry["state"] == "LLM_TURN_LIMIT" for entry in entries(journal))


@pytest.mark.parametrize("limit", [1, 4, 12])
def test_turn_limit_includes_last_tool_result_and_summary(tmp_path, limit):
    agent, client, journal = policy(tmp_path, "designer", [
        turn("cards.search", str(index), content=str(index)) for index in range(20)])
    assert agent.max_turns == 24
    agent.max_turns = limit
    result = agent.decide({})
    assert len(client.messages) == len(result["outputs"]) == limit
    assert result["content"] == str(limit - 1)
    limited = [entry for entry in entries(journal) if entry["state"] == "LLM_TURN_LIMIT"]
    assert len(limited) == 1
    assert limited[0]["role"] == "designer"
    assert limited[0]["decision"] == {"turns": limit}


def test_final_answer_at_limit_is_not_truncation(tmp_path):
    agent, client, journal = policy(tmp_path, "designer", [turn(content="Done")])
    agent.max_turns = 1
    assert agent.decide({}) == {"content": "Done", "outputs": [], "end_reason": "text reply"}
    assert len(client.messages) == 1
    assert not any(entry["state"] == "LLM_TURN_LIMIT" for entry in entries(journal))


def test_malformed_calls_are_journalled_and_bounded(tmp_path):
    agent, client, journal = policy(tmp_path, "designer", [None] * 20)
    agent.decide({})
    assert len(client.messages) == 3
    calls = [entry for entry in entries(journal) if entry["state"] in {"LLM_CALL", "LLM_ERROR"}]
    assert len(calls) == 3
    assert [entry["decision"]["turn"] for entry in calls] == [1, 2, 3]


def test_non_object_tool_call_can_be_corrected(tmp_path):
    malformed = turn()
    malformed["tool_calls"] = [None]
    agent, client, _ = policy(tmp_path, "designer", [malformed, turn(content="Corrected")])
    result = agent.decide({})
    assert result["content"] == "Corrected"
    assert result["outputs"][0]["error"]["code"] == "TypeError"
    assert client.messages[-1][-2]["is_error"] is True
