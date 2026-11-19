import json

import pytest

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.tools import ToolRegistry


def error(code="rate_limit", retryable=True, message="offline failure"):
    return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
            "error": {"code": code, "retryable": retryable, "message": message}}


def answer(calls=None):
    return {"tool_calls": calls or [], "content": "done", "stop_reason": "stop", "usage": {}}


def run_policy(tmp_path, monkeypatch, responses, **options):
    sleeps = []
    monkeypatch.setattr("chemflow.agent.llm.time.sleep", sleeps.append)
    client = FakeLLMClient(responses)
    journal = Journal(tmp_path / "journal.jsonl")
    result = LLMPolicy("designer", client, ToolRegistry(), journal=journal,
                       **options).decide({})
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    return result, client, entries, sleeps


@pytest.mark.parametrize("backoff,expected", [
    (2.0, [2, 4, 8, 16, 32, 60, 60]),
    (100.0, [60] * 7),
    (0.0, [0] * 7),
])
def test_backoff_cap_and_attempt_accounting(tmp_path, monkeypatch, backoff, expected):
    result, client, entries, sleeps = run_policy(
        tmp_path, monkeypatch, [error()] * 7 + [answer()],
        retry_backoff_s=backoff, max_provider_attempts=8, max_turns=1, max_retries=0)
    assert sleeps == expected and len(client.messages) == 8
    assert result["content"] == "done"
    assert all(messages == client.messages[0] for messages in client.messages)
    errors = [entry["decision"] for entry in entries if entry["state"] == "LLM_ERROR"]
    assert [entry["attempt"] for entry in errors] == list(range(1, 8))
    assert all(entry["code"] == "rate_limit" and entry["turn"] == 1 for entry in errors)
    assert not any(entry["state"] in {"LLM_FAILED", "LLM_TURN_LIMIT"} for entry in entries)
    assert entries[-1]["decision"]["turns"] == 1


@pytest.mark.parametrize("max_retries,expected_calls,expected_turns", [(1, 4, 2), (2, 6, 3)])
def test_provider_retries_do_not_spend_or_reset_tool_budget(
        tmp_path, monkeypatch, max_retries, expected_calls, expected_turns):
    bad_tool = answer([{"name": "not.allowed", "arguments": {}}])
    result, client, entries, sleeps = run_policy(
        tmp_path, monkeypatch, [error(), bad_tool, error(), bad_tool, error(), answer()],
        max_retries=max_retries, max_turns=3, max_provider_attempts=2)
    assert len(client.messages) == expected_calls
    assert sleeps == [2] * expected_turns
    assert entries[-1]["decision"]["turns"] == expected_turns
    assert len(result["outputs"]) == 2
    assert all("error" in output for output in result["outputs"])
    assert not any(entry["state"] == "LLM_FAILED" for entry in entries)
    assert [entry["decision"]["attempt"] for entry in entries
            if entry["state"] == "LLM_ERROR"] == [1] * expected_turns


@pytest.mark.parametrize("retryable,attempts", [(False, 1), (True, 6)])
def test_terminal_failure_metadata_and_key_redaction(tmp_path, monkeypatch, retryable, attempts):
    secret = "r112-private-key"
    monkeypatch.setenv("GLM_API_KEY", secret)
    result, client, entries, sleeps = run_policy(
        tmp_path, monkeypatch, [error(retryable=retryable, message=secret)] * 7,
        provider="glm", max_provider_attempts=6)
    failed = [entry for entry in entries if entry["state"] == "LLM_FAILED"]
    assert len(client.messages) == attempts and len(sleeps) == attempts - 1
    assert len(failed) == 1 and failed[0]["role"] == "designer"
    assert failed[0]["decision"] == {"code": "rate_limit", "attempts": attempts}
    assert secret not in json.dumps(entries) + json.dumps(result)
    assert "[REDACTED]" in json.dumps(entries)


@pytest.mark.parametrize("options", [
    {"max_provider_attempts": 0}, {"max_provider_attempts": True},
    {"max_provider_attempts": 1.5}, {"retry_backoff_s": -1},
    {"retry_backoff_s": float("nan")}, {"retry_backoff_s": float("inf")},
])
def test_invalid_retry_configuration(options):
    with pytest.raises(ValueError):
        LLMPolicy("designer", FakeLLMClient([]), ToolRegistry(), **options)


def test_unrecovered_malformed_response_is_invalid(tmp_path, monkeypatch):
    _, _, entries, _ = run_policy(tmp_path, monkeypatch, [None], max_turns=1)
    assert any(entry["state"] == "LLM_FAILED" and
               entry["decision"]["code"] == "malformed_response" for entry in entries)
