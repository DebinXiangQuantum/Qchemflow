"""Reviewer regression tests for R123-R124 (written by Claude; implementers must not edit).

Found in the post-fix GLM matrix (2026-10-14, 7b9fb46). Three or four GLM configurations ran
concurrently against the Coding Plan endpoint, and 15 of 220 rows were invalid (LLM_FAILED):
- R123: 13 came from sustained rate limiting (code 1302). R112 retries 6 times with 2 s
  backoff doubling, which is about 62 s of waiting in all, and the endpoint's limit outlasts
  that. The attempt budget must cover several minutes of throttling.
- R124: 2 came from `invalid_tool_arguments`. The model emitted a tool call whose arguments
  were not valid JSON. OpenAIClient turned the whole response into a non-retryable provider
  error, so the decision failed. This is a model mistake the model can correct, the same as a
  schema validation error.

Contract:
- R123: `max_provider_attempts` defaults to at least 10, and backoff still doubles from
  `retry_backoff_s` up to a 60 s cap. That gives several minutes before LLM_FAILED.
- R124: a tool call whose arguments are not valid JSON reaches `LLMPolicy` as a call. The
  model gets a structured tool error that quotes the parse problem, and the loop continues.
  The response is not a provider error, and LLM_FAILED is not journalled. Valid calls in the
  same response are still executed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import test_review_m4a as m4a

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy

openai = pytest.importorskip("openai")


def _states(path):
    return [json.loads(line)["state"] for line in path.read_text().splitlines()]


def test_r123_rate_limits_are_retried_for_minutes(tmp_path, monkeypatch) -> None:
    from chemflow.agent import llm

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    limited = {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
               "error": {"code": "1302", "message": "rate limit", "retryable": True}}
    client = FakeLLMClient([limited] * 9 + [{"tool_calls": [], "content": "final",
                                             "stop_reason": "stop", "usage": {}}])
    journal = tmp_path / "j.jsonl"
    policy = LLMPolicy("designer", client, m4a._tools(), journal=Journal(journal))
    assert policy.max_provider_attempts >= 10
    assert policy.decide({})["content"] == "final"
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0, 60.0]
    assert "LLM_FAILED" not in _states(journal)


def _completion(arguments):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-offline", "object": "chat.completion", "created": 1,
        "model": "glm-test", "usage": {"prompt_tokens": 8, "completion_tokens": 3,
                                       "total_tokens": 11},
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": None, "tool_calls": [
                         {"id": f"call-{index}", "type": "function",
                          "function": {"name": "cf_" + name.encode().hex(),
                                       "arguments": text}}
                         for index, (name, text) in enumerate(arguments)]}}]})


def test_r124_unparseable_tool_arguments_go_back_to_the_model(tmp_path) -> None:
    from chemflow.agent.llm_openai import OpenAIClient

    responses = [
        _completion([("cards.search", '{"query": "sqd"'),  # truncated JSON
                     ("cards.search", '{"query": "sci"}')]),
    ]

    class Endpoint:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if responses:
                return responses.pop(0)
            from openai.types.chat import ChatCompletion

            return ChatCompletion.model_validate({
                "id": "done", "object": "chat.completion", "created": 1, "model": "glm-test",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}]})

    endpoint = Endpoint()
    client = OpenAIClient("glm-test", SimpleNamespace(chat=SimpleNamespace(completions=endpoint)))
    journal = tmp_path / "j.jsonl"
    result = LLMPolicy("designer", client, m4a._tools(), journal=Journal(journal)).decide({})
    assert result["content"] == "done" and len(endpoint.calls) == 2
    assert "LLM_FAILED" not in _states(journal)
    bad, good = result["outputs"]
    assert "error" in bad and "json" in json.dumps(bad["error"]).lower()
    assert "result" in good
    second_request = endpoint.calls[1]["messages"]
    assert sum(message["role"] == "tool" for message in second_request) == 2
