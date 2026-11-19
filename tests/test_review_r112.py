"""Reviewer regression tests for R112 (written by Claude; implementers must not edit).

Found in the full-suite GLM benchmark (2026-10-13):
- 43 of 68 glm-4.7 calls returned HTTP 429 (code 1302, rate limit).
- Later every model returned 1113 (balance or resource package exhausted on the general
  endpoint; the user's token package is served at the Coding Plan endpoint).
- `LLMPolicy.decide` stopped at the first provider error, and the run silently fell back to the
  scripted controls. The harness then scored those runs as the LLM policy's decisions, which
  made them meaningless.

Contract:
- Retryable provider errors (`error.retryable` true: rate limits, connection errors, 5xx) are
  retried with exponential backoff: `retry_backoff_s` (default 2 s) doubling up to 60 s, at most
  `max_provider_attempts` (default 6) per model call.
- Retries count neither as turns nor as tool-error retries. Each failed attempt is journalled
  (state LLM_ERROR, with the attempt number). Non-retryable errors stop at once.
- A decision whose model call still failed is journalled as LLM_FAILED.
- In the eval harness, a row with any LLM_FAILED gets `llm_failed: true` and `valid: false`.
  `aggregate` reports `invalid` and computes `correct_valid` over valid rows only, so
  infrastructure failures are never scored as policy decisions.
"""
from __future__ import annotations

import json

import test_review_m4a as m4a

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy


def _error(code, retryable):
    return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
            "error": {"code": code, "message": f"{code} for test", "retryable": retryable}}


def _answer(text):
    return {"tool_calls": [], "content": text, "stop_reason": "stop", "usage": {}}


def _states(path):
    return [json.loads(line)["state"] for line in path.read_text().splitlines()]


def test_retryable_errors_are_retried_with_backoff(tmp_path, monkeypatch) -> None:
    from chemflow.agent import llm

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    client = FakeLLMClient([_error("rate_limit", True), _error("rate_limit", True),
                            _answer("final")])
    journal = tmp_path / "j.jsonl"
    result = LLMPolicy("designer", client, m4a._tools(), journal=Journal(journal),
                       retry_backoff_s=2.0).decide({})
    assert result["content"] == "final" and len(client.messages) == 3
    assert sleeps == [2.0, 4.0]
    assert _states(journal).count("LLM_ERROR") == 2 and "LLM_FAILED" not in _states(journal)


def test_non_retryable_errors_stop_and_exhausted_retries_fail_loudly(tmp_path, monkeypatch) -> None:
    from chemflow.agent import llm

    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    auth = FakeLLMClient([_error("authentication", False), _answer("never")])
    journal = tmp_path / "auth.jsonl"
    LLMPolicy("designer", auth, m4a._tools(), journal=Journal(journal)).decide({})
    assert len(auth.messages) == 1 and "LLM_FAILED" in _states(journal)
    limited = FakeLLMClient([_error("rate_limit", True)] * 10)
    journal = tmp_path / "limited.jsonl"
    LLMPolicy("designer", limited, m4a._tools(), journal=Journal(journal),
              max_provider_attempts=3).decide({})
    assert len(limited.messages) == 3 and "LLM_FAILED" in _states(journal)


def test_the_harness_never_scores_llm_failures_as_policy_decisions() -> None:
    from test_review_g5 import _suite

    from chemflow.eval.runner import EvaluationTask, aggregate_rows, score_task

    task = EvaluationTask.model_validate(next(item for item in _suite()["tasks"]
                                              if item["id"] == "h2-ground"))
    exact = task.expected.value_hartree
    summary = {"decision": "stop", "candidate": "n8", "spent": {"wall_s": 0.1}, "results": [
        {"node": "n8", "value": f"{exact} hartree", "relation": "equivalent",
         "gate_verdict": "PASS", "screening_reference": True}]}
    failed = score_task(task, summary, [{"state": "LLM_FAILED", "decision": {}}])
    clean = score_task(task, summary, [])
    assert failed["llm_failed"] and not failed["valid"]
    assert clean["valid"] and not clean["llm_failed"]
    aggregate = aggregate_rows([failed, clean])
    assert aggregate["invalid"] == 1 and aggregate["correct_valid"] == 1
