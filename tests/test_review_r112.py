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


def _error(code, retryable):
    return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
            "error": {"code": code, "message": f"{code} for test", "retryable": retryable}}


def _answer(text):
    return {"tool_calls": [], "content": text, "stop_reason": "stop", "usage": {}}


def _states(path):
    return [json.loads(line)["state"] for line in path.read_text().splitlines()]


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
