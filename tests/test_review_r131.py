"""Reviewer tests for R131 (written by Claude; implementers must not edit).

Found 2026-10-14. The live glm-5.3 rerun on 2b1d608 hit the GLM Coding Plan's 5-hour usage
window: HTTP 429 with code 1308, "已达到 5 小时的使用上限。您的限额将在 2026-10-14 18:20:12
重置。" ("You have reached the 5-hour usage limit; your quota resets at 2026-10-14 18:20:12").

ChemFlow treated 1308 as an ordinary rate limit, so every decision retried with backoff and
then failed. The harness then started every remaining task anyway, producing a run full of
invalid rows that each spent minutes in retries.

Contract:
- `status_error` classifies a usage-window or quota stop as non-retryable. That covers code
  1308, the message "使用上限", "usage limit" with a reset time, and the existing balance
  codes such as 1113. The code is kept.
- `quota_reset(message)` returns the reset timestamp string, e.g. "2026-10-14 18:20:12",
  or None.
- The eval harness stops scheduling tasks after the first task whose journal shows
  LLM_FAILED with a quota code. Each remaining task gets a row with `decision: "not_run"`,
  `valid: false`, `completed: false`, `correct: false`, and an `error` naming the quota and
  its reset time.
  - Tasks already running finish normally.
  - `aggregate` counts `not_run`, and report.md states that the run stopped on the
    provider quota, with the reset time.
  - The exit status is 0.
"""
from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

MESSAGE = ("Error code: 429 - {'error': {'code': '1308', 'message': '已达到 5 小时的使用上限。"
           "您的限额将在 2026-10-14 18:20:12 重置。'}}")


def test_r131_usage_window_is_not_retried() -> None:
    from chemflow.agent.llm_errors import quota_reset, status_error

    exc = SimpleNamespace(status_code=429, code=None, body={"error": {
        "code": "1308", "message": "已达到 5 小时的使用上限。您的限额将在 2026-10-14 18:20:12 重置。"}})
    code, retryable = status_error(exc)
    assert code == "1308" and retryable is False
    assert quota_reset(MESSAGE) == "2026-10-14 18:20:12"
    assert quota_reset("rate limited, try later") is None
    throttled = SimpleNamespace(status_code=429, code=None, body={"error": {
        "code": "1302", "message": "rate limit"}})
    assert status_error(throttled) == ("1302", True)


def test_r131_the_harness_stops_on_a_quota_and_marks_the_rest_not_run(tmp_path,
                                                                      monkeypatch) -> None:
    from chemflow.agent import cli
    from chemflow.eval import runner

    class Exhausted:
        model = "fake-glm"

        def complete(self, messages, tools):
            return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
                    "error": {"code": "1308", "message": MESSAGE, "retryable": False}}

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Exhausted())
    out = tmp_path / "out"
    results = runner.run(Namespace(suite="core", suite_file=None,
                                   tasks="h2-ground,h2-atomization,h2-budget-refusal",
                                   policy="glm", model=None, roles="designer", jobs=1, out=out))
    rows = {row["id"]: row for row in results["rows"]}
    assert set(rows) == {"h2-ground", "h2-atomization", "h2-budget-refusal"}
    first = rows["h2-ground"]
    assert first["llm_failed"] and not first["valid"]
    for task in ("h2-atomization", "h2-budget-refusal"):
        row = rows[task]
        assert row["decision"] == "not_run" and not row["valid"] and not row["correct"]
        assert "2026-10-14 18:20:12" in row["error"]
    assert results["aggregate"]["not_run"] == 2
    report = (out / "report.md").read_text()
    assert "quota" in report.lower() and "2026-10-14 18:20:12" in report
