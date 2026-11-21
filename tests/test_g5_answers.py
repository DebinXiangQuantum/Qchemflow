from __future__ import annotations

import json
import sqlite3
from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_g5_evaluation import arguments, synthetic

from chemflow.agent import cli as agent_cli
from chemflow.eval.runner import Expected, aggregate_rows, load_suite, run, score_task


@pytest.mark.parametrize("candidate,correct", [("chosen", True), ("bad", False),
                                              (None, False), ("missing", False)])
def test_answer_selection_cannot_be_rescued_by_reference(candidate, correct):
    task, summary, journal = synthetic()
    summary["candidate"] = candidate
    row = score_task(task, summary, journal)
    assert row["answer_node"] == candidate
    assert row["reference_correct"] and row["correct"] is correct
    if candidate in (None, "missing"):
        assert row["answer_value"] is None and not row["gate_passed"]


def test_stop_recommended_control_is_scored_without_inferring_a_choice():
    task, summary, journal = synthetic()
    summary["candidate"] = "bad"
    row = score_task(task, summary, journal)
    assert row["decision"] == "stop" and row["answer_value"] == 900
    assert not row["correct"] and not row["gate_passed"]
    summary["candidate"] = "chosen"
    summary["results"].append(deepcopy(summary["results"][1]))
    assert not score_task(task, summary, journal)["correct"]


@pytest.mark.parametrize("decisions,actual,expected", [
    (None, "stop", None), (["stop"], "stop", True), (["stop"], "revise", False),
    (["stop", "escalate"], "escalate", True), (["refused"], "failed", False),
])
def test_decision_expectations_are_independent(decisions, actual, expected):
    task, summary, journal = synthetic()
    task = task.model_copy(update={"expected": task.expected.model_copy(
        update={"decision_in": decisions})})
    summary["decision"] = actual
    row = score_task(task, summary, journal)
    assert row["decision_ok"] is expected and row["correct"]
    assert aggregate_rows([row])["decision_ok"] == int(expected is True)


@pytest.mark.parametrize("expected", [
    {}, {"value_hartree": float("inf")}, {"outcome": "refused"},
    {"outcome": "refused", "refusal_code": "not-a-code"},
    {"outcome": "refused", "refusal_code": "CF-E070", "decision_in": []},
    {"outcome": "refused", "refusal_code": "CF-E070", "decision_in": ["imagined"]},
    {"value_hartree": -1, "refusal_code": "CF-E070"},
])
def test_expectation_schema(expected):
    with pytest.raises(ValidationError):
        Expected.model_validate(expected)


def refusal_case():
    task = next(task for task in load_suite("core").tasks if task.expected.outcome == "refused")
    spent = {"wall_s": 0, "sim_shots": 0, "qpu_s": 0, "qpu_shots": 0}
    summary = {"decision": "refused", "candidate": None, "spent": spent,
               "refusal": {"code": "CF-E070", "node": "n3", "nothing_consumed": True,
                           "spent_at_refusal": spent.copy()}, "results": []}
    return task, summary


def test_refusal_requires_code_and_positive_no_consumption_evidence():
    task, summary = refusal_case()
    row = score_task(task, summary, [])
    assert row["correct"] and row["completed"] and row["decision_ok"]
    assert row["refusal_code"] == "CF-E070" and not row["reference_correct"]
    assert not row["gate_passed"] and row["answer_value"] is None
    for updates in ({"code": "CF-E080"}, {"nothing_consumed": False},
                    {"spent_at_refusal": {}}, {"spent_at_refusal": None}):
        changed = deepcopy(summary)
        changed["refusal"].update(updates)
        assert not score_task(task, changed, [])["correct"]
    summary["spent"]["sim_shots"] = 1
    assert not score_task(task, summary, [])["correct"]
    summary["spent"]["sim_shots"] = None
    assert not score_task(task, summary, [])["correct"]
    assert not score_task(task, None, [{"reason": "CF-E070"}])["correct"]


def test_prior_costs_allowed_but_nothing_spent_beyond_refused_node():
    task, summary = refusal_case()
    summary["spent"]["wall_s"] = 0.25
    summary["refusal"]["spent_at_refusal"]["wall_s"] = 0.25
    assert score_task(task, summary, [])["correct"]
    summary["spent"]["wall_s"] += 0.01
    assert not score_task(task, summary, [])["correct"]


def test_budget_refusal_before_execution_through_shared_agent(tmp_path):
    results = run(arguments(tmp_path / "out", tasks="h2-budget-refusal"))
    row = results["rows"][0]
    assert row["correct"] and row["decision"] == "refused"
    assert row["wall_s"] == row["sim_shots"] == row["qpu_s"] == row["qpu_shots"] == 0
    directory = tmp_path / "out/h2-budget-refusal"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["candidate"] is None and summary["refusal"]["nothing_consumed"]
    with sqlite3.connect(directory / "results.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    journal = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    assert journal[-1]["state"] == "REFUSED"
    assert not any(entry["state"] == "SCREEN" for entry in journal)


@pytest.mark.parametrize("task_id", ["h4-selection-1p5", "h4-selection-2p0"])
def test_selection_tasks_have_both_passing_and_failing_candidates(tmp_path, task_id, monkeypatch):
    from chemflow.agent.policies import ScriptedPolicy

    decide = ScriptedPolicy.decide

    def with_low_shot_candidate(self, observation):
        proposal = decide(self, observation)
        if self.role == "designer" and "candidates" in proposal:
            proposal["candidates"].append({"template": "sqd", "shots": 1})
        return proposal

    monkeypatch.setattr(ScriptedPolicy, "decide", with_low_shot_candidate)
    results = run(arguments(tmp_path / "out", tasks=task_id))
    directory = tmp_path / "out" / task_id
    summary = json.loads((directory / "summary.json").read_text())
    journal = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    critiques = [entry for entry in journal if entry["state"] == "CRITIQUE"]
    candidates = critiques[-1]["observation"]["candidates"]
    assert {candidate["gate_passed"] for candidate in candidates} == {False, True}
    assert summary["candidate"] == critiques[-1]["decision"]["candidate"]
    assert results["rows"][0]["answer_node"] == summary["candidate"]


def test_fake_good_and_bad_designers_choose_candidate_and_control_answers(tmp_path, monkeypatch):
    class Client:
        def __init__(self, template):
            self.template = template

        def complete(self, messages, tools):
            if messages[-1]["role"] == "tool":
                assert json.loads(messages[-1]["content"])["result"]["data"]["plan_hash"]
                return {"content": "Candidate submitted", "tool_calls": []}
            candidate = ({"template": "sci", "select_cutoff": 0.01}
                         if self.template == "sci" else {"template": "sqd", "shots": 1})
            return {"content": "Candidate", "tool_calls": [{"name": "design.submit",
                "arguments": {"candidates": [candidate]}}]}

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    rows = []
    for template in ("sci", "sqd"):
        client = Client(template)
        monkeypatch.setattr(agent_cli, "client_factory", lambda *args, client=client: client)
        result = run(arguments(tmp_path / template, policy="openai", model="fake",
                               tasks="h4-selection-2p0"))
        rows.append(result["rows"][0])
    assert all(row["reference_correct"] for row in rows)
    assert all(row["correct"] for row in rows)
    assert rows[0]["answer_from_candidate"] and not rows[1]["answer_from_candidate"]
    assert rows[0]["decision_ok"] and not rows[1]["decision_ok"]
