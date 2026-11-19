from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from chemflow.agent import cli as agent_cli
from chemflow.agent.task import TaskRecord
from chemflow.cli import main
from chemflow.eval.runner import (
    Suite,
    _spectral_values,
    aggregate_rows,
    load_suite,
    run,
    score_task,
    select_tasks,
)
from chemflow.signing import TaskVerifier
from chemflow.units import Q


def arguments(out, **overrides):
    return argparse.Namespace(**{"suite": "core", "suite_file": None, "policy": "scripted",
                                 "model": None, "roles": "designer", "tasks": "h2-ground",
                                 "out": out, **overrides})


@pytest.mark.parametrize("field,value", [
    ("id", "../escape"), ("id", "public.pem"), ("category", "unknown"),
    ("reference_source", ""), ("reference_source", "   "), ("spec", {}),
    ("expected", {"value_hartree": float("nan")}),
    ("expected", {"value_hartree": -1, "tolerance_hartree": 0}),
    ("expected", {"value_hartree": -1, "tolerance_hartree": -1}),
    ("expected", {"value_hartree": -1, "tolerance_hartree": float("inf")}),
    ("expected", {"value_hartree": -1,
                  "relation_at_least": "exact"}),
])
def test_invalid_suite_task(field, value):
    data = load_suite("core").model_dump(mode="json")
    data["tasks"][0][field] = value
    with pytest.raises(ValidationError):
        Suite.model_validate(data)


def test_suite_uniqueness_empty_extra_and_loading(tmp_path):
    data = load_suite("core").model_dump(mode="json")
    for tasks in ([], [data["tasks"][0], data["tasks"][0]]):
        with pytest.raises(ValidationError):
            Suite.model_validate({**data, "tasks": tasks})
    with pytest.raises(ValidationError):
        Suite.model_validate({**data, "typo": True})
    for name, path in ((None, None), ("core", tmp_path / "suite.yaml"), ("../core", None)):
        with pytest.raises(ValueError):
            load_suite(name, path)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_suite(path=path) == load_suite("core")
    path.write_text("tasks: [")
    with pytest.raises(yaml.YAMLError):
        load_suite(path=path)


@pytest.mark.parametrize("selection", ["not-real", "h2-ground,not-real", "", "h2-ground,",
                                       "h2-ground,h2-ground"])
def test_unknown_or_duplicate_selection_leaves_no_artifacts(tmp_path, selection):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="task ids"):
        run(arguments(out, tasks=selection))
    assert not out.exists()


def test_selection_preserves_suite_order():
    suite = load_suite("core")
    assert select_tasks(suite, None) == suite.tasks
    assert [task.id for task in select_tasks(suite, "h2-atomization, h2-ground")] == [
        "h2-ground", "h2-atomization"]


def synthetic():
    task = load_suite("core").tasks[0]
    summary = {"decision": "stop", "candidate": "chosen",
               "spent": {"wall_s": 2.5, "sim_shots": 400,
                                             "qpu_s": 1.25, "qpu_shots": 100},
               "results": [{"node": "bad", "value": "900 hartree", "relation": "approximate",
                            "gate_verdict": "FAIL"},
                           {"node": "chosen", "value": f"{task.expected.value_hartree} hartree",
                            "relation": "equivalent", "gate_verdict": "PASS"},
                           {"node": "reference", "value": f"{task.expected.value_hartree} hartree",
                            "relation": "equivalent", "gate_verdict": "PASS",
                            "screening_reference": True}]}
    journal = [{"state": "REFERENCE", "tool_calls": [{"name": "plan.base"}]},
               {"state": "LLM_CALL", "tool_calls": [],
                "decision": {"tool_calls": [{"name": "not executed"}]}},
               {"state": "LLM_TOOL", "tool_calls": [{"name": "check"}, {"name": "run"}]},
               {"state": "LLM_ERROR", "decision": {"message": "parse failure"}}]
    return task, summary, journal


def test_every_scoring_field_and_aggregate():
    task, summary, journal = synthetic()
    row = score_task(task, summary, journal)
    assert row == {"id": "h2-ground", "category": "molecular", "completed": True,
                   "correct": True, "gate_passed": True, "relation": "equivalent",
                   "relation_ok": True, "integrity_events": [], "decision": "stop",
                   "wall_s": 2.5, "sim_shots": 400, "qpu_s": 1.25, "qpu_shots": 100,
                   "llm_calls": 2, "tool_calls": 3, "answer_node": "chosen",
                   "answer_value": task.expected.value_hartree, "reference_correct": True,
                   "decision_ok": True, "refusal_code": None, "llm_failed": False, "valid": True,
                   "design": None, "answer_from_candidate": False}
    assert aggregate_rows([row, row]) == {
        "n": 2, "completed": 2, "correct": 2, "gate_passed": 2, "relation_ok": 2,
        "integrity_events": 0, "wall_s": 5, "sim_shots": 800, "qpu_s": 2.5,
        "qpu_shots": 200, "llm_calls": 4, "tool_calls": 6,
        "reference_correct": 2, "decision_ok": 2, "invalid": 0, "correct_valid": 2,
        "errors": 0, "not_run": 0, "relation_rated": 2, "design_screened": 0, "design_declined": 0,
        "design_refused": 0, "answer_from_candidate": 0}


def test_correctness_gates_relation_and_integrity_are_independent():
    task, summary, journal = synthetic()
    reference = summary["results"][1]
    reference.update(gate_verdict="FAIL (test)", relation="approximate")
    row = score_task(task, summary, journal)
    assert row["correct"] and row["completed"]
    assert not row["gate_passed"] and not row["relation_ok"]
    reference["value"] = "0 hartree"
    assert not score_task(task, summary, journal)["correct"]
    task = task.model_copy(update={"spec": task.spec.model_copy(
        update={"gates": [gate for gate in task.spec.gates
                          if gate.kind != "relation_at_least"]})})
    assert score_task(task, summary, journal)["relation_ok"]


@pytest.mark.parametrize("value", ["nan hartree", "inf hartree", "invalid", "1 unknown"])
def test_unusable_headline_is_not_completed(value):
    task, summary, journal = synthetic()
    summary["results"][1]["value"] = value
    row = score_task(task, summary, journal)
    assert not row["completed"] and not row["correct"]


def test_unit_conversion_and_tolerance_boundary():
    task, summary, journal = synthetic()
    reference = summary["results"][1]
    reference["value"] = f"{Q(task.expected.value_hartree, 'hartree').to('eV').value} eV"
    assert score_task(task, summary, journal)["correct"]
    for multiple, correct in ((0.99, True), (1.01, False)):
        value = task.expected.value_hartree + multiple * task.accuracy_threshold
        reference["value"] = f"{value} hartree"
        assert score_task(task, summary, journal)["correct"] is correct


def test_integrity_events_count_each_code_once_per_entry():
    task, summary, journal = synthetic()
    journal.extend([
        {"observation": {"diagnostics": ["CF-E060", "CF-E120", "CF-E130"]},
         "decision": {"error": "CF-E120 repeated in same event"}},
        {"reason": "CF-E120 in another entry"},
        {"reason": "CF-E011 CF-E0600 XCF-E120"}])
    row = score_task(task, summary, journal)
    assert row["integrity_events"] == ["CF-E060", "CF-E120", "CF-E130", "CF-E120"]
    assert row["correct"]
    assert aggregate_rows([row])["integrity_events"] == 4


def test_failed_or_ambiguous_runs_and_unknown_resources():
    task, summary, journal = synthetic()
    failed = score_task(task, None, journal)
    assert not failed["completed"] and not failed["correct"] and not failed["gate_passed"]
    assert failed["relation"] is None and not failed["relation_ok"]
    assert failed["decision"] == "failed" and failed["wall_s"] is None
    assert aggregate_rows([failed])["wall_s"] is None
    summary["candidate"] = None
    assert not score_task(task, summary, journal)["completed"]
    assert score_task(task, summary, journal)["reference_correct"]
    summary["results"] = [summary["results"][1]]
    assert not score_task(task, summary, journal)["correct"]
    summary["candidate"] = "chosen"
    assert score_task(task, summary, journal)["correct"]
    assert not score_task(task, summary, journal)["reference_correct"]
    summary["spent"]["qpu_s"] = None
    assert aggregate_rows([score_task(task, summary, journal)])["qpu_s"] is None


def test_spectral_scalar_is_record_bound_and_not_parsed_as_energy(tmp_path):
    task = next(task for task in load_suite("core").tasks if task.category == "spectral")
    summary = {"record_hash": "signed", "candidate": "chosen",
               "results": [{"screening_reference": True, "node": "reference",
               "result_id": "reference", "value": "spectral density (101 points)",
               "relation": "equivalent", "gate_verdict": "PASS"},
               {"node": "chosen", "result_id": "chosen", "value": "spectral density (101 points)",
                "relation": "approximate", "gate_verdict": "FAIL"}]}
    with sqlite3.connect(tmp_path / "results.sqlite") as connection:
        connection.execute("CREATE TABLE results (id TEXT, record_hash TEXT, node_id TEXT, "
                           "value REAL, unit TEXT)")
        connection.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?)", [
            ("reference", "signed", "reference", task.expected.value_hartree, "dimensionless"),
            ("chosen", "signed", "chosen", 0.5, "dimensionless")])
    scalars = _spectral_values(tmp_path, summary)
    row = score_task(task, summary, [], spectral_values=scalars)
    assert row["reference_correct"] and not row["correct"] and row["answer_value"] == 0.5
    assert not score_task(task, summary, [])["completed"]
    summary["record_hash"] = "different task"
    assert _spectral_values(tmp_path, summary) == {}


def test_cli_filtering_signing_and_shared_run_path(tmp_path, monkeypatch):
    observed = []
    original = agent_cli.run

    def invoke(args):
        observed.append(args)
        original(args)

    monkeypatch.setattr(agent_cli, "run", invoke)
    monkeypatch.setattr(sys, "argv", ["chemflow", "eval", "run", "--suite", "core",
        "--policy", "scripted", "--tasks", "h2-ground", "--out", str(tmp_path / "out")])
    main()
    assert len(observed) == 1
    out = tmp_path / "out"
    verifier = TaskVerifier((out / "public.pem").read_bytes())
    record = TaskRecord.model_validate_json((out / "h2-ground/task.json").read_text())
    verifier.verify(record)
    assert record.approver == "eval-harness"
    assert {path.name for path in out.iterdir() if path.is_dir()} == {"h2-ground"}
    assert [path.name for path in out.rglob("*.pem")] == ["public.pem"]
    for path in out.rglob("*"):
        if path.is_file():
            assert b"PRIVATE KEY" not in path.read_bytes()
    results = json.loads((out / "results.json").read_text())
    assert results["aggregate"]["correct"] == 1
    report = (out / "report.md").read_text()
    assert results["metadata"]["git_commit"] in report
    assert "scripted" in report and "Aggregate" in report
    with pytest.raises(ValueError, match="empty"):
        run(arguments(out))


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_fake_llm_through_harness(tmp_path, monkeypatch, provider):
    factory_calls = []
    clients = []

    class Client:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if messages[-1]["role"] == "tool":
                assert json.loads(messages[-1]["content"])["result"]["data"]["plan_id"]
                return {"content": "SCI plan ready", "tool_calls": []}
            context = json.loads(messages[-1]["content"])
            assert "plan.candidates" in {tool["name"] for tool in tools}
            return {"content": "Screen SCI", "tool_calls": [{"name": "plan.candidates",
                "arguments": {"plan_id": context["observation"]["base_plan_id"],
                              "candidates": [{"template": "sci", "select_cutoff": 0.01}]}}]}

    def factory(provider, role, model):
        factory_calls.append((provider, role, model))
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr(agent_cli, "client_factory", factory)
    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-a-key")
    results = run(arguments(tmp_path / "out", policy=provider, model="fake-model"))
    assert factory_calls == [(provider, "designer", "fake-model")]
    row = results["rows"][0]
    summary = json.loads((tmp_path / "out/h2-ground/summary.json").read_text())
    chosen = next(item for item in summary["results"] if item["node"] == summary["candidate"])
    task = load_suite("core").tasks[0]
    number, unit = chosen["value"].split()
    value = Q(float(number), unit).to("hartree").value
    assert row["completed"] and row["gate_passed"] and row["reference_correct"]
    assert row["answer_node"] == summary["candidate"] and row["answer_value"] == value
    assert row["correct"] == (abs(value - task.expected.value_hartree) <= task.accuracy_threshold)
    assert row["relation"] == chosen["relation"]
    assert row["relation_ok"] == (chosen["relation"] in ("equivalent", "variational_upper_bound"))
    assert row["llm_calls"] == clients[0].calls > 0
    assert row["tool_calls"] > 0 and row["integrity_events"] == []
    assert results["metadata"]["policy"] == provider
    assert "fake-model" in (tmp_path / "out/report.md").read_text()


def test_live_policy_requires_opt_in_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("CHEMFLOW_LIVE_LLM", raising=False)
    with pytest.raises(ValueError, match="CHEMFLOW_LIVE_LLM"):
        run(arguments(tmp_path / "out", policy="openai", model="fake"))
    assert not (tmp_path / "out").exists()


def test_task_failure_is_scored_and_next_task_runs(tmp_path, monkeypatch):
    original = agent_cli.run

    def invoke(args):
        if args.out.name == "h2-ground":
            raise SystemExit(2)
        original(args)

    monkeypatch.setattr(agent_cli, "run", invoke)
    results = run(arguments(tmp_path / "out", tasks="h2-ground,h2-stretched"))
    assert results["aggregate"]["n"] == 2 and results["aggregate"]["completed"] == 1
    assert results["rows"][0]["decision"] == "failed"
    assert results["rows"][1]["correct"]


def test_wrong_file_reference_does_not_change_signed_spec(tmp_path):
    data = load_suite("core").model_dump(mode="json")
    spec = deepcopy(data["tasks"][0]["spec"])
    data["tasks"][0]["expected"]["value_hartree"] += 1
    suite_file = tmp_path / "wrong.yaml"
    suite_file.write_text(yaml.safe_dump(data))
    results = run(arguments(tmp_path / "out", suite=None, suite_file=suite_file))
    assert results["rows"][0]["completed"] and not results["rows"][0]["correct"]
    record = TaskRecord.model_validate_json((tmp_path / "out/h2-ground/task.json").read_text())
    assert record.spec.model_dump(mode="json") == spec
