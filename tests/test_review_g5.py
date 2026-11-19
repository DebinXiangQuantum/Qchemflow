"""Reviewer regression tests for G5, the agent-evaluation harness (written by Claude;
implementers must not edit).

From the ChemGraph comparison (user, 2026-10-13): "5 is used for testing the performance of
chemflow". ChemGraph scores an LLM agent on 13 chemistry tasks by tool sequence and final
number. ChemFlow's harness scores whole agent runs on signed tasks.

Contract:
- `chemflow eval list` names the suites.
- `chemflow eval run --suite <name> --policy scripted|anthropic|openai [--model M] [--roles r,..]
  [--tasks id,..] --out <dir>` signs each task with an ephemeral harness key (approver
  "eval-harness") and runs it through the same code path as `chemflow agent run`, into
  <dir>/<task id>/.
- It writes <dir>/results.json (`rows` per task, plus `aggregate`) and <dir>/report.md.
- Suites live in `chemflow/eval/suites/<name>.yaml`. Each task has `id`, `category`, `spec` (a
  TaskSpec), `expected` {`value_hartree`, optional `relation_at_least`} and `reference_source`,
  which says how the expected value was computed independently of the candidate.
- Correctness uses the accuracy bar the task itself declares (its abs_error_vs_exact or
  l1_error_vs_exact gate), never a tolerance set by the suite. This is the user's rule that
  accuracy bars come from the program or task (A1c).
- Each row has: id, category, completed, correct, gate_passed, relation, relation_ok,
  integrity_events (CF-E060/E120/E130 seen in the journal), decision, wall_s, sim_shots, qpu_s,
  qpu_shots, llm_calls, tool_calls.
- `aggregate` has n, completed, correct, gate_passed and integrity_events totals.
- The scripted policy is deterministic: the same rows apart from times.

Revised by the reviewer after the first implementation scored ChemFlow's exact screening
reference, which every policy computes, instead of the agent's answer:
- `chemflow agent run` writes the critic's chosen `candidate` into summary.json.
- Rows add `answer_node`, `answer_value` and `reference_correct`. `answer_value` is the value of
  the result the agent chose: the critic's candidate, or the recommended control on `stop`.
  `correct` means that answer is within tolerance. `reference_correct` separately checks the
  screening reference, as a pipeline sanity check.
- `expected` may give `decision_in` (acceptable decisions; the row then has `decision_ok`).
  For tasks whose right outcome is a refusal, it gives `outcome: refused` plus `refusal_code`;
  `correct` then means the run was refused with that code before spending.
- The core suite has at least 10 tasks: at least 2 of category `selection` (the candidates
  differ in gate outcome, so choosing matters) and at least 1 refusal task.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")
CHEAP = "h2-ground,h2-atomization"  # the two cheapest core tasks, run in the test


def _suite(name="core"):
    from importlib.resources import files

    import yaml

    return yaml.safe_load(files("chemflow.eval.suites").joinpath(f"{name}.yaml").read_text())


def _eval(*arguments):
    return subprocess.run([CHEMFLOW, "eval", *arguments], check=False, capture_output=True,
                          text=True, timeout=1800)


def test_the_core_suite_is_valid_broad_and_documents_its_references() -> None:
    from chemflow.agent.task import TaskSpec

    suite = _suite()
    tasks = suite["tasks"]
    assert len({task["id"] for task in tasks}) == len(tasks) >= 10
    assert {"molecular", "reaction", "lattice", "spectral", "interaction", "thermo",
            "selection"} <= {task["category"] for task in tasks}
    assert sum(task["category"] == "selection" for task in tasks) >= 2
    assert any(task["expected"].get("outcome") == "refused" for task in tasks)
    assert {item["id"] for item in tasks} >= set(CHEAP.split(","))
    for task in tasks:
        TaskSpec.model_validate(task["spec"])
        assert task["reference_source"]
        assert "tolerance_hartree" not in task["expected"]  # the bar is the task's own gate
        if task["expected"].get("outcome") != "refused":
            assert any(gate["kind"] in ("abs_error_vs_exact", "l1_error_vs_exact")
                       for gate in task["spec"]["gates"])
    assert "core" in _eval("list").stdout


def test_expected_values_agree_with_the_reviewers_independent_pyscf_fci() -> None:
    from pyscf import fci, gto, scf

    expected = {task["id"]: task["expected"]["value_hartree"] for task in _suite()["tasks"]}
    h2 = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)).run()
    e_h2 = fci.FCI(h2).kernel()[0]
    h = scf.UHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)).run()
    assert expected["h2-ground"] == pytest.approx(e_h2, abs=1e-8)
    assert expected["h2-atomization"] == pytest.approx(2 * h.e_tot - e_h2, abs=1e-8)


def _run(out):
    completed = _eval("run", "--suite", "core", "--policy", "scripted", "--tasks", CHEAP,
                      "--out", str(out))
    assert completed.returncode == 0, completed.stderr
    return json.loads((out / "results.json").read_text())


def test_scripted_runs_are_scored_and_reported(tmp_path) -> None:
    results = _run(tmp_path / "a")
    rows = {row["id"]: row for row in results["rows"]}
    assert set(rows) == set(CHEAP.split(","))
    for row in rows.values():
        assert row["completed"] and row["correct"] and row["reference_correct"]
        assert row["answer_node"] and row["gate_passed"]
        assert row["relation_ok"] and row["integrity_events"] == []
        assert row["llm_calls"] == 0 and row["tool_calls"] >= 1 and row["wall_s"] > 0
        assert (tmp_path / "a" / row["id"] / "summary.json").exists()
    assert results["aggregate"]["n"] == 2 and results["aggregate"]["correct"] == 2
    assert "h2-atomization" in (tmp_path / "a" / "report.md").read_text()


def test_the_scripted_baseline_is_deterministic(tmp_path) -> None:
    def stable(results):
        return [{key: value for key, value in row.items() if not key.endswith("_s")}
                for row in sorted(results["rows"], key=lambda row: row["id"])]

    assert stable(_run(tmp_path / "a")) == stable(_run(tmp_path / "b"))


def test_a_wrong_expected_value_is_scored_incorrect(tmp_path, monkeypatch) -> None:
    """The score must come from comparing values, not from the run merely finishing."""
    import yaml

    suite = _suite()
    for task in suite["tasks"]:
        if task["id"] == "h2-ground":
            task["expected"]["value_hartree"] += 1.0
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(suite))
    completed = _eval("run", "--suite-file", str(path), "--policy", "scripted", "--tasks",
                      "h2-ground", "--out", str(tmp_path / "out"))
    assert completed.returncode == 0, completed.stderr
    row = json.loads((tmp_path / "out" / "results.json").read_text())["rows"][0]
    assert row["completed"] and not row["correct"]


def test_correctness_is_the_agents_answer_not_the_screening_reference() -> None:
    from chemflow.eval.runner import EvaluationTask, score_task

    task = EvaluationTask.model_validate(next(item for item in _suite()["tasks"]
                                              if item["id"] == "h2-ground"))
    exact = task.expected.value_hartree
    summary = {"decision": "stop", "candidate": "n4", "spent": {"wall_s": 0.1},
               "results": [
                   {"node": "n4", "value": f"{exact + 0.0205} hartree", "relation":
                    "variational_upper_bound", "gate_verdict": "FAIL (threshold exceeded)"},
                   {"node": "n8", "value": f"{exact} hartree", "relation": "equivalent",
                    "gate_verdict": "PASS", "screening_reference": True}]}
    row = score_task(task, summary, [])
    assert row["answer_node"] == "n4" and row["reference_correct"]
    assert not row["correct"]  # the agent chose a 20 mHa answer; the reference does not save it


def test_a_refusal_task_is_correct_only_when_refused_before_spending(tmp_path) -> None:
    refusal = next(task for task in _suite()["tasks"]
                   if task["expected"].get("outcome") == "refused")
    completed = _eval("run", "--suite", "core", "--policy", "scripted", "--tasks", refusal["id"],
                      "--out", str(tmp_path / "out"))
    assert completed.returncode == 0, completed.stderr
    row = json.loads((tmp_path / "out" / "results.json").read_text())["rows"][0]
    assert row["correct"] and row["decision"] in ("refused", "failed")
    assert refusal["expected"]["refusal_code"] in json.dumps(row)
