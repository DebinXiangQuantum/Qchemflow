"""Reviewer regression tests for R126-R128 (written by Claude; implementers must not edit).

These come from the open questions of the GLM benchmark (docs/g5-glm-benchmark.md). The user
approved all three on 2026-10-14.

- R126, verified declines. `refuse-sim-shots` asks whether the standard 1024 + 2048-shot SQD
  batch fits a 2048-shot budget. An LLM designer could decline only with free text, which the
  harness cannot check. Contract:
  - `design.decline` takes an optional `plan_id`, the plan the designer refuses to run.
  - ChemFlow forecasts that plan against the remaining task budget without executing
    anything, using the executor's own preflight.
  - If the forecast refuses, the run ends as a budget refusal: CF-E070, `nothing_consumed`
    true, the same outcome the scripted path produces when it submits the batch. The refusal
    names the plan.
  - If the plan fits, the claim is false. The tool returns a structured error, and the
    designer's turn continues.
  - A decline without `plan_id` keeps R117's meaning: no candidate, and the run continues
    on its controls.
- R127, relation requirements in the signed task. Every suite task kept its required relation
  in `expected.relation_at_least`, which the agent never sees. By the A1c rule the bar comes
  from the task. Contract:
  - Each suite task's spec carries a `relation_at_least` gate with the former requirement.
  - `Expected` no longer accepts `relation_at_least`.
  - A row's `relation_ok` checks the answer against the task's own relation gates.
  - The scripted policy still answers correctly when the requirement is visible.
- R128, turn-limit reminder. glm-5.3 ran out of its 24 turns in 7 composition tasks without
  submitting. Contract:
  - For a role with terminal tools (designer: design.submit/decline; modeler: model.choose),
    the model is told how many turns remain before each of its last two calls. The message
    names the terminal tools.
  - Roles without terminal tools get no such message.
"""
from __future__ import annotations

import json

import pytest
from test_review_r114_118 import Scripted

BATCH = [{"template": "sqd", "select_cutoff": None, "shots": 1024},
         {"template": "sqd", "select_cutoff": None, "shots": 2048}]


def _eval_full(tmp_path, monkeypatch, tasks, script=None, policy="glm"):
    from argparse import Namespace

    from chemflow.agent import cli
    from chemflow.eval import runner

    if policy == "glm":
        monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
        monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")
        monkeypatch.setattr(cli, "client_factory",
                            lambda provider, role, model: Scripted(script))
    out = tmp_path / "out"
    results = runner.run(Namespace(suite="full", suite_file=None, tasks=tasks, policy=policy,
                                   model=None, roles="designer", jobs=1, out=out))

    def journal(task):
        return [json.loads(line) for line in
                (out / task / "journal.jsonl").read_text().splitlines()]

    return {row["id"]: row for row in results["rows"]}, journal


# R126 ----------------------------------------------------------------------------------------


# R127 ----------------------------------------------------------------------------------------

def _suite(name):
    from importlib.resources import files

    import yaml

    return yaml.safe_load(files("chemflow.eval.suites").joinpath(f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", ["core", "full"])
def test_r127_relation_requirements_are_task_gates(name) -> None:
    for task in _suite(name)["tasks"]:
        relations = [gate for gate in task["spec"]["gates"]
                     if gate["kind"] == "relation_at_least"]
        assert len(relations) == 1 and relations[0]["relation"], task["id"]
        assert "relation_at_least" not in task["expected"], task["id"]


def test_r127_expected_no_longer_carries_a_relation() -> None:
    from pydantic import ValidationError

    from chemflow.eval.runner import Expected

    with pytest.raises(ValidationError):
        Expected.model_validate({"value_hartree": -1.0,
                                 "relation_at_least": "variational_upper_bound"})


def test_r127_relation_ok_is_scored_against_the_task_gates() -> None:
    from chemflow.eval.runner import EvaluationTask, score_task

    task = EvaluationTask.model_validate(next(item for item in _suite("core")["tasks"]
                                              if item["id"] == "h2-ground"))
    required = next(gate.relation for gate in task.spec.gates
                    if gate.kind == "relation_at_least")
    assert required == "variational_upper_bound"
    exact = task.expected.value_hartree

    def row(relation):
        return score_task(task, {"decision": "stop", "candidate": "n6", "spent": {}, "results": [
            {"node": "n6", "value": f"{exact} hartree", "relation": relation,
             "gate_passed": True}]}, [])

    assert row("variational_upper_bound")["relation_ok"] is True
    assert row("approximate")["relation_ok"] is False


def test_r127_scripted_answers_meet_the_visible_relation_gate(tmp_path, monkeypatch) -> None:
    rows, _ = _eval_full(tmp_path, monkeypatch, "water-valence,lih-stretched", policy="scripted")
    for task, row in rows.items():
        assert row["correct"] and row["relation_ok"], (task, row["answer_node"], row["relation"])


# R128 ----------------------------------------------------------------------------------------

def _reminders(messages):
    return [message["content"] for message in messages
            if message.get("role") == "user" and "turn" in str(message.get("content")).lower()
            and ("design.submit" in str(message.get("content")) or
                 "model.choose" in str(message.get("content")))]
