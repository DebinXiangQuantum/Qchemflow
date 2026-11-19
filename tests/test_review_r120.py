"""Reviewer regression tests for R120 (written by Claude; implementers must not edit).

A live glm-5.3 probe after R114-R118 (5 full-suite tasks, 2026-10-13) went better. Three
designs were screened, and the agent's answer came from GLM's candidate in all three. The
other two were refused for a reason the model could have fixed. GLM composed new plans with
plan.new instead of extending base_plan_id. It hit the 12-turn limit while still reading cards
and the registry, and never learned why its plan was unusable. The orchestrator refused the
plans later, with "Designer changed frozen controls, references or gates".

Contract:
- `design.submit` checks that the plan extends the run's base plan: the base nodes as a
  prefix, the same references, no gates, and at least one new target. If it does not, the
  model gets a structured error naming observation.base_plan_id, and its turn continues.
  The orchestrator's own check stays in place.
- The designer system prompt says that candidates must be built on base_plan_id, and that
  plans started with plan.new are refused.
- `LLMPolicy` defaults to `max_turns` of at least 20. `chemflow agent run` and
  `chemflow eval run` take `--max-turns N` and pass it to every LLM role. The harness records
  `max_turns` in results.json metadata. The Namespace attribute is optional; when absent, the
  default applies.
"""
from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import test_review_m4a as m4a
from test_review_r114_118 import (
    Scripted,
    _base,
    _designer_calls_before_screening,
    _eval,
    _results,
    _screened,
    _turn,
)

from chemflow.agent.llm import LLMPolicy

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")


def test_r120_submit_rejects_plans_not_built_on_the_base_and_the_turn_continues(
        tmp_path, monkeypatch) -> None:
    def script(call, messages):
        if call == 1:
            return _turn([("plan.new", {"identifier": "scratch"})])
        if call == 2:
            scratch = [data for name, data in _results(messages) if name == "plan.new"][-1]
            return _turn([("design.submit", {"plan_id": scratch["plan_id"]})])
        if call == 3:
            return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        if call == 4:
            built = [data for name, data in _results(messages) if name == "plan.candidates"][-1]
            return _turn([("design.submit", {"plan_id": built["plan_id"]})])
        return _turn(content="should never be asked")

    _, rows, journal, clients = _eval(tmp_path, monkeypatch, "h2-ground", script)
    entries = journal("h2-ground")
    assert _designer_calls_before_screening(entries) == 4
    submits = [entry["decision"] for entry in entries if entry["state"] == "LLM_TOOL" and
               entry["decision"].get("name") == "design.submit"]
    assert "error" in submits[0] and "base_plan_id" in json.dumps(submits[0]["error"])
    assert "result" in submits[1]
    assert len(_screened(entries)) == 1
    assert rows["h2-ground"]["design"] == "screened"
    system = clients[0].messages[0][0]["content"]
    assert "plan.new" in system and "base_plan_id" in system


def test_r120_turn_budget_default_and_cli_option(tmp_path, monkeypatch) -> None:
    assert LLMPolicy("designer", Scripted(lambda call, messages: _turn()),
                     m4a._tools()).max_turns >= 20
    from chemflow.agent import cli
    from chemflow.eval import runner

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Scripted(
        lambda call, messages: _turn([("cards.search", {"query": "sqd"})])))
    out = tmp_path / "out"
    results = runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground",
                                   policy="glm", model=None, roles="designer", jobs=1,
                                   out=out, max_turns=3))
    assert results["metadata"]["max_turns"] == 3
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    assert _designer_calls_before_screening(entries) == 3
    for command in ("agent", "eval"):
        completed = subprocess.run([CHEMFLOW, command, "run", "--help"],
                                   capture_output=True, text=True, check=True)
        assert "--max-turns" in completed.stdout, command
