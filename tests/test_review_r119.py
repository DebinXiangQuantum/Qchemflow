"""Reviewer regression tests for R119 (written by Claude; implementers must not edit).

Found by the GLM benchmark with every role on the LLM (2026-10-13, glm-5.3, roles
planner,modeler,designer,critic,analyst, commit be07775). All 17 molecular tasks that reached
the modeler failed with "CF-E080: Expecting value: line 1 column 1 (char 0)".
`Orchestrator.run` parsed the LLM modeler's free-text answer with `json.loads`, and GLM
answered in prose. Every one of those tasks fixes its active space, so the modeler had nothing
to choose, and a crash on its prose cost the whole task.

Contract:
- The modeler's decision is never parsed from free text.
- When the task fixes the active space, or uses the full molecular space, the LLM modeler is
  not asked for one. This covers a declared active space and one derived from lattice,
  spectral or reaction species. No modeler LLM call is made. MODEL is journalled with the
  fixed choice.
- When the task says "auto", the LLM modeler finishes with
  `model.choose {n_elec, n_orb, reason}`. This is a modeler-only tool, and calling it ends the
  modeler's turn. The choice is used if it passes the existing feasibility checks: orbital cap
  and electron/spin parity. If the modeler does not choose, or chooses something infeasible,
  the run uses the diagnostics' suggested active space. It journals MODEL_FALLBACK with a
  non-empty reason, and the CONTROLS entry says `fallback: true`. The run never fails because
  of the modeler's text.
"""
from __future__ import annotations

import json
from argparse import Namespace

import test_review_m3a as m3a
from test_review_r114_118 import Scripted, _turn

from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore


def _auto_run(tmp_path, script):
    store = TaskStore("reviewer-secret")
    client = Scripted(script)
    modeler = LLMPolicy("modeler", client, None, journal=Journal(tmp_path / "policy.jsonl"))
    agent = Orchestrator(store, lambda draft: store.freeze(draft, "reviewer"), tmp_path,
                         max_rounds=1, policies={"modeler": modeler})
    run = agent.run(m3a._spec("auto", {}))
    entries = [json.loads(line) for line in run.journal.read_text().splitlines()]
    return run, entries, client


def _entry(entries, state):
    return next(entry for entry in entries if entry["state"] == state)


def test_r119_a_fixed_active_space_is_not_asked_of_the_modeler(tmp_path, monkeypatch) -> None:
    from chemflow.agent import cli
    from chemflow.eval import runner

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Scripted(
        lambda call, messages: _turn(content="The active space is (2e, 2o); proceed.")))
    out = tmp_path / "out"
    results = runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground",
                                   policy="glm", model=None, roles="modeler", jobs=1, out=out))
    row = results["rows"][0]
    assert row["decision"] != "error" and row["completed"] and row["correct"], row.get("error")
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    assert not [entry for entry in entries
                if entry["state"] == "LLM_CALL" and entry.get("role") == "modeler"]
    assert _entry(entries, "CONTROLS")["observation"]["active"] == {"n_elec": 2, "n_orb": 2}


def test_r119_auto_uses_the_modelers_structured_choice(tmp_path) -> None:
    def script(call, messages):
        if call == 1:
            return _turn([("model.choose", {"n_elec": 6, "n_orb": 6,
                                            "reason": "N2 valence pi/sigma space"})])
        return _turn(content="should never be asked")

    _, entries, client = _auto_run(tmp_path, script)
    assert client.calls == 1
    controls = _entry(entries, "CONTROLS")
    assert controls["observation"]["active"] == {"n_elec": 6, "n_orb": 6}
    assert not controls["observation"].get("fallback")
    assert "MODEL_FALLBACK" not in [entry["state"] for entry in entries]


def test_r119_prose_or_infeasible_choices_fall_back_to_the_diagnostics(tmp_path) -> None:
    for name, script in (
            ("prose", lambda call, messages: _turn(content="I recommend CAS(6,6).")),
            ("odd", lambda call, messages: _turn([("model.choose", {
                "n_elec": 7, "n_orb": 6, "reason": "odd count for a singlet"})]))):
        _, entries, _ = _auto_run(tmp_path / name, script)
        fallback = _entry(entries, "MODEL_FALLBACK")
        assert fallback["reason"].strip(), name
        suggested = _entry(entries, "MODEL")["observation"]["diagnostics"]["suggested_active"]
        controls = _entry(entries, "CONTROLS")
        assert controls["observation"]["fallback"] is True, name
        assert controls["observation"]["active"] == suggested, name
