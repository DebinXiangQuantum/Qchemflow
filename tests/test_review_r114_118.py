"""Reviewer regression tests for R114-R118 (written by Claude; implementers must not edit).

Found by the 55-task GLM benchmark (2026-10-13, commit be07775). GLM's designs were refused in
44 of 55 tasks for glm-5.3 and 41 for glm-4.7; the scripted designer screened candidates in 29.
The journals show why:
- R114: unanswered rows (refusal tasks) were scored `relation_ok: false`, so the relation total
  mixed "wrong relation" with "nothing to rate".
- R115: every refusal reason in the journal was empty. The design path ended in StopIteration
  when no plan had been built, and `str(StopIteration())` is "". Two tasks failed outright:
  a ChemFlow error while screening (CF-E120: "Plan gates forbidden or targets missing" in
  h5-doublet, "Spectral target identity mismatch" in impurity-asymmetric) escaped the design
  path's `except`, because AdapterError is a CheckError, not a ValueError.
- R116: all 170 of GLM's `plan.add_node` calls failed validation. The model sent the nested
  `node` object as a JSON string, as OpenAI-compatible models often do when a schema uses
  `$ref`/`$defs`.
- R117: nothing told the designer how to finish. It explored cards and the registry, ran
  `algorithm.verify` on targets it cannot handle (148 errors), rebuilt plans from scratch with
  `plan.new`, and ended by text reply (16-26 tasks), three failing turns in a row (17-23) or
  the turn limit (9-13). The orchestrator then screened whatever `plan.candidates` or
  `plan.add_node` had returned last, or nothing.
- R118: `plan.candidates` crashed instead of explaining: `select_cutoff: null` raised
  KeyError('select_cutoff'), a plan without a reference raised IndexError, an interaction
  plan raised KeyError('hamiltonian'). `algorithm.register` with an unknown id raised a bare
  KeyError of the id.

Contract:
- R114: a row without an answer (`answer_node` null) has `relation_ok: null`. `aggregate`
  counts `relation_ok` over true values and reports `relation_rated`, the number of rows with a
  non-null `relation_ok`.
- R115: any failure of the candidate path refuses the design, never the task: the existing
  exception types plus every CheckError (so every AdapterError). Programming errors outside
  these types still propagate. DESIGN_REFUSED always carries a non-empty reason; when nothing
  was built, it names that and how the designer's turn ended.
- R116: tool parameter schemas are self-contained (no `$ref`, no `$defs`). When a tool
  argument whose schema is an object or array arrives as a string holding valid JSON of that
  type, it is decoded once before validation. The journal keeps the call as the model sent it.
- R117: the designer finishes explicitly. `design.submit {plan_id}` names the plan to screen.
  `design.decline {reason}` declines to propose, with the model's reason. Both are designer
  tools, and either one ends the designer's turn at once. Without either, the orchestrator
  screens the plan from the latest successful plan-building call: plan.candidates,
  plan.add_node, plan.refine, plan.bind or plan.remove_node. A decline is journalled as
  DESIGN_DECLINED with the reason, and the run continues on its controls. The designer's system
  prompt states this protocol. Its context stays the JSON in the second message, with
  `observation.base_plan_id`.
- R117 (harness): rows get `design`: "screened" (a SCREEN happened), "declined", "refused" or
  null. They also get `answer_from_candidate`, true when the answer node is a screened
  candidate. `aggregate` counts `design_screened`, `design_declined`, `design_refused` and
  `answer_from_candidate`.
- R118: null optional candidate fields take the template defaults. Unsupported plans are
  refused with a message that names the reason: a missing reference, or the reference
  operator. They are never refused with KeyError, IndexError or a wrapped one.
  `algorithm.verify` and `algorithm.register` appear in a role's schemas only for tasks they can
  verify (impurity-model targets). An unknown report id is refused with a message that points
  to `algorithm.verify`.
"""
from __future__ import annotations

import json
from argparse import Namespace

FINAL = "This candidate is ready."


def _turn(calls=(), content=""):
    calls = [{"id": f"c{index}", "name": name, "arguments": arguments}
             for index, (name, arguments) in enumerate(calls)]
    return {"tool_calls": calls, "content": content,
            "stop_reason": "tool_calls" if calls else "stop", "usage": {}}


def _base(messages):
    return json.loads(messages[1]["content"])["observation"]["base_plan_id"]


def _results(messages):
    """(tool name, data) for every successful tool result the model has seen."""
    seen = []
    for message in messages:
        if message.get("role") == "tool":
            output = json.loads(message["content"])
            if "result" in output:
                seen.append((output["name"], output["result"]["data"]))
    return seen


class Scripted:
    """A fake GLM client; `script(call_number, messages)` returns each response."""

    model = "fake-glm"

    def __init__(self, script):
        self.script, self.calls, self.messages = script, 0, []

    def complete(self, messages, tools):
        self.calls += 1
        self.messages.append(list(messages))
        return self.script(self.calls, messages)


def _eval(tmp_path, monkeypatch, tasks, script=None, policy="glm"):
    from chemflow.agent import cli
    from chemflow.eval import runner

    clients = []
    if policy == "glm":
        monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
        monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")

        def factory(provider, role, model):
            clients.append(Scripted(script))
            return clients[-1]

        monkeypatch.setattr(cli, "client_factory", factory)
    out = tmp_path / "out"
    results = runner.run(Namespace(suite="core", suite_file=None, tasks=tasks, policy=policy,
                                   model=None, roles="designer", jobs=1, out=out))
    rows = {row["id"]: row for row in results["rows"]}

    def journal(task):
        return [json.loads(line) for line in
                (out / task / "journal.jsonl").read_text().splitlines()]

    return results, rows, journal, clients


def _designer_calls_before_screening(entries):
    calls = 0
    for entry in entries:
        if entry["state"] == "PROPOSE":
            return calls
        calls += entry["state"] == "LLM_CALL" and entry.get("role") == "designer"
    return calls


def _screened(entries):
    critique = next(entry for entry in entries if entry["state"] == "CRITIQUE")
    return critique["observation"]["candidates"]


# R114 ----------------------------------------------------------------------------------------

def test_r114_rows_without_an_answer_have_no_relation_verdict(tmp_path, monkeypatch) -> None:
    results, rows, _, _ = _eval(tmp_path, monkeypatch, "h2-ground,h2-budget-refusal",
                                policy="scripted")
    assert rows["h2-budget-refusal"]["correct"]
    assert rows["h2-budget-refusal"]["relation_ok"] is None
    assert rows["h2-ground"]["relation_ok"] is True
    assert results["aggregate"]["relation_ok"] == 1
    assert results["aggregate"]["relation_rated"] == 1


# R115 ----------------------------------------------------------------------------------------

def test_r115_a_designer_that_builds_nothing_is_refused_with_a_reason(tmp_path,
                                                                      monkeypatch) -> None:
    _, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground",
                                lambda call, messages: _turn(content="I would try SQD."))
    row = rows["h2-ground"]
    assert row["completed"] and row["correct"] and row["design"] == "refused"
    refused = [entry for entry in journal("h2-ground") if entry["state"] == "DESIGN_REFUSED"]
    assert refused and all(entry["reason"].strip() for entry in refused)


# R116 ----------------------------------------------------------------------------------------


# R117 ----------------------------------------------------------------------------------------


# R118 ----------------------------------------------------------------------------------------

def _base_plan(tools):
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    return empty, tools.call("modeler", "plan.base", {
        "plan_id": empty, "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]
