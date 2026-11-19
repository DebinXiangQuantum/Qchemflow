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

import pytest
import test_review_a1a as a1a
import test_review_a1b as a1b
import test_review_m4a as m4a

from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy

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


def test_r115_chemflow_errors_while_screening_refuse_the_design_not_the_task(
        tmp_path, monkeypatch) -> None:
    from chemflow.diagnostics import AdapterError
    from chemflow.runtime.executor import Executor

    original = Executor.run_plan

    def run_plan(self, plan, *args, **kwargs):
        if any(node.refinement == "sci" for node in plan.nodes):
            raise AdapterError("CF-E120", None, "Plan target identity mismatch", "test")
        return original(self, plan, *args, **kwargs)

    monkeypatch.setattr(Executor, "run_plan", run_plan)

    def script(call, messages):
        if call == 1:
            return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        built = [data for name, data in _results(messages) if name == "plan.candidates"]
        return _turn([("design.submit", {"plan_id": built[-1]["plan_id"]})])

    _, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    row = rows["h2-ground"]
    assert row["decision"] != "error" and row["completed"] and row["correct"]
    refused = [entry for entry in journal("h2-ground") if entry["state"] == "DESIGN_REFUSED"]
    assert refused and "CF-E120" in refused[0]["reason"]


# R116 ----------------------------------------------------------------------------------------

def test_r116_tool_parameter_schemas_are_self_contained() -> None:
    from chemflow.agent.tools import ROLES

    tools = m4a._tools()
    for role in ROLES:
        for schema in tools.schemas(role):
            text = json.dumps(schema["parameters"])
            assert "$ref" not in text and "$defs" not in text, (role, schema["name"])


def test_r116_json_encoded_object_arguments_are_decoded(tmp_path) -> None:
    tools = m4a._tools()
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty,
                                               "active": {"n_elec": 2, "n_orb": 2}})["data"]
    node = {"id": "extra0", "op": "chemflow.molecule", "inputs": [],
            "params": [{"key": "atoms", "value_json": "\"H 0 0 0\""},
                       {"key": "spin", "value_json": "1"},
                       {"key": "charge", "value_json": "0"}],
            "effects": "pure", "refinement": None, "binding": None}
    candidates = [{"template": "sci", "select_cutoff": 0.01, "shots": None}]
    client = Scripted(lambda call, messages: [
        _turn([("plan.add_node", {"plan_id": empty, "node": json.dumps(node),
                                  "target": False}),
               ("plan.candidates", {"plan_id": base["plan_id"],
                                    "candidates": json.dumps(candidates)})]),
        _turn(content=FINAL)][call - 1])
    result = LLMPolicy("designer", client, tools,
                       journal=Journal(tmp_path / "j.jsonl")).decide({})
    outputs = {output["name"]: output for output in result["outputs"]}
    assert "result" in outputs["plan.add_node"], outputs["plan.add_node"]
    assert "result" in outputs["plan.candidates"], outputs["plan.candidates"]
    assert outputs["plan.candidates"]["result"]["data"]["ids"]
    entries = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines()]
    sent = next(entry for entry in entries if entry["state"] == "LLM_TOOL")["calls"][0]
    assert isinstance(sent["arguments"]["node"], str)  # journalled as the model sent it


# R117 ----------------------------------------------------------------------------------------

def test_r117_submit_selects_the_plan_and_ends_the_designer(tmp_path, monkeypatch) -> None:
    def script(call, messages):
        base = _base(messages)
        if call == 1:
            return _turn([("plan.candidates", {"plan_id": base, "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        if call == 2:
            return _turn([("plan.candidates", {"plan_id": base, "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None},
                {"template": "sci", "select_cutoff": 0.001, "shots": None}]})])
        if call == 3:
            first = next(data for name, data in _results(messages) if name == "plan.candidates")
            return _turn([("design.submit", {"plan_id": first["plan_id"]})])
        return _turn([("cards.search", {"query": "sqd"})], "should never be asked")

    _, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    entries = journal("h2-ground")
    assert _designer_calls_before_screening(entries) == 3
    assert len(_screened(entries)) == 1  # the submitted plan, not the later one
    row = rows["h2-ground"]
    assert row["correct"] and row["design"] == "screened"
    assert isinstance(row["answer_from_candidate"], bool)


def test_r117_decline_is_a_recorded_design_decision(tmp_path, monkeypatch) -> None:
    reason = "FCI is exact and cheap at two orbitals; no candidate can improve on it."

    def script(call, messages):
        if call == 1:
            return _turn([("design.decline", {"reason": reason})])
        return _turn([("cards.search", {"query": "sqd"})], "should never be asked")

    results, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    entries = journal("h2-ground")
    assert _designer_calls_before_screening(entries) == 1
    states = [entry["state"] for entry in entries]
    assert "SCREEN" not in states and "DESIGN_REFUSED" not in states
    declined = next(entry for entry in entries if entry["state"] == "DESIGN_DECLINED")
    assert declined["reason"] == reason
    row = rows["h2-ground"]
    assert row["completed"] and row["correct"] and row["design"] == "declined"
    assert not row["answer_from_candidate"]
    assert results["aggregate"]["design_declined"] == 1
    assert results["aggregate"]["design_screened"] == 0


def test_r117_without_submit_the_latest_edit_is_screened(tmp_path, monkeypatch) -> None:
    def script(call, messages):
        if call == 1:
            return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None},
                {"template": "sci", "select_cutoff": 0.001, "shots": None}]})])
        if call == 2:
            built = next(data for name, data in _results(messages) if name == "plan.candidates")
            return _turn([("plan.remove_node", {"plan_id": built["plan_id"],
                                                "node_id": max(built["ids"].values())})])
        return _turn(content=FINAL)

    results, rows, journal, _ = _eval(tmp_path, monkeypatch, "h2-ground", script)
    assert len(_screened(journal("h2-ground"))) == 1
    assert rows["h2-ground"]["design"] == "screened"
    assert results["aggregate"]["design_screened"] == 1


def test_r117_the_designer_is_told_how_to_finish(tmp_path, monkeypatch) -> None:
    _, _, _, clients = _eval(tmp_path, monkeypatch, "h2-ground",
                             lambda call, messages: _turn(content="nothing"))
    system = clients[0].messages[0][0]["content"]
    for name in ("design.submit", "design.decline", "plan.candidates"):
        assert name in system, name
    tools = m4a._tools()
    designer = {schema["name"] for schema in tools.schemas("designer")}
    critic = {schema["name"] for schema in tools.schemas("critic")}
    assert {"design.submit", "design.decline"} <= designer
    assert not {"design.submit", "design.decline"} & critic


# R118 ----------------------------------------------------------------------------------------

def _base_plan(tools):
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    return empty, tools.call("modeler", "plan.base", {
        "plan_id": empty, "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]


def test_r118_null_candidate_fields_take_the_template_defaults() -> None:
    tools = m4a._tools()
    _, base = _base_plan(tools)
    built = tools.call("designer", "plan.candidates", {"plan_id": base, "candidates": [
        {"template": "sci", "select_cutoff": None, "shots": None},
        {"template": "sqd", "select_cutoff": None, "shots": None}]})["data"]
    assert len(built["ids"]) == 2
    plan = tools.plans.get(built["plan_id"])
    sci = next(node for node in plan.nodes if node.refinement == "sci")
    assert isinstance(sci.params["select_cutoff"], float)
    sample = next(node for node in plan.nodes if node.op == "sample_configurations")
    assert isinstance(sample.params["shots"], int) and sample.params["shots"] > 0


def test_r118_unsupported_plans_are_refused_with_a_reason() -> None:
    tools = m4a._tools()
    empty, _ = _base_plan(tools)
    with pytest.raises(Exception) as caught:
        tools.call("designer", "plan.candidates", {"plan_id": empty, "candidates": [
            {"template": "sci", "select_cutoff": 0.01, "shots": None}]})
    assert not isinstance(caught.value, (KeyError, IndexError))
    message = str(caught.value)
    assert "reference" in message.lower() and "list index" not in message


def test_r118_algorithm_tools_are_offered_only_where_they_can_verify() -> None:
    from chemflow.agent.task import TaskSpec, TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.runtime import Executor

    molecular = {schema["name"] for schema in m4a._tools().schemas("designer")}
    assert not {"algorithm.verify", "algorithm.register"} & molecular
    spec = TaskSpec.model_validate({
        "question": "AIM impurity spectral function", "target": a1b.TARGET,
        "gates": [a1a.ACCURACY], "budget": {"wall_s": 900},
        "allowed_effects": ["pure", "simulator"], "decision": "d"})
    store = TaskStore("reviewer")
    registry = a1b._registry()
    tools = make_tools(registry, Executor(registry), store.freeze(spec, "reviewer"), store)
    impurity = {schema["name"] for schema in tools.schemas("designer")}
    assert {"algorithm.verify", "algorithm.register"} <= impurity
    with pytest.raises(Exception) as caught:
        tools.call("designer", "algorithm.register", {"report_id": "made-up", "name": "x"})
    assert "algorithm.verify" in str(caught.value)
