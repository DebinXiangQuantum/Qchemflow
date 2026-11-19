import json
from copy import deepcopy

import pytest
from test_g5_evaluation import synthetic
from test_m3b_llm import policy

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.tools import ToolInput, ToolRegistry
from chemflow.design import supports_verification
from chemflow.eval.runner import aggregate_rows, score_task, write_report
from chemflow.ir.schema import Node, Program


class Payload(ToolInput):
    label: str


class Arguments(ToolInput):
    payload: Payload | None
    items: list[Payload]
    note: str


def turn(name=None, arguments=None):
    return {"content": "", "tool_calls": [] if name is None else [
        {"id": "call", "name": name, "arguments": arguments or {}}]}


def records(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def test_schema_inlines_repeated_nullable_and_array_definitions():
    registry = ToolRegistry()
    schema = registry._schema_for("plan.add_node", Arguments)
    assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"payload", "items", "note"}
    payload = schema["properties"]["payload"]["anyOf"][0]
    items = schema["properties"]["items"]["items"]
    for nested in (payload, items):
        assert nested["type"] == "object"
        assert nested["required"] == ["label"]
        assert nested["additionalProperties"] is False
    payload["properties"]["label"]["type"] = "integer"
    assert items["properties"]["label"]["type"] == "string"
    assert registry._schema_for("plan.add_node", Arguments) != schema


@pytest.mark.parametrize("payload,items,valid", [
    ('{"label":"object"}', '[{"label":"array"}]', True),
    ({"label": "object"}, ["{\"label\":\"nested\"}"], True),
    (None, [], True),
    ("not valid JSON", [], False),
    ("[]", [], False),
    (json.dumps('{"label":"double encoded"}'), [], False),
    ({"label": "object"}, "not valid JSON", False),
    ({"label": "object"}, "{}", False),
])
def test_container_decoding_is_once_typed_and_preserves_journal(tmp_path, payload, items, valid):
    registry = ToolRegistry()
    registry.register("plan.add_node", {"designer"}, lambda data: data.model_dump(), Arguments)
    arguments = {"payload": payload, "items": items, "note": '{"keep":"string"}'}
    original = deepcopy(arguments)
    client = FakeLLMClient([turn("plan.add_node", arguments), turn()])
    journal = Journal(tmp_path / "journal.jsonl")
    result = LLMPolicy("designer", client, registry, journal=journal).decide({})
    output = result["outputs"][0]
    if valid:
        data = output["result"]["data"]
        assert data["note"] == original["note"]
        assert data["payload"] is None or isinstance(data["payload"], dict)
        assert isinstance(data["items"], list)
        assert all(isinstance(item, dict) for item in data["items"])
    else:
        assert output["error"]["code"] == "ValidationError"
        assert "result" not in output
    logged = next(entry for entry in records(journal) if entry["state"] == "LLM_TOOL")
    assert logged["calls"][0]["arguments"] == original
    assert arguments == original


@pytest.mark.parametrize("responses,reason", [
    ([turn()], "text reply"),
    ([turn("missing")] * 3, "consecutive tool errors"),
    ([turn("cards.search", {"query": "sci"})] * 3, "turn limit"),
    ([None] * 3, "LLM_FAILED"),
])
def test_designer_termination_reason_is_explicit(tmp_path, responses, reason):
    agent, _, journal = policy(tmp_path, "designer", responses)
    agent.max_turns = 3
    assert agent.decide({})["end_reason"] == reason
    assert records(journal)[-1]["decision"]["end_reason"] == reason


@pytest.mark.parametrize("terminal,arguments", [
    ("design.submit", {"plan_id": None}),
    ("design.decline", {"reason": "No applicable recipe"}),
])
def test_terminal_tools_stop_remaining_calls_in_same_response(tmp_path, terminal, arguments):
    response = turn(terminal, arguments)
    response["tool_calls"].extend(turn("plan.new", {"identifier": "must-not-build"})["tool_calls"])
    agent, client, journal = policy(tmp_path, "designer", [response])
    if terminal == "design.submit":
        base, _ = Orchestrator._base_plan(agent.record, agent.record.spec.target.active)
        agent.tools.design_base_plan_id = agent.tools.plans.put(base)
        trial, _ = Orchestrator._candidate_plan(base, [{"template": "sci"}])
        arguments["plan_id"] = agent.tools.plans.put(trial)
        response["tool_calls"][0]["arguments"] = arguments
    existing = set(agent.tools.plans._plans)
    result = agent.decide({})
    assert result["end_reason"] == terminal
    assert len(client.messages) == len(result["outputs"]) == 1
    assert set(agent.tools.plans._plans) == existing
    assert [entry["decision"]["name"] for entry in records(journal)
            if entry["state"] == "LLM_TOOL"] == [terminal]
    for role in ("planner", "modeler", "critic", "analyst", "orchestrator"):
        assert terminal not in {schema["name"] for schema in agent.tools.schemas(role)}
        with pytest.raises(PermissionError):
            agent.tools.call(role, terminal, arguments)


def test_invalid_decline_does_not_end_turn(tmp_path):
    agent, client, _ = policy(tmp_path, "designer", [
        turn("design.decline", {"reason": "  "}),
        turn("design.decline", {"reason": "No useful candidate"})])
    result = agent.decide({})
    assert len(client.messages) == 2
    assert result["outputs"][0]["error"]["code"] == "ValidationError"
    assert result["end_reason"] == "design.decline"


@pytest.mark.parametrize("operator", ["interaction_energy", "spectral_function"])
def test_unsupported_reference_names_operator_without_adapter_wrapping(tmp_path, operator):
    agent, _, _ = policy(tmp_path, "designer", [])
    plan = Program(name="unsupported", nodes=[Node(id="reference", op=operator)],
                   references=["reference"])
    plan_id = agent.tools.plans.put(plan)
    with pytest.raises(ValueError, match=f"not available for {operator} references") as caught:
        agent.tools.call("designer", "plan.candidates", {
            "plan_id": plan_id, "candidates": [{"template": "sci"}]})
    assert "supported: ground_energy, reaction_energy" in str(caught.value)
    assert caught.value.__cause__ is None


def test_null_defaults_match_first_scripted_choices(tmp_path):
    agent, _, _ = policy(tmp_path, "designer", [])
    base, _ = Orchestrator._base_plan(agent.record, agent.record.spec.target.active)
    nulls = [{"template": template, "shots": None, "select_cutoff": None}
             for template in ("sci", "sqd")]
    trial, _ = Orchestrator._candidate_plan(base, nulls)
    explicit, _ = Orchestrator._candidate_plan(base, [
        {"template": "sci", "select_cutoff": 0.01},
        {"template": "sqd", "shots": 1024}])
    assert trial == explicit
    assert trial.nodes[:len(base.nodes)] == base.nodes
    assert trial.references == base.references


@pytest.mark.parametrize("kind,supported", [
    ("ground_energy", True), ("molecular_ground_energy", True),
    ("lattice_ground_energy", True), ("spectral_function", True),
    ("interaction_energy", False), ("reaction_energy", False),
])
def test_verification_scope_uses_exact_target_validator(kind, supported):
    assert supports_verification({"kind": kind, "model": {"impurity_model": {}}}) is supported
    assert not supports_verification({"kind": kind, "model": {"other": {}}})
    assert not supports_verification({"kind": kind, "model": {
        "impurity_model": {}, "other": {}}})


def test_design_and_relation_totals_are_distinct_and_reported(tmp_path):
    task, summary, _ = synthetic()
    screened = score_task(task, summary, [{"state": "SCREEN", "outputs": ["chosen"]}])
    declined = score_task(task, summary, [{"state": "DESIGN_DECLINED"}])
    refused = score_task(task, summary, [{"state": "DESIGN_REFUSED"}])
    unanswered = score_task(task, {**summary, "candidate": None}, [])
    rows = [screened, declined, refused, unanswered]
    assert [row["design"] for row in rows] == ["screened", "declined", "refused", None]
    assert [row["answer_from_candidate"] for row in rows] == [True, False, False, False]
    assert unanswered["relation_ok"] is None
    aggregate = aggregate_rows(rows)
    expected = {"relation_ok": 3, "relation_rated": 3, "design_screened": 1,
                "design_declined": 1, "design_refused": 1, "answer_from_candidate": 1}
    assert {key: aggregate[key] for key in expected} == expected
    write_report({"metadata": {"suite": "core", "policy": "scripted", "model": None,
                               "roles": "designer", "git_commit": "offline"},
                  "rows": rows, "aggregate": aggregate}, tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert all(f"| {key} | {value} |" in report for key, value in expected.items())


def test_reviewer_submit_rejects_unknown_plan_ids():
    import test_review_m4a as m4a

    tools = m4a._tools()
    with pytest.raises(ValueError, match="Unknown plan_id"):
        tools.call("designer", "design.submit", {"plan_id": "0" * 32})
