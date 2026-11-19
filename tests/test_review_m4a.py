"""Reviewer regression tests for M4a (written by Claude; implementers must not edit)."""
import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor

ROLES = ("planner", "modeler", "designer", "critic", "analyst", "orchestrator")


def _tools():
    spec = TaskSpec.model_validate({
        "question": "q", "target": {
            "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
        "budget": {}, "allowed_effects": ["pure"], "decision": "d"})
    store = TaskStore("reviewer")
    record = store.freeze(spec, "reviewer")
    return make_tools(molecular_registry(), Executor(molecular_registry()), record, store)


def _strict_problems(schema, path="$", defs=None) -> list[str]:
    """Strict tool use: every object closed (additionalProperties false), all properties
    required (optional values as nullable types), recursively through $ref/anyOf/items."""
    defs = schema.get("$defs", {}) if defs is None else defs
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return _strict_problems(defs.get(name, {}), f"{path}->{name}", defs)
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: open object")
        if set(schema.get("required", [])) != set(properties):
            problems.append(f"{path}: required != properties")
        for key, value in properties.items():
            problems += _strict_problems(value, f"{path}.{key}", defs)
    for keyword in ("anyOf", "oneOf", "allOf"):
        for index, option in enumerate(schema.get(keyword, [])):
            problems += _strict_problems(option, f"{path}|{keyword}{index}", defs)
    if "items" in schema:
        problems += _strict_problems(schema["items"], f"{path}[]", defs)
    return problems


def test_every_exported_tool_schema_is_strict_compatible() -> None:
    tools = _tools()
    failures = {}
    for role in ROLES:
        for item in tools.schemas(role):
            problems = _strict_problems(item["parameters"])
            if problems:
                failures[item["name"]] = problems[:3]
    assert not failures, failures


def test_plans_travel_by_handle_not_by_value() -> None:
    tools = _tools()
    for role in ("modeler", "designer"):
        for item in tools.schemas(role):
            if item["name"].startswith("plan.") and item["name"] != "plan.new":
                properties = item["parameters"].get("properties", {})
                assert "plan_id" in properties, item["name"]
                assert "plan" not in properties, item["name"]


def test_every_tool_has_a_meaningful_description() -> None:
    tools = _tools()
    for role in ROLES:
        for item in tools.schemas(role):
            description = item.get("description", "")
            assert len(description) >= 40, item["name"]
            assert not description.startswith("ChemFlow "), item["name"]


def test_role_model_defaults_follow_the_user_decision() -> None:
    pytest.importorskip("anthropic")
    from chemflow.agent.llm_anthropic import ClaudeClient

    expected = {"designer": "claude-opus-5-5", "critic": "claude-opus-5-5",
                "planner": "claude-sonnet-5-5", "modeler": "claude-sonnet-5-5",
                "analyst": "claude-sonnet-5-5"}
    for role, model in expected.items():
        assert ClaudeClient.for_role(role).model == model, role


def test_openai_adapter_requires_an_explicit_model() -> None:
    pytest.importorskip("openai")
    from chemflow.agent.llm_openai import OpenAIClient

    with pytest.raises((TypeError, ValueError)):
        OpenAIClient()
