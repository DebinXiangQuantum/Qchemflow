"""Reviewer regression tests for M4a (written by Claude; implementers must not edit)."""
import pytest

ROLES = ("planner", "modeler", "designer", "critic", "analyst", "orchestrator")


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
