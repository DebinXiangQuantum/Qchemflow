import pytest

from chemflow.hooks import evaluate, normalize_tool


@pytest.mark.parametrize("name", ["mcp__chemflow__plan_add_node",
    "mcp__plugin_chemflow_chemflow__plan_add_node", "mcp_chemflow_plan_add_node",
    "extension__plan_add_node", "plan_add_node"])
def test_prefixes(name):
    assert normalize_tool(name) == "plan.add_node"


@pytest.mark.parametrize("shape", [
    {"hook_event_name": "PreToolUse", "session_id": "c", "cwd": "."},
    {"hook_event_name": "pre_tool_call", "session_id": "x", "cwd": "."},
    {"event": "before_tool", "working_dir": ".", "tool_call_id": "g"},
    {"hook_event_name": "pre_tool_call", "profile": "designer", "extra": {}, "cwd": "."},
])
def test_framework_shapes(shape, tmp_path):
    shape.update({"tool_name": "mcp__chemflow__run", "tool_input": {"scale": "production"}})
    assert evaluate(shape, journal=str(tmp_path / "hook.journal")) is None
    shape.update({"tool_name": "Bash", "tool_input": {"command": "cat task.json"}})
    assert evaluate(shape, journal=str(tmp_path / "hook.journal")) is None


def test_stop_requires_report(tmp_path):
    path = str(tmp_path / "hook.journal")
    event = {"hook_event_name": "PostToolUse", "tool_name": "run", "tool_input": {}}
    assert evaluate(event, journal=path) is None
    event["hook_event_name"] = "Stop"
    assert "report_write" in evaluate(event, journal=path)
    event.update({"hook_event_name": "PostToolUse", "tool_name": "report_write"})
    evaluate(event, journal=path)
    event["hook_event_name"] = "Stop"
    assert evaluate(event, journal=path) is None


@pytest.mark.parametrize("target", ["human.pem", "human.pub", "keys/private.pem",
    "task.json", "task.ledger.json", ".chemflow-keys/private.pem"])
def test_framework_write_protects_trust_paths(target, tmp_path):
    event = {"hook_event_name": "PreToolUse", "tool_name": "Write", "cwd": str(tmp_path),
             "tool_input": {"file_path": target, "content": "replacement"}}
    assert "protected" in evaluate(event, journal=str(tmp_path / "hook.journal"),
                                   protected=(str(tmp_path),))
