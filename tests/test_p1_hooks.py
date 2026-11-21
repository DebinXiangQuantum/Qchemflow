import pytest

from chemflow.hooks import evaluate, normalize_tool


@pytest.mark.parametrize("name", ["mcp__chemflow__screen_review",
    "mcp__plugin_chemflow_chemflow-qcl__screen_review", "mcp_chemflow_screen_review",
    "extension__screen_review", "screen_review"])
def test_prefixes(name):
    assert normalize_tool(name) == "screen.review"


@pytest.mark.parametrize("shape", [
    {"hook_event_name": "PreToolUse", "session_id": "c", "cwd": "."},
    {"hook_event_name": "pre_tool_call", "session_id": "x", "cwd": "."},
    {"event": "before_tool", "working_dir": ".", "tool_call_id": "g"},
    {"hook_event_name": "pre_tool_call", "profile": "qcl", "extra": {}, "cwd": "."},
])
def test_framework_shapes(shape, tmp_path):
    shape.update({"tool_name": "mcp__chemflow__screen_run", "tool_input": {}})
    assert evaluate(shape, journal=str(tmp_path / "hook.journal")) is None
    shape.update({"tool_name": "Bash", "tool_input": {"command": "cat task.json"}})
    assert evaluate(shape, journal=str(tmp_path / "hook.journal")) is None


def test_screen_runs_are_journalled_and_stopping_is_never_blocked(tmp_path):
    # screen_run writes the report itself, so no report tool is required before stopping.
    path = tmp_path / "hook.journal"
    event = {"hook_event_name": "PostToolUse", "tool_name": "mcp__chemflow__screen_run",
             "tool_input": {}, "session_id": "s"}
    assert evaluate(event, journal=str(path)) is None
    assert '"tool":"screen.run"' in path.read_text()
    event["hook_event_name"] = "Stop"
    assert evaluate(event, journal=str(path)) is None


def test_hook_journals_stay_out_of_user_projects(tmp_path, monkeypatch):
    project, home = tmp_path / "project", tmp_path / "home"
    project.mkdir()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for tool in ("Bash", "mcp__plugin_chemflow_chemflow-qcl__screen_run"):
        evaluate({"hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": {},
                  "session_id": "s", "cwd": str(project)})
    assert not list(project.iterdir())
    text = "".join(path.read_text() for path in (home / ".chemflow").rglob("*journal*")
                   if path.is_file())
    assert "screen.run" in text and "Bash" not in text


@pytest.mark.parametrize("target", ["human.pem", "human.pub", "keys/private.pem",
    "task.json", "task.ledger.json", ".chemflow-keys/private.pem"])
def test_framework_write_protects_trust_paths(target, tmp_path):
    event = {"hook_event_name": "PreToolUse", "tool_name": "Write", "cwd": str(tmp_path),
             "tool_input": {"file_path": target, "content": "replacement"}}
    assert "protected" in evaluate(event, journal=str(tmp_path / "hook.journal"),
                                   protected=(str(tmp_path),))
