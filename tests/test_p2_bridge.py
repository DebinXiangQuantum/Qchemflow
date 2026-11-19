import asyncio
import json

import pytest
from test_m3a_task import spec

from chemflow.cli import main
from chemflow.mcp import create_server
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def test_persistent_bridge(tmp_path, monkeypatch, capsys):
    private, public = generate_keypair(b"human")
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    task.write_text(TaskSigner(private, b"human").freeze(spec(), "human").model_dump_json())
    fingerprint = TaskVerifier(public).fingerprint

    def call(profile, name, arguments):
        monkeypatch.setattr("sys.argv", ["chemflow", "tool", "call", "--profile", profile,
                            "--task", str(task), "--public-key", str(key),
                            "--key-fingerprint", fingerprint, "--name", name,
                            "--args-json", json.dumps(arguments)])
        main()
        return json.loads(capsys.readouterr().out)

    identifier = call("designer", "plan_new", {})["data"]["plan_id"]
    assert (tmp_path / ".chemflow-plans" / f"{identifier}.json").exists()
    assert call("designer", "plan_render", {"plan_id": identifier})["data"]
    server = create_server("designer", str(task), public_key=str(key),
                           key_fingerprint=fingerprint)
    assert server._tool_manager.get_tool("plan_render")
    assert server._tool_manager.get_tool("plan_new")
    rendered = asyncio.run(server.call_tool("plan_render", {"plan_id": identifier}))
    assert not rendered.is_error
    from_server = json.loads(rendered.content[0].text)["data"]
    assert from_server == call("designer", "plan_render", {"plan_id": identifier})["data"]
    with pytest.raises(SystemExit) as refused:
        call("critic", "plan_render", {"plan_id": identifier})
    assert refused.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "PermissionError"
    with pytest.raises(SystemExit):
        call("designer", "plan_render", {"plan_id": "../../escaped"})
