"""The CLI tool bridge and the MCP server expose the same qcl tools for a signed task."""

import asyncio
import json

import pytest
from test_qcl_mcp import _signed

from chemflow.cli import main
from chemflow.mcp import create_server
from chemflow.signing import TaskVerifier

pytest.importorskip("qcl")


def test_persistent_bridge(tmp_path, monkeypatch, capsys):
    _, task, key = _signed(tmp_path)
    fingerprint = TaskVerifier(key.read_bytes()).fingerprint

    def call(name, arguments, profile="qcl"):
        monkeypatch.setattr("sys.argv", ["chemflow", "tool", "call", "--profile", profile,
                            "--task", str(task), "--public-key", str(key),
                            "--key-fingerprint", fingerprint, "--name", name,
                            "--args-json", json.dumps(arguments)])
        main()
        return json.loads(capsys.readouterr().out)

    methods = call("methods_list", {})["data"]
    assert methods["reference"] == "fci"
    server = create_server("qcl", str(task), public_key=str(key), key_fingerprint=fingerprint)
    assert {"task_get", "methods_list", "screen_review", "screen_run"} <= {
        tool.name for tool in asyncio.run(server.list_tools())}
    listed = asyncio.run(server.call_tool("methods_list", {}))
    assert not listed.is_error
    assert json.loads(listed.content[0].text)["data"] == methods
    with pytest.raises(SystemExit) as refused:
        call("plan_new", {})
    assert refused.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "PermissionError"
    with pytest.raises(SystemExit) as refused:
        call("methods_list", {}, profile="designer")
    assert refused.value.code == 2
