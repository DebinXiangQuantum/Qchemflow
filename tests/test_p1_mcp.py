import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("qcl")

import anyio
from mcp import ClientSession
from mcp.client._memory import create_client_server_memory_streams
from test_qcl_mcp import _signed

from chemflow.mcp import create_server


def test_h2_screen_through_mcp_client(tmp_path):
    _, task, key = _signed(tmp_path)
    server = create_server("qcl", str(task), public_key=str(key))

    async def exercise():
        async with (create_client_server_memory_streams() as (client_streams, server_streams),
                    anyio.create_task_group() as group):
            group.start_soon(server._lowlevel_server.run, *server_streams,
                             server._lowlevel_server.create_initialization_options())
            async with ClientSession(*client_streams) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert {item.name for item in listed.tools} == {
                    "task_get", "methods_list", "diagnose_model", "screen_review",
                    "screen_run", "results_query"}
                candidates = [{"template": "sci", "select_cutoff": 1e-3}]
                reviewed = await client.call_tool("screen_review", {"candidates": candidates})
                assert not reviewed.is_error
                assert json.loads(reviewed.content[0].text)["data"]["relations"] == {
                    "candidate0": "variational_upper_bound"}
                screened = await client.call_tool("screen_run", {"candidates": candidates})
                assert not screened.is_error
                output = json.loads(screened.content[0].text)["data"]
                assert output["decision"] in ("stop", "escalate")
                results = await client.call_tool("results_query", {})
                assert json.loads(results.content[0].text)["data"]
                refused = await client.call_tool("screen_review", {"candidates": [
                    {"template": "vqe"}]})
                assert refused.is_error
            group.cancel_scope.cancel()

    anyio.run(exercise)


def test_the_tool_server_needs_no_secret(tmp_path, monkeypatch):
    import inspect

    parameters = set(inspect.signature(create_server).parameters)
    assert not {"secret", "secret_file", "passphrase"} & parameters
    assert "public_key" in parameters
    monkeypatch.delenv("CHEMFLOW_TASK_SECRET", raising=False)
    _, task, key = _signed(tmp_path)
    assert create_server("qcl", str(task), public_key=str(key)) is not None
    with pytest.raises(ValueError, match="Unknown role profile"):
        create_server("designer", str(task), public_key=str(key))
