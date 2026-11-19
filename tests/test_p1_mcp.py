import asyncio
import json

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import ClientSession
from mcp.client._memory import create_client_server_memory_streams
from test_m3a_task import spec

from chemflow.agent.tools import make_tools
from chemflow.mcp import create_server
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def test_profiles_and_binding(tmp_path):
    private, public = generate_keypair(b"human-passphrase")
    signer = TaskSigner(private, b"human-passphrase")
    key = tmp_path / "human.pub"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    record = signer.freeze(spec(), "human")
    task.write_text(record.model_dump_json())
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, TaskVerifier(public))

    async def exercise():
        for profile in ("planner", "modeler", "designer", "critic", "analyst", "full"):
            server = create_server(profile, str(task), public_key=str(key))
            listed = await server.list_tools()
            expected = ({item["name"] for role in ("planner", "modeler", "designer",
                         "critic", "analyst") for item in tools.schemas(role)}
                        if profile == "full" else
                        {item["name"] for item in tools.schemas(profile)})
            assert {item.name for item in listed} == {name.replace(".", "_") for name in expected}
            assert all(len(item.name) <= 40 for item in listed)
            assert all(item.input_schema == next(schema["parameters"] for role in
                       ("planner", "modeler", "designer", "critic", "analyst")
                       for schema in tools.schemas(role)
                       if schema["name"].replace(".", "_") == item.name) for item in listed)
        designer = create_server("designer", str(task), public_key=str(key))
        created = await designer.call_tool("plan_new", {})
        plan_id = json.loads(created.content[0].text)["data"]["plan_id"]
        with pytest.raises(Exception) as refused:
            await designer.call_tool("run", {"plan_id": plan_id, "scale": "production"})
        assert private.decode() not in str(created) + str(refused.value)

    asyncio.run(exercise())
    assert "plan.new" in (tmp_path / "task.journal").read_text()
    task.write_text(record.model_copy(update={"signature": "0" * 64}).model_dump_json())
    with pytest.raises(ValueError, match="CF-E120"):
        create_server("designer", str(task), public_key=str(key))


def test_h2_through_mcp_client(tmp_path):
    private, public = generate_keypair(b"human-passphrase")
    key = tmp_path / "human.pub"
    key.write_bytes(public)
    task = tmp_path / "h2.json"
    task.write_text(TaskSigner(private, b"human-passphrase").freeze(
        spec(), "human").model_dump_json())
    server = create_server("full", str(task), public_key=str(key))

    async def exercise():
        async with (create_client_server_memory_streams() as (client_streams, server_streams),
                    anyio.create_task_group() as group):
                group.start_soon(server._lowlevel_server.run, *server_streams,
                                 server._lowlevel_server.create_initialization_options())
                async with ClientSession(*client_streams) as client:
                    await client.initialize()
                    listed = await client.list_tools()
                    assert any(item.name == "plan_base" for item in listed.tools)
                    created = await client.call_tool("plan_new", {})
                    plan_id = json.loads(created.content[0].text)["data"]["plan_id"]
                    added = await client.call_tool("plan_add_node", {"plan_id": plan_id,
                        "node": {"id": "initial", "op": "chemflow.molecule", "inputs": [],
                                 "params": [], "effects": "pure", "refinement": None,
                                 "binding": None}})
                    assert not added.is_error
                    edited_id = json.loads(added.content[0].text)["data"]["plan_id"]
                    based = await client.call_tool("plan_base", {"plan_id": edited_id,
                        "active": {"n_elec": 2, "n_orb": 2}})
                    base_id = json.loads(based.content[0].text)["data"]["plan_id"]
                    checked = await client.call_tool("check", {"plan_id": base_id})
                    assert not checked.is_error
                    screened = await client.call_tool("run", {"plan_id": base_id})
                    assert not screened.is_error
                    output = json.loads(screened.content[0].text)["data"]
                    assert output["gates"][0]["passed"]
                    results = await client.call_tool("results_query", {})
                    assert json.loads(results.content[0].text)["data"]
                group.cancel_scope.cancel()

    anyio.run(exercise)
