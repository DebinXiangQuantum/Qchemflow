import asyncio

from test_m3a_task import spec

from chemflow.agent.tools import make_tools
from chemflow.mcp import create_server
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def test_list_tools_matches_registry_strict_schema(tmp_path):
    private, public = generate_keypair(b"human")
    key = tmp_path / "human.pub"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    record = TaskSigner(private, b"human").freeze(spec(), "human")
    task.write_text(record.model_dump_json())
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, TaskVerifier(public))

    async def check():
        for profile in ("planner", "modeler", "designer", "critic", "analyst", "full"):
            server = create_server(profile, str(task), public_key=str(key))
            listed = await server.list_tools()
            roles = ("planner", "modeler", "designer", "critic", "analyst") if profile == "full" else (profile,)
            expected = {item["name"].replace(".", "_"): item["parameters"]
                        for role in roles for item in tools.schemas(role)}
            assert {item.name: item.input_schema for item in listed} == expected

    asyncio.run(check())
