"""Task-bound, role-scoped MCP adapter for the ChemFlow tool registry."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

from pydantic import ValidationError

from chemflow.agent.task import TaskRecord
from chemflow.agent.tools import make_tools
from chemflow.diagnostics import AdapterError
from chemflow.domains import load_registry
from chemflow.ids import canonical_json
from chemflow.runtime import Executor
from chemflow.signing import TaskVerifier

PROFILES = ("planner", "modeler", "designer", "critic", "analyst", "full")


def _install_strict_schema(server, name: str, schema: dict) -> None:
    # MCPServer 2.3.0 has no public explicit input-schema registration API.
    # Keep this private-SDK dependency here; list_tools regression tests guard it.
    registered = server._tool_manager.get_tool(name)
    if registered is None:
        raise RuntimeError(f"MCP tool not registered: {name}")
    registered.parameters = schema


def load_task(path: str, verifier: TaskVerifier) -> tuple[TaskRecord, TaskVerifier]:
    try:
        record = TaskRecord.model_validate_json(Path(path).read_text())
        verifier.verify(record)
        return record, verifier
    except (OSError, ValueError, TypeError, ValidationError, AdapterError):
        raise ValueError("CF-E120: Missing or invalid signed task record") from None


def create_server(profile: str, task: str, *, public_key: str,
                  journal: str | None = None, key_fingerprint: str | None = None,
                  qpu: str | None = None, gpu: bool = False, packs: str = "molecular"):
    from mcp.server.mcpserver import MCPServer

    tools = create_tools(profile, task, public_key=public_key,
                         key_fingerprint=key_fingerprint, qpu=qpu, gpu=gpu, packs=packs)
    journal = journal or str(Path(task).with_suffix(".journal"))
    server = MCPServer("chemflow")
    role = "orchestrator" if profile == "full" else profile
    schemas = tools.schemas(role)
    owners = {item["name"]: role for item in schemas}
    if profile == "full":
        schemas = []
        owners = {}
        for candidate in PROFILES[:-1]:
            for item in tools.schemas(candidate):
                if item["name"] not in owners:
                    owners[item["name"]] = candidate
                    schemas.append(item)
    for schema in schemas:
        name = schema["name"]
        mcp_name = name.replace(".", "_")
        input_model = tools._tools[name][2]

        def dispatch_factory(tool_name: str, model: type, owner: str):
            async def dispatch(**kwargs):
                arguments = model.model_validate(kwargs).model_dump(mode="json", exclude_unset=True)
                outcome = "error"
                try:
                    result = tools.call(owner, tool_name, arguments)
                    outcome = "ok"
                    return result
                finally:
                    entry = {"profile": profile, "session_id": None, "tool": tool_name,
                             "args_hash": hashlib.sha256(canonical_json(arguments).encode()).hexdigest(),
                             "outcome": outcome}
                    await _append_journal(Path(journal), entry)

            parameters = [inspect.Parameter(field_name, inspect.Parameter.KEYWORD_ONLY,
                          default=inspect.Parameter.empty if field.is_required() else field.default,
                          annotation=field.annotation)
                          for field_name, field in model.model_fields.items()]
            dispatch.__signature__ = inspect.Signature(parameters)
            dispatch.__annotations__ = {field_name: field.annotation
                                        for field_name, field in model.model_fields.items()}
            return dispatch

        server.add_tool(dispatch_factory(name, input_model, owners[name]), name=mcp_name,
                        description=schema["description"], structured_output=False)
        _install_strict_schema(server, mcp_name, schema["parameters"])
    return server


def create_tools(profile: str, task: str, *, public_key: str,
                 key_fingerprint: str | None = None, qpu: str | None = None,
                 gpu: bool = False, packs: str = "molecular"):
    if profile not in PROFILES:
        raise ValueError("Unknown role profile")
    verifier = TaskVerifier(Path(public_key).read_bytes())
    if key_fingerprint is not None and key_fingerprint != verifier.fingerprint:
        raise ValueError("CF-E120: Public key fingerprint mismatch")
    record, store = load_task(task, verifier)
    registry = load_registry([name.strip() for name in packs.split(",")])
    if gpu:
        from chemflow.backends.gpu import register_gpu

        register_gpu(registry)
    if qpu is not None:
        from chemflow.backends.lqcloud import LqcloudBackend, register_lqcloud

        register_lqcloud(registry, LqcloudBackend(backend=qpu),
                         identifier=f"lqcloud_{qpu.lower()}")
    return make_tools(registry, Executor(registry), record, store,
                      plan_directory=Path(task).resolve().parent / ".chemflow-plans")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-bound ChemFlow MCP tool server",
                                     allow_abbrev=False)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--key-fingerprint")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--journal")
    parser.add_argument("--qpu", choices=("MQ02",))
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--packs", default="molecular")
    args = parser.parse_args()
    try:
        server = create_server(args.profile, args.task, public_key=args.public_key,
                               key_fingerprint=args.key_fingerprint, journal=args.journal,
                               qpu=args.qpu, gpu=args.gpu, packs=args.packs)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"CF-E120: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.transport == "http":
        server.run("streamable-http", host="127.0.0.1", port=args.port)
    else:
        server.run("stdio")


async def _append_journal(path: Path, entry: dict) -> None:
    import anyio

    await anyio.to_thread.run_sync(_write_journal, path, entry)


def _write_journal(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(entry) + "\n")
