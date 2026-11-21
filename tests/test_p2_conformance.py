import asyncio
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from test_m3a_task import spec

from chemflow.plugin_build import SOURCE, TARGETS, _tools, build
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


@pytest.fixture
def packages(tmp_path):
    private, public = generate_keypair(b"human")
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    task.write_text(TaskSigner(private, b"human").freeze(spec(), "human").model_dump_json())
    outputs = {}
    for target in TARGETS:
        outputs[target] = tmp_path / target
        build(target, outputs[target], key, task=task)
    return outputs, key, task, TaskVerifier(public).fingerprint


def _servers(target, output):
    if target in ("agent-plugins", "codex"):
        root = output / "plugins/chemflow" if target == "codex" else output
        return json.loads((root / "mcp.json").read_text())["mcpServers"]
    if target == "claude-code":
        return json.loads((output / ".mcp.json").read_text())["mcpServers"]
    if target == "goose":
        return yaml.safe_load((output / "goose-config.yaml").read_text())["extensions"]
    if target == "hermes":
        return yaml.safe_load((output / "hermes-config.yaml").read_text())["mcp_servers"]
    return {}


def test_manifests_skills_and_servers(packages):
    outputs, key, task, fingerprint = packages
    for target, output in outputs.items():
        skills = (output / ".agents/skills" if target == "goose" else
                  output / "plugins/chemflow/skills" if target == "codex" else output / "skills")
        for source in (SOURCE / "skills").glob("*/SKILL.md"):
            content = (skills / source.parent.name / "SKILL.md").read_text()
            assert content == source.read_text()
            metadata = yaml.safe_load(content.split("---", 2)[1])
            assert metadata["name"] == source.parent.name
            assert len(metadata["description"]) <= 1024
        for name, entry in _servers(target, output).items():
            assert entry["args"][:2] == ["--profile", name.removeprefix("chemflow-")]
            if target == "goose":
                assert entry["type"] == "stdio" and entry["env_keys"] == []
                assert entry["args"][2:] == ["--task", str(task), "--public-key", str(key),
                                             "--key-fingerprint", fingerprint]
                assert Path(entry["cmd"]).is_absolute() and Path(entry["cmd"]).exists()
                continue
            assert entry["args"][2:] == ["--task", str(task), "--public-key", str(key),
                                         "--key-fingerprint", fingerprint]
            assert Path(entry["command"]).is_absolute() and Path(entry["command"]).exists()
    agent = json.loads((outputs["agent-plugins"] / "plugin.json").read_text())
    assert agent["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    mcp = json.loads((outputs["agent-plugins"] / "mcp.json").read_text())
    assert mcp["$schema"] == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
    assert mcp["mcpServers"]["chemflow-qcl"]["type"] == "stdio"
    codex = json.loads((outputs["codex"] / "plugins/chemflow/plugin.json").read_text())
    assert codex["extensions"]["com.openai"]["interface"]["displayName"] == "ChemFlow"
    marketplace = json.loads((outputs["codex"] / ".agents/plugins/marketplace.json").read_text())
    assert marketplace["plugins"][0]["source"] == {"source": "local", "path": "./plugins/chemflow"}
    assert (outputs["codex"] / "plugins/chemflow/AGENTS.md").read_text() == (SOURCE / "rules/chemflow-rules.md").read_text()
    assert "AGENTS.md" in (outputs["goose"] / ".goosehints").read_text()


def test_hooks_and_hermes_register(packages):
    outputs, _, _, _ = packages
    payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "screen_run",
                          "tool_input": {}})
    for target in ("agent-plugins", "codex", "claude-code", "goose"):
        path = (outputs[target] / ".agents/plugins/chemflow/hooks/hooks.json" if target == "goose"
                else outputs[target] / "plugins/chemflow/hooks/hooks.json" if target == "codex"
                else outputs[target] / "hooks/hooks.json")
        hooks = json.loads(path.read_text())["hooks"]
        assert set(hooks) == {"PreToolUse", "PostToolUse", "Stop"}
        command = hooks["PreToolUse"][0]["hooks"][0]["command"].split()
        assert hooks["PreToolUse"][0]["matcher"] == ".*"
        if target == "goose":
            assert hooks["PreToolUse"][0]["on_failure"] == "block"
        result = subprocess.run(command, input=payload, text=True, capture_output=True, check=False)
        assert result.returncode == 0 and not result.stderr
    hermes = outputs["hermes"]
    config = yaml.safe_load((hermes / "hermes-config.yaml").read_text())
    assert config["hooks"]["pre_tool_call"][0]["fail_closed"] is True
    descriptor = importlib.util.spec_from_file_location("generated_hermes", hermes / "__init__.py")
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)

    class Context:
        def __init__(self):
            self.skills = []
            self.hooks = {}

        def register_skill(self, name, path):
            self.skills.append((name, path))

        def register_hook(self, event, callback):
            self.hooks[event] = callback

    context = Context()
    module.register(context)
    assert len(context.skills) == len(list((SOURCE / "skills").iterdir()))
    assert context.hooks["pre_tool_call"](json.loads(payload)) is None


def test_stdio_mcp_lists_profile(packages):
    outputs, _, _, _ = packages
    entry = _servers("agent-plugins", outputs["agent-plugins"])["chemflow-qcl"]

    async def exercise():
        parameters = StdioServerParameters(command=entry["command"], args=entry["args"],
                                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        async with (stdio_client(parameters) as (reader, writer),
                    ClientSession(reader, writer) as session):
            await session.initialize()
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == set(_tools("qcl"))

    asyncio.run(exercise())
