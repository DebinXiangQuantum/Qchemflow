"""Materialize framework adapters from the single-source plugin directory."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from chemflow.agent.tools import DESCRIPTIONS
from chemflow.mcp import PROFILES
from chemflow.plugin_constants import AGENT_MCP_SCHEMA, AGENT_PLUGIN_SCHEMA
from chemflow.signing import TaskVerifier

TARGETS = ("agent-plugins", "codex", "claude-code", "goose", "hermes", "pi")
SOURCE = Path(__file__).resolve().parents[2] / "plugin"
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent / "plugin"
VERSION = "0.1.0"


def _write(path: Path, value: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((json.dumps(value, indent=2) + "\n") if isinstance(value, dict) else value)


def _roles(profiles: str | None, target: str) -> list[str]:
    chosen = (profiles.split(",") if profiles else
              list(PROFILES[:-1]) if target == "claude-code" else ["designer"])
    if not chosen or len(chosen) != len(set(chosen)) or any(role not in PROFILES for role in chosen):
        raise ValueError("Invalid profile list")
    return chosen


def _tools(role: str) -> list[str]:
    if role == "full":
        return list(dict.fromkeys(tool for candidate in PROFILES[:-1] for tool in _tools(candidate)))
    source = (SOURCE / "agents" / f"{role}.md").read_text()
    match = re.search(r"Allowed MCP tools: (.*)", source)
    if match is None:
        raise ValueError(f"Missing role tools: {role}")
    return re.findall(r"`([^`]+)`", match.group(1))


def _servers(roles: list[str], task: Path, key: Path, fingerprint: str,
             command: Path, *, typed: bool = False, packs: str | None = None) -> dict:
    servers = {}
    for role in roles:
        entry = {"command": str(command), "args": ["--profile", role, "--task", str(task),
                 "--public-key", str(key), "--key-fingerprint", fingerprint]}
        if packs is not None:
            entry["args"].extend(["--packs", packs])
        if typed:
            entry = {"type": "stdio", **entry}
        servers[f"chemflow-{role}"] = entry
    return servers


def _hooks(command: Path, template: dict, *, protect: tuple[Path, Path], goose: bool = False,
           root: bool = False) -> dict:
    result = {}
    for event in ("PreToolUse", "PostToolUse", "Stop"):
        executable = "${CLAUDE_PLUGIN_ROOT}/hooks/chemflow-hook" if root else str(command)
        flags = " ".join(f"--protect {shlex.quote(str(directory))}" for directory in protect)
        hook = {"type": "command", "command": f"{executable} {flags}", "timeout": 30}
        handler = {"matcher": ".*", "hooks": [hook]}
        if goose and event == "PreToolUse":
            handler["on_failure"] = "block"
        result[event] = [handler]
    if not all(template.get(event) for event in result):
        raise ValueError("Incomplete hook template")
    return {"hooks": result}


def _copy_skills(destination: Path) -> None:
    shutil.copytree(SOURCE / "skills", destination, dirs_exist_ok=True)


def build(target: str, out: Path, public_key: Path, *, task: Path | None = None,
          profiles: str | None = None, server_command: Path | None = None,
          hook_command: Path | None = None, packs: str | None = None) -> None:
    if target not in TARGETS:
        raise ValueError("Unknown plugin target")
    if packs is not None:
        from chemflow.domains import get_pack

        for name in packs.split(","):
            get_pack(name.strip())
    key = public_key.expanduser().resolve(strict=True)
    fingerprint = TaskVerifier(key.read_bytes()).fingerprint
    task = (task or Path.home() / ".chemflow/tasks/current.json").expanduser().resolve()
    command = (server_command or Path(sys.executable).parent / "chemflow-mcp").resolve(strict=True)
    hook = (hook_command or Path(sys.executable).parent / "chemflow-hook").resolve(strict=True)
    protect = (task.parent, key.parent)
    if not command.is_file() or not hook.is_file():
        raise ValueError("Server and hook commands must be files")
    roles = _roles(profiles, target)
    template = json.loads((SOURCE / "hooks/hooks.template.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    name = "chemflow"
    description = "Task-bound molecular chemistry tools and integrity hooks"
    if target in ("agent-plugins", "codex"):
        plugin_root = out / "plugins/chemflow" if target == "codex" else out
        manifest = {"$schema": AGENT_PLUGIN_SCHEMA, "name": name, "version": VERSION,
                    "description": description, "author": {"name": "ChemFlow"}, "license": "MIT",
                    "keywords": ["chemistry", "quantum", "mcp"]}
        if target == "codex":
            manifest["extensions"] = {"com.openai": {
                "hooks": "./hooks/hooks.json", "interface": {
                    "displayName": "ChemFlow", "shortDescription": description,
                    "developerName": "ChemFlow", "category": "Science",
                    "capabilities": ["MCP tools", "skills", "hooks"]}}}
            _write(plugin_root / "AGENTS.md", (SOURCE / "rules/chemflow-rules.md").read_text())
            _write(out / ".agents/plugins/marketplace.json", {
                "name": name, "interface": {"displayName": "ChemFlow"}, "plugins": [{
                    "name": name, "source": {"source": "local", "path": "./plugins/chemflow"},
                    "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                    "category": "Science"}]})
        _write(plugin_root / "plugin.json", manifest)
        _write(plugin_root / "mcp.json", {"$schema": AGENT_MCP_SCHEMA, "mcpServers":
                _servers(roles, task, key, fingerprint, command, typed=True, packs=packs)})
        _copy_skills(plugin_root / "skills")
        _write(plugin_root / "hooks/hooks.json", _hooks(hook, template, protect=protect))
    elif target == "claude-code":
        _write(out / ".claude-plugin/plugin.json", {"name": name, "version": VERSION,
                "description": description, "author": {"name": "ChemFlow"}})
        _write(out / ".mcp.json", {"mcpServers": _servers(roles, task, key, fingerprint,
                                                       command, packs=packs)})
        _copy_skills(out / "skills")
        _write(out / "hooks/hooks.json", _hooks(hook, template, protect=protect))
        for role in roles:
            if role == "full":
                continue
            body = (SOURCE / "agents" / f"{role}.md").read_text()
            contract = " ".join(body.split("Allowed MCP tools:", 1)[0].splitlines()[2:]).strip()
            delegation = f"Delegate to the ChemFlow {role} when this role's scoped work is needed: {contract}"
            allow = [f"mcp__plugin_chemflow_chemflow-{role}__{tool}" for tool in _tools(role)]
            if any(len(tool) > 64 for tool in allow):
                raise ValueError("Claude tool name exceeds 64 characters")
            frontmatter = yaml.safe_dump({"name": f"chemflow-{role}",
                                          "description": delegation,
                                          "tools": allow}, sort_keys=False)
            _write(out / "agents" / f"{role}.md", f"---\n{frontmatter}---\n{body}")
    elif target == "goose":
        _copy_skills(out / ".agents/skills")
        _write(out / "AGENTS.md", (SOURCE / "rules/chemflow-rules.md").read_text())
        _write(out / ".goosehints", "Follow AGENTS.md for ChemFlow integrity rules.\n")
        _write(out / ".agents/plugins/chemflow/hooks/hooks.json",
                _hooks(hook, template, protect=protect, goose=True))
        extensions = {server: {"type": "stdio", "name": server, "display_name": server,
                      "enabled": True, "bundled": False, "cmd": entry["command"],
                      "args": entry["args"], "env_keys": [], "envs": {}, "timeout": 300}
                      for server, entry in _servers(roles, task, key, fingerprint,
                                                     command, packs=packs).items()}
        _write(out / "goose-config.yaml", yaml.safe_dump({"extensions": extensions}, sort_keys=False))
    elif target == "hermes":
        _copy_skills(out / "skills")
        _write(out / "AGENTS.md", (SOURCE / "rules/chemflow-rules.md").read_text())
        _write(out / "plugin.yaml", yaml.safe_dump({"name": name, "version": VERSION,
                "description": description}, sort_keys=False))
        _write(out / "hermes-config.yaml", yaml.safe_dump({
            "mcp_servers": _servers(roles, task, key, fingerprint, command, packs=packs),
            "hooks": {"pre_tool_call": [{"matcher": ".*", "command":
                       f"{hook} --protect {shlex.quote(str(task.parent))} --protect {shlex.quote(str(key.parent))}",
                       "timeout": 30, "fail_closed": True}],
                      "post_tool_call": [{"matcher": ".*", "command":
                       f"{hook} --protect {shlex.quote(str(task.parent))} --protect {shlex.quote(str(key.parent))}",
                       "timeout": 30}]}}, sort_keys=False))
        names = [path.name for path in sorted((SOURCE / "skills").iterdir()) if path.is_dir()]
        _write(out / "__init__.py", "from pathlib import Path\n\n"
               "from chemflow.hooks import evaluate\n\n\n"
               "def register(ctx):\n"
               "    root = Path(__file__).parent\n" + "".join(
                   f"    ctx.register_skill({skill!r}, str(root / 'skills' / {skill!r}))\n"
                   for skill in names) +
               "    def pre_tool_call(event):\n"
               "        try:\n"
               "            reason = evaluate(event)\n"
               "        except (OSError, ValueError, TypeError, KeyError):\n"
               "            reason = 'ChemFlow hook failed closed'\n"
               "        return {'action': 'block', 'message': reason} if reason else None\n"
               "    ctx.register_hook('pre_tool_call', pre_tool_call)\n")
    else:
        _copy_skills(out / "skills")
        _write(out / "package.json", {"name": "chemflow", "version": VERSION,
                "pi": {"extensions": ["extensions/chemflow.ts"], "skills": ["skills"]}})
        bridge = ["import { spawnSync } from 'node:child_process';",
                  "import { Type } from '@sinclair/typebox';", "import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';",
                  "", "export default function (pi: ExtensionAPI) {"]
        for role in roles:
            for tool in _tools(role):
                original = next(name for name in DESCRIPTIONS if name.replace(".", "_") == tool)
                exposed = f"chemflow_{role}_{tool}" if len(roles) > 1 else tool
                bridge.extend([f"  pi.registerTool({{ name: '{exposed}', description: {DESCRIPTIONS[original]!r},",
                               "    parameters: Type.Object({}, { additionalProperties: true }),",
                               "    async execute(_id, params) {",
                               (f"      const result = spawnSync({str(Path(sys.executable).parent / 'chemflow')!r}, "
                               f"['tool', 'call', '--profile', '{role}', '--task', {str(task)!r}, "
                               f"'--public-key', {str(key)!r}, '--key-fingerprint', {fingerprint!r}, "
                               + (f"'--packs', {packs!r}, " if packs is not None else "") +
                               f"'--name', {original!r}, '--args-json', JSON.stringify(params)], "
                               "{ encoding: 'utf8' });"),
                               "      if (result.status !== 0) throw new Error(result.stdout || result.stderr);",
                               "      return { content: [{ type: 'text', text: result.stdout }] };",
                               "    } });"])
        bridge.extend(["  pi.on('tool_call', async (event, ctx) => {",
                       "    const result = spawnSync(" + repr(str(hook)) + ", " +
                       repr(['--protect', str(task.parent), '--protect', str(key.parent)]) + ", {",
                       "      input: JSON.stringify({ hook_event_name: 'PreToolUse', tool_name: event.toolName,",
                       "        tool_input: event.input, cwd: ctx.cwd }), encoding: 'utf8' });",
                       "    if (result.status === 2) return { block: true, reason: result.stderr.trim() };",
                       "    if (result.status !== 0) return { block: true, reason: 'ChemFlow hook failed closed' };",
                       "  });", "}"])
        _write(out / "extensions/chemflow.ts", "\n".join(bridge) + "\n")
