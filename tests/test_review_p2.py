"""Reviewer regression tests for P2 (written by Claude; implementers must not edit).

Uses the real installed framework CLIs where available (no model calls): Claude Code's
`claude plugin validate`, and Codex's marketplace/plugin install in an isolated CODEX_HOME.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BIN = Path(sys.executable).parent


def _build(target: str, tmp_path: Path) -> Path:
    from chemflow.signing import generate_keypair

    _, public_pem = generate_keypair(b"reviewer")
    key = tmp_path / "keys" / "public.pem"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(public_pem)
    out = tmp_path / "out" / target
    subprocess.run(args=[str(BIN / "chemflow"), "plugin", "build", "--target", target,
                    "--out", str(out), "--public-key", str(key)], check=True,
                   capture_output=True, text=True)
    return out


def test_claude_code_manifest_passes_the_real_validator(tmp_path) -> None:
    out = _build("claude-code", tmp_path)
    manifest = json.loads((out / ".claude-plugin" / "plugin.json").read_text())
    assert isinstance(manifest["author"], dict) and manifest["author"].get("name")
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(check=False, args=["claude", "plugin", "validate", str(out)], capture_output=True,
                            text=True, env={**os.environ, "HOME": str(home)}, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_codex_package_is_marketplace_rooted_and_installs(tmp_path) -> None:
    out = _build("codex", tmp_path)
    marketplace = json.loads((out / ".agents" / "plugins" / "marketplace.json").read_text())
    entry = marketplace["plugins"][0]
    plugin_root = (out / entry["source"]["path"]).resolve()
    assert (plugin_root / "plugin.json").is_file()
    assert (plugin_root / "mcp.json").is_file()
    assert not (plugin_root / ".agents").exists()
    plugin_manifest = json.loads((plugin_root / "plugin.json").read_text())
    assert isinstance(plugin_manifest["author"], dict)
    if shutil.which("codex") is None:
        pytest.skip("codex CLI not installed")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    env = {**os.environ, "CODEX_HOME": str(codex_home)}
    added = subprocess.run(check=False, args=["codex", "plugin", "marketplace", "add", str(out)],
                           capture_output=True, text=True, env=env, timeout=120)
    assert added.returncode == 0, added.stdout + added.stderr
    name = f"{entry['name']}@{marketplace['name']}"
    installed = subprocess.run(check=False, args=["codex", "plugin", "add", name], capture_output=True,
                               text=True, env=env, timeout=120)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    listing = subprocess.run(check=False, args=["codex", "plugin", "list"], capture_output=True, text=True,
                             env=env, timeout=120)
    assert "installed" in listing.stdout and "enabled" in listing.stdout


def _hook(event: dict, *args: str, home: Path) -> int:
    result = subprocess.run(check=False, args=[str(BIN / "chemflow-hook"), *args], input=json.dumps(event),
                            capture_output=True, text=True,
                            env={**os.environ, "HOME": str(home)}, timeout=60)
    return result.returncode


def test_hook_blocks_writes_to_protected_dirs_but_not_reads_or_unrelated_files(tmp_path) -> None:
    protected = tmp_path / "chemflow-state"
    protected.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    base = {"hook_event_name": "PreToolUse", "session_id": "s", "cwd": str(project)}
    flags = ("--protect", str(protected))

    def pre(tool: str, tool_input: dict) -> int:
        return _hook({**base, "tool_name": tool, "tool_input": tool_input}, *flags, home=home)

    assert pre("Read", {"file_path": str(project / "my_task.json")}) == 0
    assert pre("Read", {"file_path": str(protected / "task.json")}) == 0
    assert pre("Write", {"file_path": str(project / "my_task.json"), "content": "x"}) == 0
    assert pre("Write", {"file_path": str(protected / "task.json"), "content": "x"}) == 2
    assert pre("Bash", {"command": f"echo x > {protected / 'task.json'}"}) == 2


def test_role_subagents_have_meaningful_descriptions(tmp_path) -> None:
    out = _build("claude-code", tmp_path)
    for agent in (out / "agents").glob("*.md"):
        header = agent.read_text().split("---")[1]
        description = next(line.split(":", 1)[1].strip() for line in header.splitlines()
                           if line.startswith("description:"))
        assert len(description) >= 60, agent.name
