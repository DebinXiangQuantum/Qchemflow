import json

from test_m3a_task import spec

from chemflow.agent.qcl_tools import DESCRIPTIONS
from chemflow.cli import main
from chemflow.plugin_build import TARGETS, build
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def test_build_targets(tmp_path):
    private, public = generate_keypair(b"human")
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    task.write_text(TaskSigner(private, b"human").freeze(spec(), "human").model_dump_json())
    for target in TARGETS:
        output = tmp_path / target
        build(target, output, key, task=task)
        assert list(output.rglob("SKILL.md"))
        if target == "codex":
            manifest = json.loads((output / "plugins/chemflow/plugin.json").read_text())
            assert manifest["extensions"]["com.openai"]["hooks"] == "./hooks/hooks.json"
        if target == "claude-code":
            assert [path.name for path in (output / "agents").glob("*.md")] == ["qcl.md"]
        for path in output.rglob("*.json"):
            if path.name in ("mcp.json", ".mcp.json"):
                assert TaskVerifier(public).fingerprint in path.read_text()
    extension = (tmp_path / "pi" / "extensions/chemflow.ts").read_text()
    for tool in ("task_get", "methods_list", "screen_review", "screen_run"):
        assert f"name: '{tool}'" in extension
        name = next(name for name in DESCRIPTIONS if name.replace(".", "_") == tool)
        assert repr(DESCRIPTIONS[name]) in extension
    for profiles in ("designer", "qcl,designer"):
        try:
            build("pi", tmp_path / "rejected", key, task=task, profiles=profiles)
        except ValueError as error:
            assert "profile" in str(error)
        else:
            raise AssertionError(f"{profiles} accepted")


def test_activate_default_task(tmp_path, monkeypatch):
    private, public = generate_keypair(b"human")
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    private_path = key_dir / "private.pem"
    private_path.write_bytes(private)
    private_path.chmod(0o600)
    (key_dir / "public.pem").write_bytes(public)
    source = tmp_path / "spec.yaml"
    source.write_text(spec().model_dump_json())
    task = tmp_path / "task.json"
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("CHEMFLOW_KEY_PASSPHRASE", "human")
    monkeypatch.setattr("sys.argv", ["chemflow", "--key-dir", str(key_dir), "task", "freeze",
                        "--from", str(source), "--approver", "human", "--out", str(task),
                        "--activate"])
    main()
    assert (tmp_path / ".chemflow/tasks/current.json").read_text() == task.read_text()
    output = tmp_path / "package"
    build("agent-plugins", output, key_dir / "public.pem")
    args = json.loads((output / "mcp.json").read_text())["mcpServers"]["chemflow-qcl"]["args"]
    assert args[args.index("--task") + 1] == str(tmp_path / ".chemflow/tasks/current.json")
