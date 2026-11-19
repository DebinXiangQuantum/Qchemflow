"""Reviewer regression tests for P1 (written by Claude; implementers must not edit).

Threat model: in Claude Code / Codex / Goose / Hermes the agent has a shell running as the same
OS user as the tool server. Anything the server can read, the agent can read. Verification must
therefore need only public material; signing keys are passphrase-encrypted and held by humans.
"""
import inspect

import pytest

pytest.importorskip("cryptography")

from chemflow.agent.task import TaskSpec
from chemflow.diagnostics import CheckError

SPEC = {
    "question": "q", "target": {
        "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
        "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
    "budget": {"wall_s": 60}, "allowed_effects": ["pure"], "decision": "d"}


def _keys(passphrase: bytes = b"human passphrase"):
    from chemflow.signing import generate_keypair

    return generate_keypair(passphrase)


def test_task_records_verify_with_public_key_only() -> None:
    from chemflow.signing import TaskSigner, TaskVerifier

    private_pem, public_pem = _keys()
    record = TaskSigner(private_pem, b"human passphrase").freeze(
        TaskSpec.model_validate(SPEC), "alice")
    TaskVerifier(public_pem).verify(record)


def test_agent_with_read_access_cannot_forge() -> None:
    from chemflow.signing import TaskSigner, TaskVerifier

    private_pem, public_pem = _keys()
    # The agent can read the encrypted private key file but not the passphrase.
    with pytest.raises((TypeError, ValueError)):
        TaskSigner(private_pem, b"wrong guess")
    # Signing with its own key does not produce a record the server accepts.
    attacker_private, _ = _keys(b"attacker")
    # D-R2: budgets name resources only (amended by the reviewer, intent unchanged).
    loose = TaskSpec.model_validate({**SPEC, "budget": {"wall_s": 1e9, "qpu_s": 1e9},
                                     "allowed_effects": ["pure", "simulator"]})
    forged = TaskSigner(attacker_private, b"attacker").freeze(loose, "pretend-human")
    with pytest.raises(CheckError) as info:
        TaskVerifier(public_pem).verify(forged)
    assert info.value.diagnostics[0].code == "CF-E120"
    # Editing a genuine record's gates breaks the signature.
    genuine = TaskSigner(private_pem, b"human passphrase").freeze(
        TaskSpec.model_validate(SPEC), "alice")
    edited = genuine.model_copy(update={"spec": loose})
    with pytest.raises(CheckError):
        TaskVerifier(public_pem).verify(edited)


def test_tool_server_needs_no_secret(monkeypatch, tmp_path) -> None:
    pytest.importorskip("mcp")
    from chemflow.mcp import create_server
    from chemflow.signing import TaskSigner

    parameters = set(inspect.signature(create_server).parameters)
    assert not {"secret", "secret_file", "passphrase"} & parameters
    assert "public_key" in parameters
    monkeypatch.delenv("CHEMFLOW_TASK_SECRET", raising=False)
    private_pem, public_pem = _keys()
    record = TaskSigner(private_pem, b"human passphrase").freeze(
        TaskSpec.model_validate(SPEC), "alice")
    task = tmp_path / "task.json"
    task.write_text(record.model_dump_json())
    key = tmp_path / "human.pub"
    key.write_bytes(public_pem)
    assert create_server("designer", str(task), public_key=str(key)) is not None


def test_run_schema_has_no_approval_gate() -> None:
    """Revised for R1: QPU use is budgeted like any device; no approvals exist."""
    from chemflow.agent.task import TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    store = TaskStore("legacy-test-store")
    record = store.freeze(TaskSpec.model_validate(SPEC), "alice")
    tools = make_tools(molecular_registry(), Executor(molecular_registry()), record, store)
    run = next(item for item in tools.schemas("designer") if item["name"] == "run")
    properties = run["parameters"]["properties"]
    assert "approval_id" not in properties and "approval" not in properties
