"""The qcl MCP profile: externally driven agents read, compile and run signed screens."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chemflow.agent.task import TaskSpec
from chemflow.mcp import QCL_PROFILE, create_tools
from chemflow.signing import TaskSigner, generate_keypair

pytest.importorskip("qcl")

SPEC = {"question": "H2 ground energy", "target": {
    "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"}, "electronic_model": {"basis": "sto-3g"},
    "active": {"n_elec": 2, "n_orb": 2}, "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
    "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"},
              {"kind": "relation_at_least", "relation": "variational_upper_bound"}],
    "budget": {"wall_s": 600, "sim_shots": 4000},
    "allowed_effects": ["pure", "simulator"], "decision": "Choose a method"}


def _signed(tmp_path):
    private, public = generate_keypair(b"human")
    record = TaskSigner(private, b"human").freeze(TaskSpec.model_validate(SPEC), "human")
    task, key = tmp_path / "task.json", tmp_path / "public.pem"
    task.write_text(record.model_dump_json())
    key.write_bytes(public)
    return record, task, key


def test_qcl_profile_tools_review_and_run_screens(tmp_path):
    _, task, key = _signed(tmp_path)
    tools = create_tools(QCL_PROFILE, str(task), public_key=str(key))
    names = {item["name"] for item in tools.schemas("orchestrator")}
    assert names == {"task.get", "methods.list", "diagnose.model", "screen.review",
                     "screen.run", "results.query"}
    assert tools.call("orchestrator", "task.get")["data"]["question"] == "H2 ground energy"
    candidates = [{"template": "sci", "select_cutoff": 1e-3}]
    reviewed = tools.call("orchestrator", "screen.review", {"candidates": candidates})["data"]
    assert reviewed["relations"] == {"candidate0": "variational_upper_bound"}
    assert reviewed["sim_shots"] == 0 and reviewed["affordable"]

    first = tools.call("orchestrator", "screen.run", {"candidates": candidates})["data"]
    assert first["decision"] == "escalate" and first["candidate"] == "candidate0"
    assert "## Methods" in first["report"]
    assert any(item.get("screening_reference") for item in first["results"])
    # The scripted rounds sample 1024 + 2048 simulator shots; the signed 4000 allows one
    # such run, and the ledger shared by the session refuses the next.
    second = tools.call("orchestrator", "screen.run", {})["data"]
    assert second["spent"]["sim_shots"] == 3072
    third = tools.call("orchestrator", "screen.run", {})["data"]
    assert third["decision"] == "refused" and third["refusal"]["code"] == "CF-E070"


def test_qcl_profile_through_the_tool_cli(tmp_path):
    _, task, key = _signed(tmp_path)
    from chemflow.signing import TaskVerifier

    fingerprint = TaskVerifier(key.read_bytes()).fingerprint
    completed = subprocess.run(
        [str(Path(sys.executable).parent / "chemflow"), "tool", "call", "--profile", "qcl",
         "--task", str(task), "--public-key", str(key), "--key-fingerprint", fingerprint,
         "--name", "methods_list", "--args-json", "{}"],
        check=False, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)["data"]
    assert [item["method"] for item in data["controls"]] == ["hf", "mp2", "ccsd", "ccsd_t"]
    assert data["reference"] == "fci"
