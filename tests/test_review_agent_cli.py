"""Reviewer regression tests for `chemflow agent run` (written by Claude; implementers must not
edit).

The CLI gap: the agent loop existed only as a Python API and examples. Contract: `chemflow
agent run --task <signed task JSON> --public-key <PEM> --out <dir> [--policy scripted|anthropic|
openai] [--model M] [--roles designer,...] [--rounds N]` verifies the signed task, runs the
orchestrator on it without re-signing, and writes report.md, journal.jsonl and summary.json
(decision, results, record hash, budget spent) into --out, with a task ledger there that
accumulates across reruns. Live LLM policies are opt-in (CHEMFLOW_LIVE_LLM=1 plus provider
credentials) and never fall back silently to the scripted policy.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from chemflow.agent.task import TaskSpec
from chemflow.signing import TaskSigner, generate_keypair

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")
SPEC = {
    "question": "Which screened method meets the H2 singlet accuracy gate?",
    "target": {"operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
               "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 2, "n_orb": 2},
               "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 100000},
    "allowed_effects": ["pure", "simulator"],
    "decision": "choose classical control or request a larger screen"}


def _signed(tmp_path, spec=SPEC, passphrase=b"human"):
    private, public = generate_keypair(passphrase)
    record = TaskSigner(private, passphrase).freeze(TaskSpec.model_validate(spec), "human")
    task = tmp_path / "task.json"
    task.write_text(record.model_dump_json())
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    return task, key, record


def _agent(*arguments, env=None):
    import os

    return subprocess.run([CHEMFLOW, "agent", "run", *arguments], check=False,
                          capture_output=True, text=True, env={**os.environ, **(env or {})},
                          timeout=900)


def test_a_signed_task_runs_end_to_end_with_the_scripted_policy(tmp_path) -> None:
    task, key, record = _signed(tmp_path)
    out = tmp_path / "run"
    completed = _agent("--task", str(task), "--public-key", str(key), "--out", str(out))
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["record_hash"] == record.hash  # the signed task, not a re-signed copy
    assert summary["decision"] in ("stop", "revise", "escalate")  # the critic's decisions
    assert summary["results"] and (out / "report.md").read_text().strip()
    journal = [json.loads(line) for line in (out / "journal.jsonl").read_text().splitlines()]
    assert journal[0]["state"] == "INTAKE"
    assert summary["decision"] in completed.stdout and str(out / "report.md") in completed.stdout
    assert any(path.name.endswith(".ledger.json") for path in out.iterdir())


def test_unsigned_or_forged_tasks_are_refused(tmp_path) -> None:
    task, _, _ = _signed(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _, other_key, _ = _signed(other, passphrase=b"someone else")  # a key that did not sign it
    out = tmp_path / "forged"
    completed = _agent("--task", str(task), "--public-key", str(other_key), "--out", str(out))
    assert completed.returncode != 0 and "CF-E120" in completed.stderr
    assert not (out / "report.md").exists()


def test_live_llm_policies_are_opt_in_and_never_fall_back(tmp_path) -> None:
    task, key, _ = _signed(tmp_path)
    out = tmp_path / "live"
    completed = _agent("--task", str(task), "--public-key", str(key), "--out", str(out),
                       "--policy", "anthropic", env={"CHEMFLOW_LIVE_LLM": "0"})
    assert completed.returncode != 0
    assert "CHEMFLOW_LIVE_LLM" in completed.stderr
    assert not (out / "report.md").exists()
