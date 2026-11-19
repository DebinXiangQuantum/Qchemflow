"""Reviewer regression tests for R107-R111 (written by Claude; implementers must not edit).

Found by the 55-task `full` benchmark and the first live GLM runs (2026-10-13):
- R107: open-shell targets (triplet H2, doublet Li, doublet H5, triplet NH) failed outright.
  The approximate controls raised CF-E020 ("cannot constrain spin"), and one control's refusal
  failed the whole run.
  - High-spin targets (n_alpha − n_beta = 2S) run the controls on an open-shell reference,
    report ⟨S²⟩, and are labelled approximate when contaminated.
  - A control that genuinely cannot run is skipped, never fatal.
- R108: He/STO-3G has no virtual orbitals. CCSD(T) divided by zero (CF-E080 for the whole
  run). With no virtuals, every correlated control equals HF exactly.
- R109: a U=8 Hubbard ring's mean-field reference did not converge and failed the run. The
  lattice SCF must be robust (damping/level shift), and an unconverged control is skipped.
- R110: `estimate_cost` raised TypeError on any plan with candidates, so designers could not
  use it.
- R111: LLM_TOOL journal entries recorded `calls: null`; the call's name and arguments must
  be journalled (the key stays redacted).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import test_review_m4a as m4a

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")
FAILED_BEFORE = ("h2-triplet", "li-doublet", "h5-doublet", "nh-triplet", "he-atom",
                 "hubbard-ring4-u8-hartree")


def test_r107_r109_previously_failing_full_suite_tasks_are_answered(tmp_path) -> None:
    completed = subprocess.run([CHEMFLOW, "eval", "run", "--suite", "full", "--policy",
                                "scripted", "--jobs", "3", "--tasks", ",".join(FAILED_BEFORE),
                                "--out", str(tmp_path / "out")],
                               check=False, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stderr[-2000:]
    rows = {row["id"]: row for row in
            json.loads((tmp_path / "out" / "results.json").read_text())["rows"]}
    assert set(rows) == set(FAILED_BEFORE)
    for task, row in rows.items():
        assert row["completed"] and row["answer_node"], (task, row["decision"])
        assert row["correct"], (task, row["answer_value"])


def test_r108_correlated_controls_equal_hf_without_virtual_orbitals() -> None:
    from pyscf import gto, scf

    from chemflow import build
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    with build.program("he") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("He 0 0 0"),
                                          builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=1)
        nodes = {}
        for method in ("hf", "mp2", "ccsd", "ccsd_t"):
            nodes[method] = builder.ground_energy(hamiltonian, sector, s=0)
            builder.refine(nodes[method], method)
            builder.bind(nodes[method], "hf" if method == "hf" else method)
    computed = Executor(molecular_registry()).run(builder.to_ir())
    reference = scf.RHF(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)).run().e_tot
    for method, node in nodes.items():
        assert abs(computed[node.id].value.to("hartree").value - reference) < 1e-10, method


def test_r110_cost_estimates_work_on_plans_with_candidates() -> None:
    tools = m4a._tools()
    plan_id = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": plan_id,
                                               "active": {"n_elec": 2, "n_orb": 2}})["data"]
    candidates = tools.call("designer", "plan.candidates", {
        "plan_id": base["plan_id"], "candidates": [{"template": "sqd", "shots": 1024}]})["data"]
    estimates = tools.call("designer", "estimate_cost", {"plan_id": candidates["plan_id"]})["data"]
    assert estimates and all("card" in item for item in estimates.values())


def test_r111_tool_calls_are_journalled_with_their_arguments(tmp_path) -> None:
    from chemflow.agent.journal import Journal
    from chemflow.agent.llm import FakeLLMClient, LLMPolicy

    call = {"id": "c1", "name": "cards.search", "arguments": {"query": "sqd"}}
    client = FakeLLMClient([{"tool_calls": [call], "content": "", "stop_reason": "tool_calls",
                             "usage": {}},
                            {"tool_calls": [], "content": "done", "stop_reason": "stop",
                             "usage": {}}])
    LLMPolicy("designer", client, m4a._tools(), journal=Journal(tmp_path / "j.jsonl")).decide({})
    entries = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines()]
    tool = next(entry for entry in entries if entry["state"] == "LLM_TOOL")
    recorded = tool["calls"][0]
    assert recorded["name"] == "cards.search" and recorded["arguments"] == {"query": "sqd"}
