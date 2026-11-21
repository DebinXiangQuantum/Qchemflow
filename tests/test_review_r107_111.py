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
