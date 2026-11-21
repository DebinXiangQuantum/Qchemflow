"""Reviewer tests for V1d, machine-checked lemmas (written by Claude; implementers must not edit).

User decision (2026-10-14): a Python proof kernel now (V1a), with Lean proofs of the lemma
table later. On 2026-10-15 the user said "yes start V1d".

The Lean project is `formal/`:
- Lean 4.15.0 and Mathlib v4.15.0, pinned in formal/lake-manifest.json;
- packages APFS-cloned from the user's quantum-advantage-lab build, so it builds offline;
- `.lake/` is git-ignored.

Contract:
- Every `LEMMAS` entry gains:
  - `lean_status`: "machine_checked", "definitional" or "cited";
  - `lean`: the fully qualified Lean theorem name, or None;
  - `lean_note`: a non-empty justification when the status is not machine_checked.
- These must be machine_checked, with Lean statements that faithfully formalise the Python
  statement and side conditions (Hermitian/self-adjoint operators on finite-dimensional inner
  product spaces):
  - exact_diagonalization
  - rayleigh_ritz
  - rayleigh_ritz_full
  - variational_principle
  - sector_decomposition (the invariance part: an operator commuting with N maps N's
    eigenspaces into themselves)
  - positive_linear_combination
  - linear_combination_exact
  - lehmann_completeness
- If the Lean proof of rayleigh_ritz covers only the lowest root, the excited-root claim is a
  separate entry `rayleigh_ritz_excited` with status "cited" or "machine_checked", and the
  prover cites it for k > 0.
- `weinstein_bound`, used by V1c's certified bounds, is a LEMMAS entry and is machine_checked:
  for Hermitian H and unit ψ, some eigenvalue lies within σ = sqrt(⟨H²⟩ − ⟨H⟩²) of ⟨H⟩.
- formal/ChemflowFormal contains no `sorry`, `admit`, `axiom` declarations, `native_decide`
  or `implemented_by`.
- `formal/manifest.json` is produced by `scripts/formal_audit.py`. It holds the toolchain,
  the Mathlib revision, the SHA-256 of every .lean source, and per theorem its name, file and
  `#print axioms` result. The axioms must be within {propext, Classical.choice, Quot.sound}.
  The manifest must match the current sources and the LEMMAS table.
- `check_certificate` verdicts gain `machine_checked` (the lemmas the certificate used that
  are machine-checked) and `trusted` (the cited or definitional lemmas, plus the
  implementation axioms it relies on). A certificate's trust base is therefore explicit.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "formal"
CORE = {"exact_diagonalization", "rayleigh_ritz", "rayleigh_ritz_full", "variational_principle",
        "sector_decomposition", "positive_linear_combination", "linear_combination_exact",
        "lehmann_completeness", "weinstein_bound"}
STANDARD = {"propext", "Classical.choice", "Quot.sound"}


def _manifest():
    return json.loads((FORMAL / "manifest.json").read_text())


def test_v1d_formal_sources_contain_no_escape_hatches() -> None:
    sources = list((FORMAL / "ChemflowFormal").rglob("*.lean"))
    assert sources
    for path in sources:
        text = path.read_text()
        for word in ("sorry", "admit", "native_decide", "implemented_by"):
            assert word not in text, (path.name, word)
        assert not any(line.lstrip().startswith("axiom ") for line in text.splitlines()), path


@pytest.mark.skipif(shutil.which("lake") is None, reason="Lean toolchain not installed")
def test_v1d_lean_builds_and_the_axiom_audit_is_clean(tmp_path) -> None:
    built = subprocess.run(["lake", "build"], cwd=FORMAL, capture_output=True, text=True, check=False,
                           timeout=1800)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    names = [item["name"] for item in _manifest()["theorems"]]
    audit = FORMAL / "ReviewerAudit.lean"
    audit.write_text("import ChemflowFormal\n" +
                     "".join(f"#print axioms {name}\n" for name in names))
    try:
        result = subprocess.run(["lake", "env", "lean", audit.name], cwd=FORMAL,
                                capture_output=True, text=True, timeout=1800, check=False)
    finally:
        audit.unlink()
    assert result.returncode == 0, result.stderr[-3000:]
    assert "sorryAx" not in result.stdout
    for line in result.stdout.splitlines():
        if "depends on axioms" in line:
            used = {item.strip() for item in line.split("[", 1)[1].rstrip("]").split(",")}
            assert used <= STANDARD, line
