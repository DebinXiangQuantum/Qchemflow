"""Reviewer regression tests for the `full` benchmark suite (written by Claude; implementers
must not edit).

User request 2026-10-13: benchmark ChemFlow thoroughly with many tasks; the user's GLM key is
for live runs. The `core` suite (11 tasks) checks the harness. `full` measures the agent across
every domain:
- At least 50 tasks: molecular ≥ 12, reaction ≥ 6, thermo ≥ 5, lattice ≥ 6, spectral ≥ 4,
  interaction ≥ 4, selection ≥ 6, refusal tasks ≥ 4.
- Each task has a `difficulty` tag (easy | medium | hard), and the suite has at least 10 hard
  tasks: stretched bonds, open shells, strong correlation, larger active spaces within the
  10-orbital screening cap.
- The refusal tasks are refusals the agent must produce at run time (budgets, effects not
  allowed by the task), not schema errors.
- Expected values come from independent routes, recorded in `reference_source`. Accuracy bars
  are the tasks' own gates (A1c).
- The scripted run of the whole suite must finish in under 30 minutes on this Mac.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")


def _suite():
    from importlib.resources import files

    import yaml

    return yaml.safe_load(files("chemflow.eval.suites").joinpath("full.yaml").read_text())


def test_the_full_suite_is_large_broad_and_hard_enough() -> None:
    from chemflow.agent.task import TaskSpec

    tasks = _suite()["tasks"]
    counts = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
        TaskSpec.model_validate(task["spec"])
        assert task["difficulty"] in ("easy", "medium", "hard") and task["reference_source"]
    assert len(tasks) >= 50 and len({task["id"] for task in tasks}) == len(tasks)
    minimum = {"molecular": 12, "reaction": 6, "thermo": 5, "lattice": 6, "spectral": 4,
               "interaction": 4, "selection": 6}
    assert all(counts.get(name, 0) >= count for name, count in minimum.items()), counts
    assert sum(task["expected"].get("outcome") == "refused" for task in tasks) >= 4
    assert sum(task["difficulty"] == "hard" for task in tasks) >= 10


def test_full_space_molecular_expectations_match_the_reviewers_pyscf_fci() -> None:
    from pyscf import fci, gto, scf

    checked = 0
    for task in _suite()["tasks"]:
        target = task["spec"]["target"]
        if task["category"] != "molecular" or task["expected"].get("outcome") == "refused":
            continue
        molecule = target["molecule"]
        mol = gto.M(atom=molecule["atoms"], basis=target["electronic_model"]["basis"],
                    charge=molecule.get("charge", 0), spin=molecule.get("spin", 0), verbose=0)
        active = target.get("active")
        if not isinstance(active, dict) or active.get("n_orb") != mol.nao or \
                active.get("n_elec") != mol.nelectron:
            continue  # only unambiguous full-space tasks are rechecked here
        mf = (scf.RHF if mol.spin == 0 else scf.ROHF)(mol).run()
        solver = fci.FCI(mf)
        solver.spin = mol.spin
        energy = solver.kernel()[0]
        assert task["expected"]["value_hartree"] == pytest.approx(energy, abs=1e-7), task["id"]
        checked += 1
    assert checked >= 4


def _hubbard_fci(lattice, n_alpha, n_beta):
    from pyscf import fci

    size, t, u = lattice["n_sites"], lattice["t"], lattice["U"]
    h1 = np.zeros((size, size))
    for site in range(size - (0 if lattice["geometry"] == "ring" else 1)):
        h1[site, (site + 1) % size] = h1[(site + 1) % size, site] = -t
    eri = np.zeros((size,) * 4)
    for site in range(size):
        eri[site, site, site, site] = u
    energy, _ = fci.direct_spin1.FCI().kernel(h1, eri, size, (n_alpha, n_beta))
    return energy


def test_lattice_expectations_match_the_reviewers_hubbard_fci() -> None:
    from scipy import constants as sc

    hartree_ev = sc.physical_constants["Hartree energy in eV"][0]
    checked = 0
    for task in _suite()["tasks"]:
        target = task["spec"]["target"]
        if task["category"] != "lattice" or target.get("lattice", {}).get("model") != "hubbard":
            continue
        lattice, sector = target["lattice"], target["sector"]
        energy = _hubbard_fci(lattice, sector["n_alpha"], sector["n_beta"])
        scale = 1 / hartree_ev if lattice.get("unit", "hartree") == "eV" else 1.0
        assert task["expected"]["value_hartree"] == pytest.approx(energy * scale, abs=1e-7), \
            task["id"]
        checked += 1
    assert checked >= 4


def test_one_task_per_category_runs_through_the_harness(tmp_path) -> None:
    tasks = _suite()["tasks"]
    chosen, seen = [], set()
    for task in tasks:
        key = "refused" if task["expected"].get("outcome") == "refused" else task["category"]
        if key not in seen and task["difficulty"] != "hard":
            seen.add(key)
            chosen.append(task["id"])
    completed = subprocess.run([CHEMFLOW, "eval", "run", "--suite", "full", "--policy", "scripted",
                                "--tasks", ",".join(chosen), "--out", str(tmp_path / "out")],
                               check=False, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stderr[-2000:]
    rows = json.loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert {row["id"] for row in rows} == set(chosen)
    assert all(row["reference_correct"] or row["decision"] == "refused" for row in rows)
