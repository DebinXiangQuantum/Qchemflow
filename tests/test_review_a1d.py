"""Reviewer regression tests for A1d (written by Claude; implementers must not edit).

A1b showed the design ladder catching an overfitted design (L1 0.049 on one instance; 0.046,
1.314 and 0.715 on random ones). A1d closes the loop and grows the problem:
1. Larger baths: the independent exact reference (`domains.fock_reference`) covers impurity
   models up to 8 sites (7 bath sites) by sector-restricted exact diagonalisation with its own
   operator algebra (it must stay independent of the candidate primitives, FULL-2 R91).
2. A designer loop `chemflow.design.design_loop(program, target_node, *, target, registry, store,
   instances, seed, max_rounds, moves=None, register_as=None, ledger=None, task=None)`. It
   verifies a candidate at the bar declared in the program (A1c), diagnoses failures from the
   recorded reports, applies one documented redesign move per round (`chemflow.design.MOVES`),
   and stops when a candidate passes or nothing is left. It returns `.outcome` ("passed" |
   "exhausted"), `.trajectory` (one entry per candidate: round, report_id, passed, max_metric,
   diagnosis, move, resources) and `.design` (the registered experimental entry or None).
   Negative results are valid outputs.
3. A benchmark `chemflow bench run dynamics aim-design-bath5` running the loop on 6-site models.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import test_review_a1a as a1a
import test_review_a1b as a1b

from chemflow import build
from chemflow.runtime import Executor

BATH5 = {"U": 2.5, "eps_bath": [-1.1, -0.4, 0.0, 0.5, 1.2],
         "t_bath": [0.45, 0.6, 0.5, 0.55, 0.4], "unit": "hartree"}


def _target(model, n_alpha, n_beta):
    n_orb = len(model["eps_bath"]) + 1
    return {"kind": "spectral_function", "model": {"impurity_model": model},
            "sector": {"n_alpha": n_alpha, "n_beta": n_beta, "n_orb": n_orb}, "orbital": 0,
            "spin": "alpha", "omega": a1a.OMEGA, "eta": a1a.ETA}


# --- 1. larger baths: the independent reference --------------------------------------------------

def test_the_reference_covers_six_sites_and_agrees_with_the_reviewers_dense_oracle() -> None:
    from chemflow.domains.fock_reference import reference

    exact = reference(_target(BATH5, 3, 3), BATH5)
    _, omega, oracle = a1a._oracle(BATH5, 3, 3)  # reviewer's own full Fock space, 4096 states
    assert a1a._l1(omega, exact.spectral, oracle) < 1e-8


def test_the_reference_reaches_eight_sites_with_the_exact_ground_energy() -> None:
    from chemflow.domains import load_registry
    from chemflow.domains.fock_reference import MAX_ORBITALS, reference

    assert MAX_ORBITALS >= 8
    model = {"U": 3.0, "eps_bath": [-1.3, -0.8, -0.3, 0.0, 0.4, 0.9, 1.4],
             "t_bath": [0.35, 0.5, 0.45, 0.6, 0.5, 0.4, 0.3], "unit": "hartree"}
    exact = reference(_target(model, 4, 4), model)
    spectral = np.asarray(exact.spectral)
    omega = np.linspace(a1a.OMEGA["start"], a1a.OMEGA["stop"], a1a.OMEGA["n"])
    assert spectral.min() > -1e-12 and 0.9 < np.trapezoid(spectral, omega) <= 1.0 + 1e-9
    with build.program("ground8") as builder:  # PySCF FCI on the same model: independent code
        hamiltonian = builder.node("impurity_model", **model)
        sector = builder.sector(n_alpha=4, n_beta=4, n_orb=8)
        energy = builder.ground_energy(hamiltonian, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "fci_pyscf")
    fci = Executor(load_registry(a1a.PACKS)).run(builder.to_ir())[energy.id]
    assert exact.diagnostics["ground_energy"] == pytest.approx(
        fci.value.to("hartree").value, abs=1e-8)


# --- 2. the designer loop ---------------------------------------------------------------------

def _loop(program, response, **kwargs):
    from chemflow.design import DesignStore, design_loop

    store = DesignStore()
    registry = a1b._registry()
    defaults = {"target": a1b.TARGET, "instances": 3, "seed": 5, "max_rounds": 5}
    return design_loop(program, response, registry=registry, store=store,
                       **{**defaults, **kwargs}), store, registry


def test_the_loop_recovers_from_the_overfitted_design_and_records_why() -> None:
    from chemflow.design import MOVES

    program, response = a1a._enriched_plan()  # the A1b design that failed seed 5
    result, store, _ = _loop(a1a._accept(program, response), response, register_as="aim_loop")
    first = result.trajectory[0]
    assert not first["passed"] and first["move"] in MOVES and first["diagnosis"]
    assert result.outcome == "passed" and result.trajectory[-1]["passed"]
    assert result.design["id"] == "design:aim_loop"
    assert result.design["evidence"] == result.trajectory[-1]["report_id"]
    for entry in result.trajectory:
        report = store.get(entry["report_id"])  # every candidate is on record
        assert {key: report.accuracy[key] for key in a1a.ACCURACY} == a1a.ACCURACY
        assert entry["resources"]["sim_shots"] >= 0
    assert all(entry["move"] in MOVES for entry in result.trajectory[:-1])
    assert result.trajectory[-1]["move"] is None  # nothing applied after the passing round


def test_the_loop_is_deterministic_for_a_seed() -> None:
    program, response = a1a._enriched_plan()
    first, _, _ = _loop(a1a._accept(program, response), response, max_rounds=2)
    program, response = a1a._enriched_plan()
    second, _, _ = _loop(a1a._accept(program, response), response, max_rounds=2)
    assert [item["report_id"] for item in first.trajectory] == [
        item["report_id"] for item in second.trajectory]


def test_an_unreachable_bar_ends_exhausted_and_registers_nothing() -> None:
    from chemflow.design import MOVES

    program, response = a1a._enriched_plan()
    strict = {**a1a.ACCURACY, "threshold": 1e-12}
    sampled_only = [move for move in MOVES if "full" not in move]
    result, store, registry = _loop(a1a._accept(program, response, strict), response,
                                    moves=sampled_only, max_rounds=3, register_as="never")
    assert result.outcome == "exhausted" and result.design is None
    assert not any("design:never" in str(item) for item in registry.search("design:"))
    assert 1 <= len(result.trajectory) <= 3 and not any(e["passed"] for e in result.trajectory)
    assert all(store.get(entry["report_id"]).accuracy["threshold"] == 1e-12
               for entry in result.trajectory)


# --- 3. the benchmark ------------------------------------------------------------------------

def test_the_bath5_design_benchmark_runs_and_reports_its_trajectory(tmp_path) -> None:
    out = tmp_path / "bench.json"
    subprocess.run([str(Path(sys.executable).parent / "chemflow"), "bench", "run", "dynamics",
                    "aim-design-bath5", "--json", str(out)], check=True, capture_output=True,
                   text=True, timeout=1800)
    data = json.loads(out.read_text())
    assert data["outcome"] in ("passed", "exhausted") and data["trajectory"]
    assert data["n_sites"] == 6 and data["accuracy"]["threshold"] > 0
    for entry in data["trajectory"]:
        assert {"report_id", "passed", "max_metric", "move", "resources"} <= set(entry)
