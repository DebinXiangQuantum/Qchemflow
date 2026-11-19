"""Reviewer regression tests for G2, fast classical baselines (written by Claude; implementers
must not edit).

From the ChemGraph comparison (user priority 2026-10-13): GFN2-xTB (tblite) and a
machine-learned potential (MACE-MP-0 small) as cheap baselines for screening and pre-optimising
geometries. Contract, pack `baselines`:
- operator `potential_energy` (input molecule; param method "gfn2-xtb" | "mace-mp-0-small")
  returns an Estimate of kind `potential_energy` in hartree. It is a model_change with respect
  to any electronic-structure target; its model protocol is the method, so it never mixes with
  ab initio energies (CF-E011).
- `geometry_optimize` and `frequencies` accept level {"method": "gfn2-xtb" | "mace-mp-0-small"}
  (ASE optimiser; finite-difference Hessian; most-abundant-isotope masses, as in G1). The
  frequency level is recorded in thermochemistry contexts.
- A missing package or model file is refused (CF-E080), with no fallback. MACE weights come from
  `CHEMFLOW_MACE_MODEL`, and their SHA-256 is recorded.
Oracles call tblite and MACE directly, by their native call paths.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy import constants as sc

from chemflow import build
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.runtime import Executor

H2O = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
BOHR = sc.physical_constants["Bohr radius"][0] * 1e10  # angstrom
MASS = {"H": 1.00782503223, "O": 15.99491461957}


def _registry():
    from chemflow.domains import load_registry

    return load_registry(["molecular", "reactions", "baselines"])


def _xyz(text):
    entries = [item.split() for item in text.split(";") if item.strip()]
    return [item[0] for item in entries], np.array([[float(x) for x in item[1:4]]
                                                    for item in entries])


def _tblite(symbols, positions_angstrom):
    from tblite.interface import Calculator

    numbers = np.array([{"H": 1, "O": 8}[symbol] for symbol in symbols])
    calculator = Calculator("GFN2-xTB", numbers, positions_angstrom / BOHR)
    calculator.set("verbosity", 0)
    return calculator.singlepoint()


def _energy_program(method, atoms=H2O):
    with build.program("baseline") as builder:
        energy = builder.node("potential_energy", molecule=builder.molecule(atoms), method=method)
    return builder.to_ir(), energy.id


def test_xtb_energy_matches_tblite_and_is_labelled_a_model_change() -> None:
    pytest.importorskip("tblite")
    program, node = _energy_program("gfn2-xtb")
    estimate = Executor(_registry()).run(program)[node]
    oracle = float(_tblite(*_xyz(H2O)).get("energy"))
    assert estimate.value.to("hartree").value == pytest.approx(oracle, abs=1e-7)
    assert estimate.relation == "model_change" and estimate.kind == "potential_energy"
    assert "tblite" in estimate.provenance.device


def test_baseline_energies_never_mix_with_electronic_structure_energies() -> None:
    from chemflow.check import check

    with build.program("mixed") as builder:
        molecule = builder.molecule(H2O)
        xtb = builder.node("potential_energy", molecule=molecule, method="gfn2-xtb")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        fci = builder.ground_energy(hamiltonian, builder.sector(n_alpha=5, n_beta=5, n_orb=7))
        builder.refine(fci, "fci")
        builder.node("reaction_energy", species0=xtb, species1=fci,
                     coefficients={"species0": 1, "species1": -1})
    issues = check(builder.to_ir(), _registry())
    assert any(item.code == "CF-E011" for item in issues), [item.message for item in issues]


def test_xtb_geometry_optimisation_reaches_a_stationary_point() -> None:
    pytest.importorskip("tblite")
    start = "O 0 0 0.10; H 0 0.80 -0.50; H 0 -0.70 -0.45"
    with build.program("opt") as builder:
        model = builder.electronic_model(basis="sto-3g")
        optimised = builder.node("geometry_optimize", molecule=builder.molecule(start),
                                 model=model, kind="minimum", level={"method": "gfn2-xtb"})
    result = Executor(_registry()).run(builder.to_ir())[optimised.id]
    final = _tblite(*_xyz(result["atoms"]))
    assert np.abs(final.get("gradient")).max() < 1e-3
    assert float(final.get("energy")) < float(_tblite(*_xyz(start)).get("energy"))


def _fd_frequencies(text, step=0.005):
    """Reviewer's own finite-difference Hessian from native tblite gradients (bohr units)."""
    symbols, positions = _xyz(text)
    coordinates = (positions / BOHR).ravel()
    hessian = np.zeros((coordinates.size, coordinates.size))
    for index in range(coordinates.size):
        shifted = []
        for sign in (1, -1):
            moved = coordinates.copy()
            moved[index] += sign * step
            shifted.append(_tblite(symbols, moved.reshape(-1, 3) * BOHR).get("gradient").ravel())
        hessian[index] = (shifted[0] - shifted[1]) / (2 * step)
    hessian = 0.5 * (hessian + hessian.T)
    masses = np.repeat([MASS[symbol] for symbol in symbols], 3)
    weighted = hessian / np.sqrt(np.outer(masses, masses))
    eigenvalues = np.linalg.eigvalsh(weighted)  # hartree / (bohr^2 amu)
    hartree = sc.physical_constants["Hartree energy"][0]
    bohr = sc.physical_constants["Bohr radius"][0]
    si = np.clip(eigenvalues, 0, None) * hartree / (bohr ** 2 * sc.atomic_mass)
    return np.sort(np.sqrt(si) / (2 * np.pi * sc.c) / 100)[-3:]  # three water modes, cm^-1


def test_xtb_frequencies_match_a_finite_difference_hessian_and_feed_thermochemistry() -> None:
    pytest.importorskip("tblite")
    with build.program("xtb-thermo") as builder:
        model = builder.electronic_model(basis="sto-3g")
        level = {"method": "gfn2-xtb"}
        geometry = builder.node("geometry_optimize", molecule=builder.molecule(H2O), model=model,
                                kind="minimum", level=level)
        frequencies = builder.node("frequencies", molecule=geometry, model=model, level=level)
        thermo = builder.node("thermochemistry", molecule=geometry, model=model,
                              frequencies=frequencies, temperature_K=298.15,
                              pressure_Pa=101325.0)
    computed = Executor(_registry()).run(builder.to_ir())
    modes = sorted(value for value in computed[frequencies.id]["freq_wavenumber"] if value > 0)
    oracle = _fd_frequencies(computed[geometry.id]["atoms"])
    assert len(modes) == 3 and np.allclose(modes, oracle, atol=2.0), (modes, oracle)
    assert computed[thermo.id]["frequency_level"] == "gfn2-xtb"


def test_a_missing_package_is_refused_not_replaced(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tblite", None)
    monkeypatch.setitem(sys.modules, "tblite.interface", None)
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run(_energy_program("gfn2-xtb")[0])
    assert "CF-E080" in str(info.value)


def test_mace_needs_its_declared_model_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHEMFLOW_MACE_MODEL", str(tmp_path / "absent.model"))
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(_registry()).run(_energy_program("mace-mp-0-small")[0])
    assert "CF-E080" in str(info.value)


def test_mace_energy_matches_the_direct_calculator() -> None:
    import hashlib
    import os
    from pathlib import Path

    pytest.importorskip("mace")
    model = Path(os.environ.get("CHEMFLOW_MACE_MODEL",
                                "/Volumes/macsoftware/caches/chemflow/mace/mace-mp-0-small.model"))
    if not model.is_file():
        pytest.skip("MACE-MP-0 small weights not installed")
    import subprocess
    import textwrap

    # The direct oracle runs in its own process: torch must not load into this one (R104).
    symbols, positions = _xyz(H2O)
    oracle_script = textwrap.dedent(f"""
        from ase import Atoms
        from mace.calculators import mace_mp
        atoms = Atoms({"".join(symbols)!r}, positions={positions.tolist()!r})
        atoms.calc = mace_mp(model={str(model)!r}, default_dtype="float64", device="cpu")
        print(repr(atoms.get_potential_energy()))
    """)
    completed = subprocess.run([sys.executable, "-c", oracle_script], capture_output=True,
                               text=True, timeout=600, check=True)
    oracle = float(completed.stdout.strip().splitlines()[-1]) / sc.physical_constants[
        "Hartree energy in eV"][0]
    program, node = _energy_program("mace-mp-0-small")
    estimate = Executor(_registry()).run(program)[node]
    assert estimate.value.to("hartree").value == pytest.approx(oracle, abs=1e-8)
    assert estimate.diagnostics["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert estimate.relation == "model_change"


@pytest.mark.parametrize("order", [("gfn2-xtb", "mace-mp-0-small"), ("mace-mp-0-small", "gfn2-xtb")])
def test_r104_xtb_and_mace_run_in_one_process_without_runtime_overrides(tmp_path, order) -> None:
    """Reviewer finding R104: on macOS, loading torch after tblite/PySCF initialised OpenMP
    aborted the process; the fix must not depend on DYLD_LIBRARY_PATH or KMP_* overrides."""
    import os
    import subprocess
    import textwrap

    pytest.importorskip("tblite")
    pytest.importorskip("mace")
    script = tmp_path / "both.py"
    script.write_text(textwrap.dedent(f"""
        from chemflow import build
        from chemflow.domains import load_registry
        from chemflow.runtime import Executor

        registry = load_registry(["molecular", "reactions", "baselines"])
        for method in {list(order)!r}:
            with build.program("one") as builder:
                node = builder.node("potential_energy", molecule=builder.molecule({H2O!r}),
                                    method=method)
            print(method, Executor(registry).run(builder.to_ir())[node.id].value.value)
        print("ok")
    """))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DYLD_", "KMP_"))}
    completed = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                               text=True, timeout=900, check=False)
    assert completed.returncode == 0 and completed.stdout.strip().endswith("ok"), (
        completed.stdout[-500:] + completed.stderr[-2000:])
