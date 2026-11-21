"""Numerical kernels behind QCL's implementations, checked against independent references.

These were exercised through the legacy executor; they are now called directly, as the QCL
worker calls them. Oracles are dense diagonalizations or PySCF's own solvers.
"""

import numpy as np
import pytest
from pyscf import gto, mcscf, scf
from test_review_m0_2 import _s2_operator

from chemflow.impls.adapt import adapt_vqe
from chemflow.impls.fci import matrix, sector_states
from chemflow.impls.prepare import prepare_sector_state
from chemflow.impls.pyscf_impls import build_molecule, classical_energy, fci_pyscf
from chemflow.impls.sci import sci_pyscf
from chemflow.impls.sector import SectorEngine
from chemflow.impls.sqd import sqd_sector
from chemflow.impls.statevector import (
    encode,
    expectation,
    prepare_hf,
    sample,
    sample_configurations,
)
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"
N2 = "N 0 0 0; N 0 0 1.10"


def _data(atoms, basis="sto-3g", active=None, spin=0):
    store = ArtifactStore()
    space = ActiveSpaceSpec(n_elec=active[0], n_orb=active[1]) if active else None
    return build_molecule(atoms, ElectronicModel(basis=basis), store, spin=spin,
                          active=space), store


def _dense_lowest(data, store, total_spin):
    states = sector_states(data.sector)
    energies, vectors = np.linalg.eigh(matrix(data.hamiltonian, store, states).toarray())
    spin = _s2_operator(states, data.sector.n_orb)
    target = total_spin * (total_spin + 1)
    return next(energy for energy, vector in zip(energies, vectors.T, strict=True)
                if abs(vector @ spin @ vector - target) < 1e-8)


@pytest.mark.parametrize("geometry,active", [(H4, (4, 4)), ("Li 0 0 0; H 0 0 1.6", (4, 5)),
                                             ("O 0 0 0; O 0 0 1.2", (2, 2))])
@pytest.mark.parametrize("total_spin", [0, 1])
def test_spin_targeted_fci_finds_the_lowest_state_of_that_spin(geometry, active, total_spin):
    data, store = _data(geometry, active=active)
    energy, spin_squared, roots = fci_pyscf(data, store, total_spin, return_roots=True)
    assert energy == pytest.approx(_dense_lowest(data, store, total_spin), abs=1e-8)
    assert spin_squared == pytest.approx(total_spin * (total_spin + 1), abs=1e-8)
    assert roots >= 1


def test_spin_targeted_sqd_and_sci_are_exact_singlets_and_upper_bounds():
    # The symmetric product of sampled strings is not S2-invariant; spin-targeted subspace
    # methods must spin-complete their determinant space.
    data, store = _data(N2, basis="6-31g", active=(8, 10))
    mf = scf.RHF(gto.M(atom=N2, basis="6-31g", verbose=0)).run()
    casci = mcscf.CASCI(mf, 10, 8)
    casci.fix_spin_(ss=0)
    singlet = casci.kernel()[0]
    prep = prepare_sector_state(data, data.sector.model_dump(), store, "lucj", "ccsd")
    samples = sample_configurations(prep, 256, 11, store=store)
    sqd, sqd_info = sqd_sector(data, data.sector, store, samples, n_batches=2, max_iter=2,
                               total_spin=0)
    sci, sci_info = sci_pyscf(data, store, select_cutoff=1e-2, total_spin=0)
    for energy, info in ((sqd, sqd_info), (sci, sci_info)):
        assert abs(info["s2"]) < 1e-8
        assert energy >= singlet - 1e-9


def test_adapt_vqe_reaches_chemical_accuracy_on_square_h4():
    data, store = _data(H4, active=(4, 4))
    result = adapt_vqe(SectorEngine(data.hamiltonian, data.sector, store), data.sector)
    exact, _ = fci_pyscf(data, store)
    assert exact - 1e-9 <= result.energy < exact + 1.6e-3
    assert 0 < result.n_operators <= 40 and result.n_gradient_evaluations > 0
    assert len(result.selected_operators) == result.n_operators


@pytest.mark.parametrize("method", ["hf", "mp2", "ccsd", "ccsd_t"])
def test_open_shell_controls_run_on_rohf_and_report_spin(method):
    data, store = _data("Li 0 0 0", spin=1)
    diagnostics = {}
    energy = classical_energy(data, method, store, diagnostics)
    assert np.isfinite(energy) and diagnostics["reference"] == "ROHF"
    assert np.isfinite(diagnostics["s2"])
    if method == "hf":
        assert energy == pytest.approx(data.mf.e_tot, abs=1e-8)


def test_finite_shot_estimates_converge_to_the_exact_expectation():
    data, store = _data("H 0 0 0; H 0 0 0.74")
    encoding, operator = encode(data.hamiltonian, store)
    prep = prepare_hf(data.hamiltonian, encoding, store, 1, 1)
    exact = expectation(operator, prep, store)
    assert exact == pytest.approx(data.mf.e_tot, abs=1e-8)
    _, mean, error = sample(operator, prep, store, 1000, seed=42)
    assert error > 0 and abs(mean - exact) < 5 * error
    _, _, larger = sample(operator, prep, store, 100_000, seed=42)
    assert larger < error


@pytest.mark.parametrize("order", [("gfn2-xtb", "mace-mp-0-small"),
                                   ("mace-mp-0-small", "gfn2-xtb")])
def test_xtb_and_mace_run_in_one_process_without_runtime_overrides(tmp_path, order):
    """R104: loading torch after tblite/PySCF initialised OpenMP aborted the process; the fix
    must not depend on DYLD_LIBRARY_PATH or KMP_* overrides."""
    import os
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("tblite")
    pytest.importorskip("mace")
    script = tmp_path / "both.py"
    script.write_text(textwrap.dedent(f"""
        from chemflow.domains.baselines import potential_energy

        molecule = {{"atoms": "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
                     "charge": 0, "spin": 0, "unit": "angstrom"}}
        for method in {list(order)!r}:
            print(method, potential_energy(None, molecule, method).value)
        print("ok")
    """))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DYLD_", "KMP_"))}
    completed = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                               text=True, timeout=900, check=False)
    assert completed.returncode == 0 and completed.stdout.strip().endswith("ok"), (
        completed.stdout[-500:] + completed.stderr[-2000:])
