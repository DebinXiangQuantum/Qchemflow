"""Stoichiometric electronic energies and stationary-point verification."""

from collections import Counter

import numpy as np


def balanced(terms):
    from pyscf.gto.mole import _std_symbol, format_atom

    counts = Counter()
    charge = 0.0
    for coefficient, molecule in terms:
        for symbol, _ in format_atom(molecule["atoms"]):
            counts[_std_symbol(symbol)] += coefficient
        charge += coefficient * molecule.get("charge", 0)
    return abs(charge) < 1e-10 and all(abs(value) < 1e-10 for value in counts.values())


def _mean_field(molecule, model, level):
    from pyscf import gto, scf

    if model.hamiltonian_class != "nonrel_bo" or model.ecp or model.dft_functional:
        raise ValueError("Geometry and Hessian adapters require the nonrelativistic HF model")
    method = level.get("method", "uhf" if molecule.get("spin", 0) else "rhf").lower()
    if method not in ("rhf", "uhf"):
        raise ValueError("Geometry and Hessian levels must be rhf or uhf")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                basis=model.basis, verbose=0)
    mf = (scf.UHF if method == "uhf" else scf.RHF)(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise ValueError("Geometry reference SCF failed to converge")
    return mf


def geometry_optimize(store, molecule, model, kind="minimum", level=None):
    from chemflow.domains import baselines

    if kind not in ("minimum", "ts"):
        raise ValueError("Geometry kind must be minimum or ts")
    method = (level or {}).get("method", "").lower()
    if method in baselines.METHODS:
        return baselines.geometry_optimize(molecule, method, kind)
    from pyscf.geomopt.geometric_solver import optimize

    mf = _mean_field(molecule, model, level or {})
    optimized = optimize(mf, transition=kind == "ts", maxsteps=100,
                         convergence_grms=1e-6, convergence_gmax=3e-6,
                         convergence_drms=1e-5, convergence_dmax=3e-5)
    atoms = "; ".join(f"{optimized.atom_symbol(index)} " + " ".join(
        f"{coordinate:.15g}" for coordinate in position)
        for index, position in enumerate(optimized.atom_coords(unit="Angstrom")))
    return {**molecule, "atoms": atoms, "unit": "angstrom", "stationary_kind": kind}


def frequencies(store, molecule, model, level=None):
    from pyscf.hessian.thermo import harmonic_analysis

    from chemflow.domains import baselines

    method = (level or {}).get("method", "uhf" if molecule.get("spin", 0) else "rhf").lower()
    if method in baselines.METHODS:
        return baselines.frequencies(molecule, method)
    mf = _mean_field(molecule, model, level or {})
    modes = harmonic_analysis(mf.mol, mf.Hessian().kernel())["freq_wavenumber"]
    return {"freq_wavenumber": [float(value.real) if abs(value.imag) < 1e-8 else
                                -float(abs(value.imag)) for value in modes],
            "n_imaginary": int(np.count_nonzero(np.abs(np.imag(modes)) > 50)),
            "max_gradient": float(np.max(np.abs(mf.nuc_grad_method().kernel()))),
            "frequency_level": method}
