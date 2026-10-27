"""Ideal-gas rigid-rotor harmonic-oscillator thermal corrections."""

import numpy as np
from scipy import constants as sc

from chemflow.diagnostics import AdapterError

THERMAL_KEYS = ("thermal_model", "temperature_K", "pressure_Pa", "frequency_level")
ISOTOPE_MASSES = {1: 1.00782503223, 6: 12.0, 7: 14.00307400443, 8: 15.99491461957}


def thermochemistry(store, molecule, model, frequencies, temperature_K=298.15,
                    pressure_Pa=101325.0, symmetry_number=None):
    from pyscf import gto
    from pyscf.data.elements import COMMON_ISOTOPE_MASSES
    from pyscf.hessian.thermo import rotational_symmetry_number

    for name, value in (("temperature_K", temperature_K), ("pressure_Pa", pressure_Pa)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    modes = np.asarray(frequencies["freq_wavenumber"], dtype=float)
    imaginary = modes[modes < 0]
    allowed = 1 if molecule.get("stationary_kind", "minimum") == "ts" else 0
    if (not np.all(np.isfinite(modes)) or
            not np.isfinite(frequencies["max_gradient"]) or
            frequencies["max_gradient"] > 1e-4 or len(imaginary) > allowed):
        raise AdapterError("CF-E020", None, "Geometry is not stationary for its declared kind",
                           "Optimize the geometry and verify its harmonic frequencies")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                basis=model.basis, verbose=0)
    sigma = rotational_symmetry_number(mol) if symmetry_number is None else symmetry_number
    if isinstance(sigma, bool) or not np.isfinite(sigma) or sigma < 1 or int(sigma) != sigma:
        raise ValueError("symmetry_number must be a positive integer")
    mass_table = list(COMMON_ISOTOPE_MASSES)
    for charge, mass in ISOTOPE_MASSES.items():
        mass_table[charge] = mass
    masses = mol.atom_mass_list(mass_table=mass_table) * sc.atomic_mass
    positions = mol.atom_coords(unit="Angstrom") * 1e-10
    relative = positions - masses @ positions / masses.sum()
    inertia = sum(mass * (np.dot(position, position) * np.eye(3) -
                          np.outer(position, position))
                  for mass, position in zip(masses, relative, strict=True))
    moments = np.sort(np.linalg.eigvalsh(inertia))
    linear = bool(moments[0] < 1e-3 * moments[-1])
    kt = sc.k * temperature_K
    q_trans = (2 * np.pi * masses.sum() * kt / sc.h ** 2) ** 1.5 * kt / pressure_Pa
    entropy = sc.k * (np.log(q_trans) + 2.5)
    internal = 1.5 * kt
    if mol.natm > 1:
        thetas = sc.h ** 2 / (8 * np.pi ** 2 *
                              moments[moments > 1e-3 * moments[-1]] * sc.k)
        if linear:
            q_rot = temperature_K / (sigma * thetas[0])
            entropy += sc.k * (np.log(q_rot) + 1)
            internal += kt
        else:
            q_rot = np.sqrt(np.pi) / sigma * np.sqrt(temperature_K ** 3 / np.prod(thetas))
            entropy += sc.k * (np.log(q_rot) + 1.5)
            internal += 1.5 * kt
    nu = modes[modes > 0] * 100 * sc.c
    zpe = 0.5 * sc.h * nu.sum()
    scaled = sc.h * nu / kt
    with np.errstate(over="ignore"):
        occupation = 1 / np.expm1(scaled)
    internal += zpe + np.sum(sc.h * nu * occupation)
    multiplicity = mol.spin + 1
    entropy += sc.k * (np.sum(scaled * occupation - np.log(-np.expm1(-scaled))) +
                       np.log(multiplicity))
    enthalpy = internal + kt
    hartree = sc.physical_constants["Hartree energy"][0]
    return {"zpe": float(zpe / hartree), "u_corr": float(internal / hartree),
            "h_corr": float(enthalpy / hartree), "s": float(entropy / hartree),
            "g_corr": float((enthalpy - temperature_K * entropy) / hartree),
            "temperature_K": temperature_K, "pressure_Pa": pressure_Pa,
            "symmetry_number": int(sigma), "spin_multiplicity": multiplicity,
            "linear": linear, "thermal_model": "rrho_ideal_gas",
            "frequency_level": frequencies.get("frequency_level",
                                                "uhf" if molecule.get("spin", 0) else "rhf"),
            "excluded_imaginary_modes": imaginary.tolist(),
            "n_imaginary": len(imaginary)}
