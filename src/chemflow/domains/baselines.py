"""Offline classical potentials, native gradients, and projected harmonic modes."""

import hashlib
import os
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy import constants as sc

from chemflow.diagnostics import AdapterError
from chemflow.domains.thermochemistry import ISOTOPE_MASSES
from chemflow.registry import ImplementationSpec, OperatorSpec
from chemflow.runtime.result import ExecutionResult
from chemflow.types import BackendKind

METHODS = ("gfn2-xtb", "mace-mp-0-small")
DEFAULT_MACE_MODEL = "/Volumes/macsoftware/caches/chemflow/mace/mace-mp-0-small.model"
BOHR = sc.physical_constants["Bohr radius"][0] * 1e10
HARTREE_EV = sc.physical_constants["Hartree energy in eV"][0]


def _unavailable(message):
    return AdapterError("CF-E080", None, message,
                        "Install the requested baseline extra and provide local model weights")


def _atoms(molecule):
    from ase import Atoms
    from pyscf.gto.mole import _std_symbol, format_atom

    entries = format_atom(molecule["atoms"], unit="Bohr")
    positions = np.array([position for _, position in entries])
    if molecule.get("unit", "angstrom") == "bohr":
        positions *= BOHR
    return Atoms([_std_symbol(symbol) for symbol, _ in entries], positions=positions)


def _backend(molecule, method):
    if method not in METHODS:
        raise _unavailable(f"Unsupported baseline method: {method}")
    diagnostics = {"method": method}
    try:
        atoms = _atoms(molecule)
        if method == "gfn2-xtb":
            from tblite.interface import Calculator

            def evaluate(positions):
                calculator = Calculator("GFN2-xTB", atoms.numbers, positions / BOHR,
                                        charge=molecule.get("charge", 0),
                                        uhf=molecule.get("spin", 0))
                calculator.set("verbosity", 0)
                result = calculator.singlepoint()
                return float(result.get("energy")), np.asarray(result.get("gradient"))

            package = "tblite"
        else:
            path = Path(os.environ.get("CHEMFLOW_MACE_MODEL", DEFAULT_MACE_MODEL)).expanduser()
            if not path.is_file():
                raise _unavailable(f"MACE weights file is missing: {path}")
            from chemflow.domains import _mace_client

            if find_spec("mace") is None or find_spec("torch") is None:
                raise _unavailable("mace-mp-0-small requires the optional mace-torch and torch packages")
            if molecule.get("charge", 0) or molecule.get("spin", 0):
                raise _unavailable("MACE-MP-0 does not model molecular charge or spin states")
            diagnostics["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

            def evaluate(positions):
                return _mace_client.evaluate(path.resolve(), diagnostics["model_sha256"],
                                             atoms.get_chemical_symbols(), positions,
                                             molecule.get("charge", 0), molecule.get("spin", 0))

            package = "mace-torch"
    except ImportError as exc:
        raise _unavailable(f"{method} requires its optional package: {exc}") from exc
    return atoms, evaluate, f"{package}:{version(package)}", diagnostics


def potential_energy(store, molecule, method):
    atoms, evaluate, device, diagnostics = _backend(molecule, method)
    energy, _ = evaluate(atoms.positions)
    return ExecutionResult(energy, device=device, diagnostics=diagnostics)


def geometry_optimize(molecule, method, kind):
    if kind != "minimum":
        raise _unavailable("Baseline BFGS supports minima only, not transition-state searches")
    atoms, evaluate, _, _ = _backend(molecule, method)
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import BFGS

    class NativePotential(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            energy, gradient = evaluate(self.atoms.positions)
            self.results = {"energy": energy * HARTREE_EV,
                            "forces": -gradient * HARTREE_EV / BOHR}

    optimized = atoms.copy()
    optimized.calc = NativePotential()
    optimizer = BFGS(optimized, logfile=None)
    if not optimizer.run(fmax=1e-5 * HARTREE_EV / BOHR, steps=300):
        raise _unavailable(f"{method} BFGS did not converge within 300 steps")
    text = "; ".join(symbol + " " + " ".join(f"{value:.15g}" for value in position)
                     for symbol, position in zip(optimized.get_chemical_symbols(),
                                                 optimized.positions, strict=True))
    return {**molecule, "atoms": text, "unit": "angstrom", "stationary_kind": kind}


def frequencies(molecule, method):
    from pyscf.data.elements import COMMON_ISOTOPE_MASSES

    atoms, evaluate, _, _ = _backend(molecule, method)
    positions = atoms.positions.copy()
    _, gradient = evaluate(positions)
    size = positions.size
    step = 0.005
    hessian = np.empty((size, size))
    for index in range(size):
        displacement = np.zeros(size)
        displacement[index] = step * BOHR
        _, plus = evaluate(positions + displacement.reshape(-1, 3))
        _, minus = evaluate(positions - displacement.reshape(-1, 3))
        hessian[:, index] = (plus - minus).ravel() / (2 * step)
    hessian = (hessian + hessian.T) / 2
    masses = np.array([ISOTOPE_MASSES.get(number, COMMON_ISOTOPE_MASSES[number])
                       for number in atoms.numbers])
    weighted = hessian / np.sqrt(np.outer(np.repeat(masses, 3), np.repeat(masses, 3)))
    relative = positions - masses @ positions / masses.sum()
    rigid = np.column_stack([
        (np.tile(axis, (len(masses), 1)) * np.sqrt(masses[:, None])).ravel()
        for axis in np.eye(3)] + [
        (np.cross(relative, axis) * np.sqrt(masses[:, None])).ravel()
        for axis in np.eye(3)])
    vectors, singular, _ = np.linalg.svd(rigid, full_matrices=True)
    rank = np.count_nonzero(singular > singular[0] * 1e-8)
    vibrational = vectors[:, rank:]
    eigenvalues = np.linalg.eigvalsh(vibrational.T @ weighted @ vibrational)
    factor = sc.physical_constants["Hartree energy"][0] / (
        (BOHR * 1e-10) ** 2 * sc.atomic_mass)
    modes = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) * factor) / (2 * np.pi * sc.c * 100)
    return {"freq_wavenumber": modes.tolist(), "n_imaginary": int(np.count_nonzero(modes < -50)),
            "max_gradient": float(np.max(np.abs(gradient))), "frequency_level": method}


def context_check(node, inputs, context, nodes):
    if node.op == "potential_energy":
        method = node.params.get("method")
        if method not in METHODS:
            raise ValueError(f"Unsupported baseline method: {method}")
        context.update(inputs.get("molecule", {}))
        context.update(type="estimate", kind="potential_energy", unit="hartree",
                       model_id=method, model_protocol_id=method)
    return []


def register(registry):
    registry.register(OperatorSpec(id="chemflow.potential_energy", version=1,
                      inputs={"molecule": "molecule"}, output="estimate",
                      semantics="Classical potential energy; electronic-structure model change"))
    registry.register(ImplementationSpec(id="potential_energy", version="1",
                      target="potential_energy", adapter=potential_energy,
                      backend_kind=BackendKind.cpu_numeric, params=frozenset({"method"}),
                      cacheable=False))
    registry.context_checks.append(context_check)
