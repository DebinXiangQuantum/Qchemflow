"""Internal-coordinate geometry, constrained surfaces and reaction paths."""

import math
import tempfile
from typing import TypedDict

import numpy as np

from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule


class OptimizationResult(TypedDict):
    atoms: str
    unit: str
    molecule: dict
    energy_hartree: float
    relaxation: dict


class NEBResult(TypedDict):
    images: list[dict]
    energies_hartree: list[float]
    climbing_image: int
    converged: bool


class IRCResult(TypedDict):
    ts: dict
    endpoints: list[dict]
    connects: bool | None


class RateResult(TypedDict):
    k_per_s: float
    kappa: float
    temperature_K: float
    model: str


def coordinates(molecule):
    from pyscf.gto import format_atom
    from scipy.constants import physical_constants

    entries = format_atom(molecule["atoms"], unit=1)
    positions = np.array([position for _, position in entries], dtype=float)
    unit = molecule.get("unit", "angstrom").lower()
    if unit == "bohr":
        positions *= physical_constants["Bohr radius"][0] * 1e10
    elif unit not in ("angstrom", "ang", "a"):
        raise ValueError("Geometry requires angstrom or bohr units")
    if not len(entries) or not np.isfinite(positions).all():
        raise ValueError("Geometry must contain finite positions")
    return [symbol for symbol, _ in entries], positions


def molecular_result(molecule, symbols, positions):
    return {**molecule, "atoms": "; ".join(
        symbol + " " + " ".join(f"{value:.15g}" for value in position)
        for symbol, position in zip(symbols, positions, strict=True)), "unit": "angstrom"}


def bond_graph(molecule):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    symbols, positions = coordinates(molecule)
    xyz = f"{len(symbols)}\n\n" + "\n".join(
        symbol + " " + " ".join(f"{value:.12f}" for value in position)
        for symbol, position in zip(symbols, positions, strict=True))
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol, charge=molecule.get("charge", 0))
    return {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in mol.GetBonds()}


def coordinate_check(kind, atoms, value):
    sizes = {"bond": 2, "angle": 3, "dihedral": 4}
    if (kind not in sizes or not isinstance(atoms, list) or len(atoms) != sizes[kind]
            or any(type(index) is not int or index < 0 for index in atoms)
            or len(set(atoms)) != len(atoms)):
        raise ValueError("Invalid internal-coordinate atom indices")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Coordinate value must be finite")
    if kind == "bond" and value <= 0 or kind == "angle" and not 0 < value < 180:
        raise ValueError("Bond must be positive; angle must lie strictly between 0 and 180")


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Undefined coordinate axis (coincident or collinear atoms)")
    return vector / norm


def geometry_transform(store, molecule, operation, atoms, value, fragment=None) -> dict:
    from scipy.spatial.transform import Rotation

    symbols, positions = coordinates(molecule)
    count = len(symbols)
    if operation == "translate_fragment":
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Separation must be finite and positive")
        cut = None
    else:
        kind = operation.removeprefix("set_")
        if operation != f"set_{kind}":
            raise ValueError("Unknown geometry operation")
        coordinate_check(kind, atoms, value)
        if max(atoms) >= count:
            raise ValueError("Atom index outside molecule")
        cut = tuple(sorted(atoms[:2] if kind == "bond" else atoms[1:3]))
    if fragment is None:
        if cut is None:
            raise ValueError("Translation requires an explicit fragment")
        bonds = bond_graph(molecule)
        if cut not in bonds:
            raise ValueError("Defining bond is absent from the perceived graph")
        bonds.remove(cut)
        moved, pending = set(), [atoms[-1]]
        while pending:
            current = pending.pop()
            if current in moved:
                continue
            moved.add(current)
            pending.extend(right if left == current else left for left, right in bonds
                           if current in (left, right))
        if cut[0] in moved and cut[1] in moved:
            raise ValueError("Ring bond requires an explicit movable fragment")
        fragment = sorted(moved)
    if (not isinstance(fragment, list) or not fragment or len(set(fragment)) != len(fragment)
            or any(type(index) is not int or not 0 <= index < count for index in fragment)
            or len(fragment) == count):
        raise ValueError("Fragment must be a nonempty proper subset of atom indices")
    if cut is None:
        fixed = sorted(set(range(count)) - set(fragment))
        delta = positions[fragment].mean(0) - positions[fixed].mean(0)
        positions[fragment] += (value - np.linalg.norm(delta)) * _unit(delta)
    elif kind == "bond":
        if atoms[-1] not in fragment or atoms[0] in fragment:
            raise ValueError("Fragment must contain the moving atom, not the fixed atom")
        delta = positions[atoms[1]] - positions[atoms[0]]
        positions[fragment] += (value - np.linalg.norm(delta)) * _unit(delta)
    else:
        if atoms[-1] not in fragment or atoms[0] in fragment:
            raise ValueError("Fragment must contain the moving atom, not the fixed atom")
        origin = positions[atoms[1]].copy()
        if kind == "angle":
            fixed = _unit(positions[atoms[0]] - origin)
            moving = _unit(positions[atoms[2]] - origin)
            axis = _unit(np.cross(fixed, moving))
            current = math.degrees(math.acos(np.clip(fixed @ moving, -1, 1)))
        else:
            axis = _unit(positions[atoms[2]] - origin)
            first = positions[atoms[0]] - origin
            last = positions[atoms[3]] - positions[atoms[2]]
            first = _unit(first - (first @ axis) * axis)
            last = _unit(last - (last @ axis) * axis)
            current = math.degrees(math.atan2(np.cross(axis, first) @ last, first @ last))
        rotation = Rotation.from_rotvec(axis * math.radians(value - current))
        positions[fragment] = origin + rotation.apply(positions[fragment] - origin)
    return molecular_result(molecule, symbols, positions)


def validate_level(level):
    if (not isinstance(level, dict) or set(level) - {"method", "functional", "dispersion"}
            or level.get("method") not in ("hf", "dft", "gfn2")
            or any(value is not None and not isinstance(value, str) for value in level.values())):
        raise ValueError("Level requires method hf, dft or gfn2 and optional functional/dispersion")
    if level.get("dispersion") is not None:
        raise ValueError("Additional dispersion is not supported by this path adapter")
    if level["method"] == "dft" and not level.get("functional"):
        raise ValueError("DFT requires an explicit functional")
    if level["method"] != "dft" and level.get("functional") is not None:
        raise ValueError("Functional applies only to DFT")


def mean_field(molecule, model, level):
    from pyscf import dft, gto, scf

    validate_level(level)
    if model.hamiltonian_class != "nonrel_bo" or model.ecp or model.dft_functional:
        raise ValueError("Path model requires all-electron nonrel_bo with functional in level")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                basis=model.basis, verbose=0)
    if level["method"] == "dft":
        calculation = (dft.UKS if mol.spin else dft.RKS)(mol)
        calculation.xc = level["functional"]
    elif level["method"] == "hf":
        calculation = (scf.UHF if mol.spin else scf.RHF)(mol)
    else:
        raise ValueError("GFN2 uses the tblite ASE calculator")
    calculation.conv_tol = 1e-12
    calculation.kernel()
    if not calculation.converged:
        raise ValueError("Path SCF did not converge")
    return calculation


def ase_atoms(molecule, model, level):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree

    validate_level(level)
    symbols, positions = coordinates(molecule)
    atoms = Atoms(symbols, positions=positions)
    if level["method"] == "gfn2":
        from chemflow.domains._path_calculator import isolated_tblite

        atoms.calc = isolated_tblite(molecule)
    else:
        class PySCFCalculator(Calculator):
            implemented_properties = ("energy", "forces")

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                geometry = molecular_result(molecule, symbols, self.atoms.positions)
                calculation = mean_field(geometry, model, level)
                self.results = {"energy": calculation.e_tot * Hartree,
                                "forces": -calculation.nuc_grad_method().kernel() * Hartree / Bohr}

        atoms.calc = PySCFCalculator()
    return atoms


def verify_constraints(molecule, constraints, gradient):
    from ase import Atoms
    from ase.units import Bohr

    symbols, positions = coordinates(molecule)
    atoms = Atoms(symbols, positions=positions)

    def values():
        measured = []
        for constraint in constraints:
            method = {"bond": atoms.get_distance, "angle": atoms.get_angle,
                      "dihedral": atoms.get_dihedral}[constraint["type"]]
            measured.append(method(*constraint["atoms"]))
        return np.array(measured)

    def difference(left, right):
        residual = left - right
        for index, constraint in enumerate(constraints):
            if constraint["type"] == "dihedral":
                residual[index] = (residual[index] + 180) % 360 - 180
        return residual

    if np.max(abs(difference(values(), np.array([item["value"] for item in constraints])))) > 1e-3:
        raise ValueError("Optimization did not satisfy the requested constraints")
    jacobian = np.empty((len(constraints), positions.size))
    step = 1e-5
    for coordinate in range(positions.size):
        atoms.positions[:] = positions
        atoms.positions.flat[coordinate] += step
        plus = values()
        atoms.positions.flat[coordinate] -= 2 * step
        jacobian[:, coordinate] = difference(plus, values()) / (2 * step)
    force = np.asarray(gradient).ravel() / Bohr
    projected = force - jacobian.T @ np.linalg.lstsq(jacobian.T, force, rcond=None)[0]
    if np.max(abs(projected)) > 1e-4:
        raise ValueError("Constrained optimization did not reach a stationary geometry")


def constrained_optimize(store, molecule, model, level, constraints) -> OptimizationResult:
    from ase.units import Hartree

    validate_level(level)
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("At least one constraint is required")
    geometry = molecule
    for constraint in constraints:
        if not isinstance(constraint, dict) or set(constraint) != {"type", "atoms", "value"}:
            raise ValueError("Constraint requires type, atoms and value")
        coordinate_check(constraint["type"], constraint["atoms"], constraint["value"])
        if max(constraint["atoms"]) >= len(coordinates(molecule)[0]):
            raise ValueError("Constraint atom index outside molecule")
    if level["method"] == "gfn2":
        from ase.constraints import FixInternals
        from ase.optimize import BFGS

        atoms = ase_atoms(geometry, model, level)
        keys = {"bond": "bonds", "angle": "angles_deg", "dihedral": "dihedrals_deg"}
        options = {key: [] for key in keys.values()}
        for constraint in constraints:
            options[keys[constraint["type"]]].append([constraint["value"], constraint["atoms"]])
        atoms.set_constraint(FixInternals(**options))
        atoms.set_positions(atoms.positions)
        if not BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500):
            raise ValueError("Constrained GFN2 optimization did not converge")
        geometry = molecular_result(molecule, atoms.get_chemical_symbols(), atoms.positions)
        energy = atoms.get_potential_energy() / Hartree
    else:
        from pyscf.geomopt.geometric_solver import optimize

        names = {"bond": "distance", "angle": "angle", "dihedral": "dihedral"}
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".constraints") as stream:
            stream.write("$set\n")
            for constraint in constraints:
                indices = " ".join(str(index + 1) for index in constraint["atoms"])
                stream.write(f"{names[constraint['type']]} {indices} {constraint['value']}\n")
            stream.write("$end\n")
            stream.flush()
            optimized = optimize(mean_field(geometry, model, level), constraints=stream.name,
                                 maxsteps=200, convergence_grms=1e-6, convergence_gmax=3e-6,
                                 convergence_drms=1e-5, convergence_dmax=3e-5)
        geometry = molecular_result(molecule,
            [optimized.atom_symbol(index) for index in range(optimized.natm)],
            optimized.atom_coords(unit="Angstrom"))
        calculation = mean_field(geometry, model, level)
        energy = calculation.e_tot
        verify_constraints(geometry, constraints, calculation.nuc_grad_method().kernel())
    return {**geometry, "molecule": geometry, "energy_hartree": float(energy),
            "relaxation": {"initial_molecule": molecule, "level": level,
                           "constraints": constraints,
                           "electronic_model": model.model_dump(mode="json")}}


def neb_path(store, reactant, product, model, level, images=7, interpolation="idpp",
             optimizer="fire", fmax=0.02, maxsteps=1000) -> NEBResult:
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE
    from ase.units import Hartree

    if type(images) is not int or not 1 <= images <= 100:
        raise ValueError("images must count 1–100 interior images")
    if interpolation not in ("linear", "idpp") or optimizer not in ("fire", "bfgs"):
        raise ValueError("Use linear/IDPP interpolation and FIRE/BFGS optimization")
    first, last = coordinates(reactant), coordinates(product)
    if first[0] != last[0] or any(reactant.get(key, 0) != product.get(key, 0)
                                for key in ("charge", "spin")):
        raise ValueError("NEB endpoints require identical ordered atoms, charge and spin")
    chain = [ase_atoms(reactant, model, level) for _ in range(images + 1)]
    chain.append(ase_atoms(product, model, level))
    band = NEB(chain, climb=True, method="improvedtangent")
    for index, image in enumerate(chain[1:-1], 1):
        fraction = index / (images + 1)
        image.positions[:] = (1 - fraction) * first[1] + fraction * last[1]
        centered = image.positions - image.positions.mean(0)
        if len(image) > 2 and np.linalg.matrix_rank(centered, tol=1e-8) < 2:
            direction = np.linalg.svd(centered)[2][0]
            normal = _unit(np.cross(direction, np.eye(3)[np.argmin(abs(direction))]))
            image.positions[-1] += math.sin(math.pi * fraction) * normal
    if interpolation == "idpp":
        from ase.mep.neb import idpp_interpolate

        idpp_interpolate(band, fmax=0.05, steps=200, traj=None, log=None)
    solver = FIRE if optimizer == "fire" else BFGS
    converged = solver(band, logfile=None).run(fmax=fmax, steps=maxsteps)
    if not converged:
        raise ValueError("Climbing-image NEB did not converge")
    energies = [float(image.get_potential_energy() / Hartree) for image in chain]
    climbing = int(np.argmax(energies))
    if climbing in (0, len(chain) - 1):
        raise ValueError("NEB has no interior barrier")
    return {"images": [molecular_result(reactant, first[0], image.positions) for image in chain],
            "energies_hartree": energies, "climbing_image": climbing, "converged": converged}


def irc(store, ts, model, level, reactant=None, product=None, fmax=0.02, maxsteps=500) -> IRCResult:
    import json
    import os
    import subprocess
    import sys

    validate_level(level)
    if (reactant is None) != (product is None):
        raise ValueError("Connectivity comparison requires both declared endpoints")
    request = {"ts": ts, "model": model.model_dump(mode="json"), "level": level,
               "reactant": reactant, "product": product, "fmax": fmax, "maxsteps": maxsteps}
    with tempfile.TemporaryDirectory(prefix="chemflow-irc-") as directory:
        result = subprocess.run([sys.executable, "-m", "chemflow.domains._path_irc"],
            input=json.dumps(request), capture_output=True, text=True, check=False,
            env={**os.environ, "JAX_COMPILATION_CACHE_DIR": directory})
    if result.returncode:
        raise ValueError(f"Sella IRC failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def _irc(ts, model, level, reactant=None, product=None, fmax=0.02, maxsteps=500):
    from sella import IRC, Sella

    atoms = ase_atoms(ts, model, level)
    if not Sella(atoms, logfile=None).run(fmax=0.005, steps=200):
        raise ValueError("Sella TS refinement did not converge")
    refined = molecular_result(ts, atoms.get_chemical_symbols(), atoms.positions)
    path = IRC(atoms, logfile=None, dx=0.05, ninner_iter=50)
    endpoints = []
    for direction in ("forward", "reverse"):
        if not path.run(fmax=fmax, steps=maxsteps, direction=direction):
            raise ValueError(f"IRC {direction} did not converge")
        endpoints.append(molecular_result(ts, atoms.get_chemical_symbols(), atoms.positions))
    connects = None
    if (reactant is None) != (product is None):
        raise ValueError("Connectivity comparison requires both declared endpoints")
    if reactant is not None:
        expected = [bond_graph(item) for item in (reactant, product)]
        observed = [bond_graph(item) for item in endpoints]
        symbols = [coordinates(item)[0] for item in (ts, reactant, product)]
        connects = symbols[0] == symbols[1] == symbols[2] and (
            observed == expected or observed == expected[::-1])
    return {"ts": refined, "endpoints": endpoints, "connects": connects}


def connected_ts(store, molecule, model, reactant, product, level, kind="ts") -> dict:
    from chemflow.domains.reactions import frequencies

    if kind != "ts" or level != {"method": "uhf" if molecule.get("spin", 0) else "rhf"}:
        raise ValueError("Barrier stationary geometry and IRC must use the same HF level")
    result = irc(store, molecule, model, {"method": "hf"}, reactant, product)
    if result["connects"] is not True:
        raise ValueError("IRC endpoints do not connect the declared reactant and product")
    geometry = {**result["ts"], "stationary_kind": "ts"}
    modes = frequencies(store, geometry, model, level)
    if modes["n_imaginary"] != 1 or modes["max_gradient"] >= 1e-4:
        raise ValueError("Barrier requires a stationary TS with exactly one imaginary mode")
    return geometry


def rate_constant(store, barrier, temperature_K, tunnelling=None, frequency_cm1=None) -> RateResult:
    if type(barrier) not in (int, float) or not math.isfinite(barrier):
        raise ValueError("Barrier must be a finite Hartree value")
    if (type(temperature_K) not in (int, float) or
            not math.isfinite(temperature_K) or temperature_K <= 0):
        raise ValueError("Temperature must be positive and finite")
    if tunnelling not in (None, "wigner"):
        raise ValueError("Tunnelling must be None or wigner")
    boltzmann, planck, hartree = 1.380649e-23, 6.62607015e-34, 4.3597447222071e-18
    kappa = 1.0
    if tunnelling == "wigner":
        if (type(frequency_cm1) not in (int, float) or
                not math.isfinite(frequency_cm1) or frequency_cm1 == 0):
            raise ValueError("Wigner requires a nonzero imaginary-mode wavenumber in cm^-1")
        frequency = abs(frequency_cm1) * 29979245800.0
        kappa += (planck * frequency / (boltzmann * temperature_K)) ** 2 / 24
    rate = kappa * boltzmann * temperature_K / planck * math.exp(
        -barrier * hartree / (boltzmann * temperature_K))
    return {"k_per_s": rate, "kappa": kappa, "temperature_K": temperature_K,
            "model": "eyring"}


def resources(**kwargs):
    return {"wall_s": None, "cpu_core_s": None, "gpu_s": 0, "sim_shots": 0,
            "qpu_s": 0, "qpu_shots": 0}


def register(registry):
    from . import drug, reactions, scans

    if "component_energy" not in registry.implementations:
        drug.register(registry)
    if "reaction_energy" not in registry.implementations:
        reactions.register(registry)
    scans.register(registry)
    specs = (
        ("geometry_transform", {"molecule": "molecule"}, "molecule", geometry_transform,
         {"operation", "atoms", "value", "fragment"},
         "Rigid RDKit bond-graph internal-coordinate and centroid-separation transformations"),
        ("constrained_optimize", {"molecule": "molecule", "model": "model"}, "molecule",
         constrained_optimize, {"level", "constraints"},
         "geomeTRIC HF/DFT or ASE tblite GFN2 constrained optimization and final energy"),
        ("neb_path", {"reactant": "molecule", "product": "molecule", "model": "model"},
         "reaction_path", neb_path,
         {"level", "images", "interpolation", "optimizer", "fmax", "maxsteps"},
         "ASE climbing-image NEB with interior image count and linear or IDPP interpolation"),
        ("irc", {"ts": "molecule", "model": "model"}, "reaction_path", irc,
         {"level", "fmax", "maxsteps"},
         "Sella bidirectional intrinsic reaction coordinate and RDKit endpoint connectivity"),
        ("connected_ts", {"molecule": "molecule", "model": "model", "reactant": "molecule",
                          "product": "molecule"}, "molecule", connected_ts, {"level", "kind"},
         "HF transition state verified by one imaginary frequency and a connecting Sella IRC"),
        ("rate_constant", {}, "rate_constant", rate_constant,
         {"barrier", "temperature_K", "tunnelling", "frequency_cm1"},
         "Model-defined Eyring rate in s^-1 with optional supplied-frequency Wigner correction"),
    )
    for name, inputs, output, adapter, parameters, description in specs:
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
            output=output, parameters=dict.fromkeys(parameters, "JSON"), semantics=description))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
            backend_kind="cpu_numeric", execution_role="classical", params=frozenset(parameters),
            resource_model=resources))
        registry.register(RefinementRule(target=name, template_id=name, relation="approximate"))
    registry.register(RefinementRule(target="reaction_energy", template_id="reaction_barrier",
                                     relation="approximate"))


def base_plan(record, active=None):
    from .scans import base_plan as scan_plan

    if record.spec.target.kind == "reaction_barrier":
        from chemflow.domains.reactions import base_plan as reaction_plan
        from chemflow.ir.schema import Edge, Node

        plan, controls = reaction_plan(record, active)
        target = record.spec.target
        reactant = next(species.molecule for species in target.species if species.role == "minimum")
        plan.nodes.extend([
            Node(id="barrier_reactant", op="molecule", params=reactant),
            Node(id="barrier_product", op="molecule", params=target.product)])
        for node in plan.nodes:
            if node.op == "geometry_optimize" and node.params.get("kind") == "ts":
                node.op = "connected_ts"
                node.inputs.update(reactant="barrier_reactant", product="barrier_product")
                plan.edges.extend([Edge(source=source, target=node.id, port=port)
                                   for port, source in node.inputs.items()
                                   if port in ("reactant", "product")])
        return plan, controls
    return scan_plan(record, active)


def identity(plan, record, contexts):
    from .scans import plan_identity

    expected, _ = base_plan(record)
    return plan_identity(plan, expected)


def validate_parameters(operation, parameters):
    if operation == "connected_ts":
        if parameters.get("level") not in ({"method": "rhf"}, {"method": "uhf"}):
            raise ValueError("Connected TS verification requires the declared HF level")
        if parameters.get("kind", "ts") != "ts":
            raise ValueError("Connected TS requires kind=ts")
    elif "level" in parameters:
        validate_level(parameters["level"])
    for key in ("fmax",):
        if key in parameters and (type(parameters[key]) not in (int, float) or
                not math.isfinite(parameters[key]) or parameters[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if "maxsteps" in parameters and (type(parameters["maxsteps"]) is not int or
                                     parameters["maxsteps"] < 1):
        raise ValueError("maxsteps must be a positive integer")
    if operation == "rate_constant" and {"barrier", "temperature_K"} <= parameters.keys():
        rate_constant(None, **parameters)
    if operation == "neb_path":
        if type(parameters.get("images", 7)) is not int or not 1 <= parameters.get("images", 7) <= 100:
            raise ValueError("images must count 1–100 interior images")
        if parameters.get("interpolation", "idpp") not in ("linear", "idpp"):
            raise ValueError("Invalid interpolation")
        if parameters.get("optimizer", "fire") not in ("fire", "bfgs"):
            raise ValueError("Invalid optimizer")
    if operation == "constrained_optimize" and "constraints" in parameters:
        constraints = parameters["constraints"]
        if not isinstance(constraints, list) or not constraints:
            raise ValueError("constraints must be a nonempty list")
        for constraint in constraints:
            if not isinstance(constraint, dict) or set(constraint) != {"type", "atoms", "value"}:
                raise ValueError("Constraint requires type, atoms and value")
            coordinate_check(constraint["type"], constraint["atoms"], constraint["value"])
    if operation == "geometry_transform" and {"operation", "atoms", "value"} <= parameters.keys():
        action = parameters["operation"]
        if action not in ("set_bond", "set_angle", "set_dihedral", "translate_fragment"):
            raise ValueError("Unknown geometry operation")
        if action != "translate_fragment":
            coordinate_check(action.removeprefix("set_"), parameters["atoms"], parameters["value"])
        elif (type(parameters["value"]) not in (int, float) or
              not math.isfinite(parameters["value"]) or parameters["value"] <= 0):
            raise ValueError("Separation must be finite and positive")
        fragment = parameters.get("fragment")
        if (action == "translate_fragment" and fragment is None or fragment is not None and (
                not isinstance(fragment, list) or not fragment or
                any(type(index) is not int or index < 0 for index in fragment) or
                len(set(fragment)) != len(fragment))):
            raise ValueError("Invalid movable fragment")
