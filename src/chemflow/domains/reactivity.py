"""Rigid bond chemistry and finite molecular energy collections (APP C)."""

import math
from collections import Counter
from typing import Literal, TypedDict

import numpy as np
from scipy.special import expit

from chemflow import build
from chemflow.domains._adapters import estimate
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule

FCI_MAX_DETERMINANTS = 250_000
METHODS = ("hf", "mp2", "ccsd_t", "fci")


class BondRecord(TypedDict):
    atoms: list[int]
    elements: list[str]
    symmetry_class: str


class BondEnumeration(TypedDict):
    bonds: list[BondRecord]


class RadicalFragments(TypedDict):
    fragments: list[dict]


class SiteRecord(TypedDict):
    atom: int
    bde_hartree: float
    bde_kcal: float
    sasa: float
    sasa_fraction: float
    score: float


class SiteReactivity(TypedDict):
    sites: list[SiteRecord]
    top_site: int | None
    evidence: Literal["empirical"]


class BoltzmannWeights(TypedDict):
    weights: list[float]
    temperature_K: float


class ConformerRecord(TypedDict):
    energy_hartree: float
    atoms: str


class ConformerSearch(TypedDict):
    conformers: list[ConformerRecord]
    method: str
    seed: int
    energy_model: str


def _entries(molecule):
    from pyscf.gto import format_atom

    return format_atom(molecule["atoms"], unit=molecule.get("unit", "angstrom"))


def perceived(molecule):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    entries = _entries(molecule)
    block = f"{len(entries)}\nChemFlow\n" + "\n".join(
        symbol + " " + " ".join(str(float(value) * 0.529177210903) for value in position)
        for symbol, position in entries)
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise ValueError("Invalid molecular coordinates")
    rdDetermineBonds.DetermineBonds(mol, charge=molecule.get("charge", 0))
    return mol


def enumerate_bonds(store, molecule, elements=None) -> BondEnumeration:
    from rdkit import Chem

    if elements is not None and (not isinstance(elements, str) or
                                 len(elements.split("-")) != 2):
        raise ValueError("elements must be an element pair such as C-H")
    mol = perceived(molecule)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    bonds = []
    for bond in mol.GetBonds():
        pair = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        symbols = [mol.GetAtomWithIdx(index).GetSymbol() for index in pair]
        if elements is None or sorted(symbols) == sorted(elements.split("-")):
            bonds.append({"atoms": pair, "elements": symbols,
                          "symmetry_class": "-".join(map(str, sorted(ranks[index]
                                                                    for index in pair)))})
    return {"bonds": sorted(bonds, key=lambda bond: bond["atoms"])}


def homolysis(store, molecule, atoms) -> RadicalFragments:
    from rdkit import Chem

    if molecule.get("charge", 0):
        raise ValueError("Neutral homolysis cannot conserve a charged parent's charge")
    mol = perceived(molecule)
    if (not isinstance(atoms, list) or len(atoms) != 2 or
            any(type(index) is not int or not 0 <= index < mol.GetNumAtoms() for index in atoms)
            or atoms[0] == atoms[1]):
        raise ValueError("A bond requires two distinct valid atom indices")
    bond = mol.GetBondBetweenAtoms(*atoms)
    if bond is None:
        raise ValueError("Requested bond does not exist")
    editable = Chem.RWMol(mol)
    editable.RemoveBond(*atoms)
    components = Chem.GetMolFrags(editable, sanitizeFrags=False)
    if len(components) != 2:
        raise ValueError("Bond must split a connected molecule into two fragments (no rings)")
    coordinates = mol.GetConformer().GetPositions()
    fragments = []
    for component in components:
        fragments.append({"atoms": "; ".join(
            mol.GetAtomWithIdx(index).GetSymbol() + " " + " ".join(
                str(float(value)) for value in coordinates[index]) for index in component),
            "charge": 0, "spin": sum(mol.GetAtomWithIdx(index).GetAtomicNum()
                                      for index in component) % 2, "unit": "angstrom"})
    return {"fragments": fragments}


def representatives(molecule, bonds):
    selected = enumerate_bonds(None, molecule,
                               None if not isinstance(bonds, str) or bonds == "all" else bonds)["bonds"]
    if isinstance(bonds, list):
        if any(not isinstance(pair, list) or len(pair) != 2 or
               any(type(index) is not int for index in pair) for pair in bonds):
            raise ValueError("bonds must contain atom-index pairs")
        requested = {tuple(sorted(pair)) for pair in bonds}
        selected = [bond for bond in selected if tuple(bond["atoms"]) in requested]
        if {tuple(bond["atoms"]) for bond in selected} != requested:
            raise ValueError("Requested bond does not exist")
    classes = {}
    for bond in selected:
        classes.setdefault(bond["symmetry_class"], bond)
    if not classes:
        raise ValueError("No matching bonds")
    return list(classes.values())


def validate_model(model):
    model = model.model_dump() if hasattr(model, "model_dump") else model
    if (set(model) - {"basis", "hamiltonian_class", "reference", "ecp", "frozen_core",
                     "dft_functional"} or not model.get("basis") or
            model.get("hamiltonian_class", "nonrel_bo") != "nonrel_bo" or
            model.get("reference", "RHF") not in ("RHF", "ROHF") or
            model.get("ecp") or model.get("frozen_core", 0) or model.get("dft_functional")):
        raise ValueError("APP C supports full-space all-electron nonrelativistic HF/correlation")
    return model


def _solve(molecule, model, method):
    from pyscf import cc, fci, gto, mp, scf

    model = validate_model(model)
    if method not in METHODS:
        raise ValueError(f"Unsupported electronic level: {method}")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                charge=molecule.get("charge", 0), spin=molecule.get("spin", 0),
                basis=model["basis"], verbose=0)
    dimension = math.prod(math.comb(mol.nao_nr(), count) for count in mol.nelec)
    if method == "fci" and dimension > FCI_MAX_DETERMINANTS:
        raise ValueError(f"FCI determinant cap {FCI_MAX_DETERMINANTS} exceeded: {dimension}")
    mean_field = (scf.ROHF(mol) if mol.spin else scf.RHF(mol)).run(conv_tol=1e-11)
    if not mean_field.converged:
        raise ValueError("SCF did not converge")
    energy = mean_field.e_tot
    if method == "fci":
        solver = fci.FCI(mean_field)
        solver.conv_tol = 1e-11
        energy = solver.kernel()[0]
        if not solver.converged:
            raise ValueError("FCI did not converge")
    elif method == "mp2" and mol.nelectron > 1:
        energy = mp.MP2(mean_field).run().e_tot
    elif method == "ccsd_t" and mol.nelectron > 1:
        solver = cc.CCSD(mean_field).run(conv_tol=1e-10)
        if not solver.converged:
            raise ValueError("CCSD did not converge")
        energy = solver.e_tot + solver.ccsd_t()
    return float(energy), mean_field


def app_component_energy(store, molecule, electronic_model, method):
    from pyscf import ao2mo

    from chemflow.types import ConstantLedger, Hamiltonian, Sector
    from chemflow.units import Q

    energy, mean_field = _solve(molecule, electronic_model, method)
    orbitals = mean_field.mo_coeff
    mol = mean_field.mol
    ham = Hamiltonian(h1_ref=store.put(orbitals.T @ mean_field.get_hcore() @ orbitals),
        h2_ref=store.put(ao2mo.restore(1, ao2mo.kernel(mol, orbitals), orbitals.shape[1])),
        n_orb=orbitals.shape[1], space=store.put(orbitals),
        ledger=ConstantLedger().add("nuclear_repulsion", Q(float(mol.energy_nuc()), "hartree")))
    sector = Sector(n_alpha=mol.nelec[0], n_beta=mol.nelec[1], n_orb=orbitals.shape[1])
    return estimate(energy, "total_electronic_energy", "approximate",
        {"hamiltonian_id": content_hash((molecule, electronic_model))}, "app_component_energy",
        diagnostics={"hamiltonian_witness": ham.model_dump(mode="json"),
                     "sector_witness": sector.model_dump(mode="json"),
                     "fci_max_determinants": FCI_MAX_DETERMINANTS})


def _level(level, allowed):
    if (not isinstance(level, dict) or set(level) - {"method", "functional", "dispersion"}
            or level.get("method") not in allowed or level.get("functional") is not None
            or level.get("dispersion") is not None):
        raise ValueError(f"level must specify one of {allowed}, without functional or dispersion")
    return level["method"]


def site_reactivity(store, molecule, model, level) -> SiteReactivity:
    import freesasa
    from rdkit import Chem

    method = _level(level, METHODS)
    mol = perceived(molecule)
    radii = [Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    areas = freesasa.calcCoord(mol.GetConformer().GetPositions().ravel().tolist(), radii,
                              freesasa.Parameters({"probe-radius": 1.4}))
    parent = _solve(molecule, model, method)[0]
    sites = []
    for bond in enumerate_bonds(store, molecule, "C-H")["bonds"]:
        hydrogen = next(index for index in bond["atoms"]
                        if mol.GetAtomWithIdx(index).GetAtomicNum() == 1)
        fragments = homolysis(store, molecule, bond["atoms"])["fragments"]
        bde = sum(_solve(fragment, model, method)[0] for fragment in fragments) - parent
        area = float(areas.atomArea(hydrogen))
        fraction = area / (4 * math.pi * (radii[hydrogen] + 1.4) ** 2)
        kcal = bde * 627.5094740631
        sites.append({"atom": hydrogen, "bde_hartree": bde, "bde_kcal": kcal,
                      "sasa": area, "sasa_fraction": fraction,
                      "score": float(fraction * expit((90 - kcal) / 3))})
    return {"sites": sites, "top_site": max(sites, key=lambda site: site["score"])["atom"]
            if sites else None, "evidence": "empirical"}


def boltzmann_weights(store, energies, temperature_K=298.15) -> BoltzmannWeights:
    values = np.asarray(energies, dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all() or
            not math.isfinite(temperature_K) or temperature_K <= 0):
        raise ValueError("Finite nonempty energies and positive temperature are required")
    with np.errstate(over="ignore", under="ignore"):
        raw = np.exp(-(values - values.min()) / temperature_K *
                     (4.3597447222071e-18 / 1.380649e-23))
    return {"weights": (raw / raw.sum()).tolist(), "temperature_K": temperature_K}


def conformer_search(store, molecule_smiles, level, count=20, rmsd=0.3) -> ConformerSearch:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    method = _level(level, ("gfn2", "mmff"))
    if type(count) is not int or not 1 <= count <= 1000 or not math.isfinite(rmsd) or rmsd <= 0:
        raise ValueError("count must be 1..1000 and rmsd must be finite and positive")
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None or not mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("A single connected SMILES molecule is required")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        raise ValueError("Conformer search currently requires a closed-shell SMILES molecule")
    mol = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 42
    parameters.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=parameters))
    if not ids:
        raise ValueError("ETKDG failed to embed conformers")
    candidates = []
    for identifier in ids:
        if method == "mmff":
            properties = AllChem.MMFFGetMoleculeProperties(mol)
            if properties is None:
                raise ValueError("MMFF parameters unavailable")
            field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=identifier)
            if field.Minimize(maxIts=1000) != 0:
                raise ValueError("MMFF optimization failed to converge")
            energy = field.CalcEnergy() / 627.5094740631
        else:
            from ase import Atoms
            from ase.optimize import BFGS
            from ase.units import Hartree
            from tblite.ase import TBLite

            atoms = Atoms([atom.GetSymbol() for atom in mol.GetAtoms()],
                          positions=mol.GetConformer(identifier).GetPositions())
            atoms.calc = TBLite(method="GFN2-xTB", verbosity=0,
                                charge=Chem.GetFormalCharge(mol))
            if not BFGS(atoms, logfile=None).run(fmax=0.01, steps=500):
                raise ValueError("GFN2 optimization failed to converge")
            energy = atoms.get_potential_energy() / Hartree
            for index, position in enumerate(atoms.positions):
                mol.GetConformer(identifier).SetAtomPosition(index, position)
        candidates.append((float(energy), identifier))
    heavy = Chem.RemoveHs(mol)
    retained, output = [], []
    for energy, identifier in sorted(candidates):
        if any(rdMolAlign.GetBestRMS(Chem.Mol(heavy), heavy, prbId=identifier, refId=previous)
               < rmsd for previous in retained):
            continue
        retained.append(identifier)
        positions = mol.GetConformer(identifier).GetPositions()
        output.append({"energy_hartree": energy, "atoms": "; ".join(
            atom.GetSymbol() + " " + " ".join(str(float(value)) for value in positions[index])
            for index, atom in enumerate(mol.GetAtoms()))})
    return {"conformers": output, "method": method, "seed": 42,
            "energy_model": "model-defined conformer potential"}


def formula(molecule):
    return Counter(symbol for symbol, _ in _entries(molecule))


def components(target):
    if target["kind"] == "ranking":
        items = target["items"]
        if (not items or len({item["name"] for item in items}) != len(items) or
                any(formula(item["molecule"]) != formula(items[0]["molecule"])
                    or item["molecule"].get("charge", 0) != items[0]["molecule"].get("charge", 0)
                    for item in items)):
            raise ValueError("Ranking requires unique names, a shared formula and total charge")
        return [item["molecule"] for item in items]
    molecule = target["molecule"]
    return [molecule, *[fragment for bond in representatives(molecule, target["bonds"])
                       for fragment in homolysis(None, molecule, bond["atoms"])["fragments"]]]


def collection_values(target, energies):
    if target["kind"] == "ranking":
        return energies
    return [energies[index] + energies[index + 1] - energies[0]
            for index in range(1, len(energies), 2)]


def energy_collection(store, target, bar_hartree, **inputs):
    energies = [inputs[f"component{index}"].value.to("hartree").value
                for index in range(len(components(target)))]
    values = collection_values(target, energies)
    details = {"values_hartree": values}
    if target["kind"] == "ranking":
        order = sorted(range(len(values)), key=lambda index: values[index])
        bounds = [0.0 if inputs[f"component{index}"].relation == "equivalent" and
                  inputs[f"component{index}"].diagnostics.get("relation_status") == "proved"
                  else bar_hartree for index in range(len(values))]
        details.update(ranking=[target["items"][index]["name"] for index in order],
            error_bounds_hartree=bounds, certified_pairs=[
                [target["items"][left]["name"], target["items"][right]["name"]]
                for offset, left in enumerate(order) for right in order[offset + 1:]
                if bounds[left] is not None and bounds[right] is not None and
                values[right] - values[left] > bounds[left] + bounds[right]],
            certification_basis="conditional on task error bars; exact inputs use zero")
    return estimate(values[0], "energy_difference", "approximate",
                    {"hamiltonian_id": content_hash(target)}, target["kind"], details)


def base_plan(record, active=None):
    if active is not None:
        raise ValueError("APP C uses full molecular orbital spaces")
    target = record.spec.target.model_dump(exclude={"operator"})
    species = components(target)
    bar = task_bar(record)
    with build.program("app-c-controls") as builder:
        controls = []
        for method in METHODS:
            inputs = {f"component{index}": builder.node("app_component_energy", molecule=molecule,
                      electronic_model=target["electronic_model"], method=method)
                      for index, molecule in enumerate(species)}
            result = builder.node(target["kind"], target=target, bar_hartree=bar, **inputs)
            if method == "fci":
                builder.ir.references = [result.id]
            else:
                controls.append(result.id)
        builder.ir.targets = [*controls, *builder.ir.references]
        return builder.to_ir(), controls


def task_bar(record):
    from chemflow.units import Q

    bars = [Q(gate.threshold, gate.unit).to("hartree").value for gate in record.spec.gates
            if gate.kind == "abs_error_vs_exact"]
    return min(bars) if bars else None


def identity(plan, record, contexts):
    from .scans import plan_identity

    expected, _ = base_plan(record)
    return plan_identity(plan, expected)


def resources(**kwargs):
    return {"wall_s": None, "cpu_core_s": None, "gpu_s": 0, "qpu_s": 0,
            "qpu_shots": 0, "sim_shots": 0}


def parameter_errors(node):
    parameters = node.params
    failures = []
    if node.op == "enumerate_bonds" and parameters.get("elements") is not None:
        pair = parameters["elements"]
        if not isinstance(pair, str) or len(pair.split("-")) != 2:
            failures.append("elements must be an element pair")
    if node.op == "homolysis" and "atoms" in parameters:
        pair = parameters["atoms"]
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(index) is not int or index < 0 for index in pair) or pair[0] == pair[1]):
            failures.append("atoms must contain two distinct nonnegative integer indices")
    if node.op == "boltzmann_weights" and "energies" in parameters:
        energies = parameters["energies"]
        if (not isinstance(energies, list) or not energies or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in energies)):
            failures.append("energies must be a nonempty list of finite numbers")
    if node.op in ("site_reactivity", "conformer_search") and "level" in parameters:
        try:
            _level(parameters["level"], METHODS if node.op == "site_reactivity" else ("gfn2", "mmff"))
        except ValueError as error:
            failures.append(str(error))
    if node.op == "app_component_energy":
        if "method" in parameters and parameters["method"] not in METHODS:
            failures.append("Unsupported electronic method")
        if "electronic_model" in parameters:
            try:
                validate_model(parameters["electronic_model"])
            except (ValueError, TypeError, AttributeError):
                failures.append("Unsupported electronic model")
    if node.op == "conformer_search":
        count = parameters.get("count", 20)
        if type(count) is not int or not 1 <= count <= 1000:
            failures.append("count must be an integer in 1..1000")
    for name in ({"rmsd"} if node.op == "conformer_search" else
                 {"temperature_K"} if node.op == "boltzmann_weights" else set()):
        value = parameters.get(name, 1)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            failures.append(f"{name} must be finite and positive")
    if node.op in ("ranking", "bond_dissociation"):
        bar = parameters.get("bar_hartree")
        if bar is not None and (type(bar) not in (int, float) or not math.isfinite(bar) or bar < 0):
            failures.append("bar_hartree must be finite and nonnegative")
        if "target" in parameters:
            from chemflow.agent.task import BondDissociationTarget, RankingTarget

            try:
                (RankingTarget if node.op == "ranking" else BondDissociationTarget).model_validate(
                    parameters["target"])
            except ValueError as error:
                failures.append(str(error))
    return failures


def context_check(node, inputs, context, nodes):
    if node.op == "app_component_energy":
        context.update(kind="total_electronic_energy", unit="hartree",
                       hamiltonian_id=content_hash((node.params["molecule"],
                                                   node.params["electronic_model"])))
    if node.op in ("ranking", "bond_dissociation"):
        context.update(kind="energy_difference", unit="hartree",
                       hamiltonian_id=content_hash(node.params["target"]))
        species = components(node.params["target"])
        if set(inputs) != {f"component{index}" for index in range(len(species))}:
            raise ValueError("Energy collection requires every component exactly once")
        for index, molecule in enumerate(species):
            expected = content_hash((molecule, node.params["target"]["electronic_model"]))
            if inputs[f"component{index}"].get("hamiltonian_id") != expected:
                from chemflow.ir.schema import Node

                from .scans import signature

                reference = Node(id="expected_component", op="app_component_energy", params={
                    "molecule": molecule,
                    "electronic_model": node.params["target"]["electronic_model"]})
                if signature(nodes, node.inputs[f"component{index}"], True) != signature(
                        {reference.id: reference}, reference.id, True):
                    raise ValueError("Collection component identity differs from its target")
    return []


def register(registry):
    specs = [
        ("enumerate_bonds", enumerate_bonds, {"molecule": "molecule"}, "bond_enumeration",
         {"elements": "str | null"}, "RDKit perceived bonds and canonical symmetry classes"),
        ("homolysis", homolysis, {"molecule": "molecule"}, "radical_fragments",
         {"atoms": "list[int]"}, "Two neutral rigid fragments by homolytic bond cleavage"),
        ("site_reactivity", site_reactivity, {"molecule": "molecule", "model": "model"},
         "site_reactivity", {"level": "dict"}, "Empirical C-H BDE and FreeSASA reactivity heuristic"),
        ("boltzmann_weights", boltzmann_weights, {}, "boltzmann_weights",
         {"energies": "list[float]", "temperature_K": "float"},
         "Model-defined canonical Boltzmann populations for supplied hartree energies"),
        ("conformer_search", conformer_search, {}, "conformer_search",
         {"molecule_smiles": "str", "level": "dict", "count": "int", "rmsd": "float"},
         "Seeded ETKDGv3, GFN2 or MMFF optimization and heavy-atom RMSD deduplication"),
        ("app_component_energy", app_component_energy, {}, "estimate",
         {"molecule": "dict", "electronic_model": "dict", "method": "str"},
         "Full-space radical-capable HF, MP2, CCSD(T), capped FCI component energies"),
        *[(kind, energy_collection, {}, "estimate", {"target": "dict", "bar_hartree": "float | null"},
           "Rigid symmetry-class bond dissociation energies (BDE)" if kind == "bond_dissociation" else
           "Same-formula energy ranking with conditional error-bar certified pairs")
          for kind in ("bond_dissociation", "ranking")]]
    for name, adapter, inputs, output, parameters, description in specs:
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
            output=output, parameters=parameters, semantics=description))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
            backend_kind="cpu_numeric", execution_role="classical", resource_model=resources,
            params=frozenset(parameters), implements=f"app_c_{name}"))
        registry.register(RefinementRule(target=name, template_id=f"app_c_{name}",
                                         relation="approximate"))
    registry.context_checks.append(context_check)
