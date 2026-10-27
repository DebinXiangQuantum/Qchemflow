"""Prepared protein–ligand structures and explicitly model-defined binding energies."""

import io

import numpy as np

from chemflow import build
from chemflow.context import Context
from chemflow.diagnostics import Diagnostic
from chemflow.ids import content_hash
from chemflow.registry import RefinementRule
from chemflow.runtime.artifacts import read_text_artifact, write_text_artifact
from chemflow.types import CostRecord, Estimate, Provenance
from chemflow.units import Q
from chemflow.verify.identity_diff import identity_hints, signature_tree

from ._adapters import register_adapter
from ._tools_client import call_tools


class BindingEstimate(Estimate):
    dG_kcal: float
    components: dict[str, float]
    stderr: float | None
    frames: int


def protein_structure(pdb_text, ph=7.0, keep=None, remove_water=True, chains=None, store=None):
    result = call_tools("protein_structure", {"pdb_text": pdb_text, "ph": ph, "keep": keep or [],
                        "remove_water": remove_water, "chains": chains})
    result["pdb_ref"] = write_text_artifact(result.pop("pdb_text"), ".pdb")
    return result


def ligand_structure(source, store=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    residue_name = "LIG"
    if "pdb_text" in source:
        residue_name = source.get("residue", "LIG")
        lines = [line for line in source["pdb_text"].splitlines()
                 if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() == residue_name]
        identifiers = {(line[21:22], line[22:27]) for line in lines}
        if len(identifiers) != 1:
            raise ValueError("Ligand residue must identify exactly one PDB residue")
        molecule = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=False)
        if molecule is not None and source.get("smiles"):
            template = Chem.MolFromSmiles(source["smiles"])
            molecule = AllChem.AssignBondOrdersFromTemplate(template, molecule)
    elif "sdf" in source:
        molecule = Chem.MolFromMolBlock(source["sdf"], removeHs=False)
    elif "xyz" in source:
        molecule = Chem.MolFromXYZBlock(source["xyz"])
        if molecule is not None:
            rdDetermineBonds.DetermineBonds(molecule, charge=source.get("charge", 0))
    else:
        molecule = Chem.MolFromSmiles(source["smiles"])
    if molecule is None:
        raise ValueError("Cannot parse ligand source")
    Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule, addCoords=True)
    if not molecule.GetNumConformers():
        options = AllChem.ETKDGv3()
        options.randomSeed = 0
        if AllChem.EmbedMolecule(molecule, options) != 0:
            raise ValueError("Ligand embedding failed")
    sdf = Chem.MolToMolBlock(molecule) + "\n$$$$\n"
    return {"sdf_ref": write_text_artifact(sdf, ".sdf"),
            "smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)),
            "formal_charge": Chem.GetFormalCharge(molecule), "n_atoms": molecule.GetNumAtoms(),
            "residue_name": residue_name}


def complex(protein, ligand, store=None):
    from openmm import Vec3, app, unit
    from rdkit import Chem

    pdb = app.PDBFile(io.StringIO(read_text_artifact(protein["pdb_ref"])))
    molecule = Chem.MolFromMolBlock(read_text_artifact(ligand["sdf_ref"]), removeHs=False)
    topology = app.Topology()
    chain = topology.addChain("L" if "L" not in {chain.id for chain in pdb.topology.chains()} else "Z")
    residue = topology.addResidue(ligand["residue_name"], chain, "1")
    atoms = [topology.addAtom(f"{atom.GetSymbol()}{atom.GetIdx()+1}",
                             app.Element.getBySymbol(atom.GetSymbol()), residue)
             for atom in molecule.GetAtoms()]
    for bond in molecule.GetBonds():
        topology.addBond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    positions = [Vec3(*xyz) for xyz in molecule.GetConformer().GetPositions()] * unit.angstrom
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.add(topology, positions)
    stream = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    return {"pdb_ref": write_text_artifact(stream.getvalue(), ".pdb"),
            "ligand_residue": f"{chain.id}:1", "ligand": ligand,
            "protonation": protein["protonation"], "protonation_method": protein["protonation_method"]}


def _residue_atoms(complex):
    from openmm import app, unit

    pdb = app.PDBFile(io.StringIO(read_text_artifact(complex["pdb_ref"])))
    coordinates = pdb.positions.value_in_unit(unit.angstrom)
    return [(residue, [(atom.name, atom.element.symbol, list(coordinates[atom.index]))
                      for atom in residue.atoms()]) for residue in pdb.topology.residues()]


def pocket(complex, cutoff_A=4.0, store=None):
    groups = _residue_atoms(complex)
    ligand = next(atoms for residue, atoms in groups
                  if f"{residue.chain.id}:{residue.id}" == complex["ligand_residue"])
    coordinates = np.asarray([xyz for _, _, xyz in ligand])
    residues = []
    for residue, atoms in groups:
        identifier = f"{residue.chain.id}:{residue.id}"
        if identifier == complex["ligand_residue"]:
            continue
        distance = np.linalg.norm(np.asarray([xyz for _, _, xyz in atoms])[:, None] - coordinates,
                                  axis=2).min()
        if distance <= cutoff_A:
            residues.append({"id": identifier, "name": residue.name, "chain": residue.chain.id,
                             "number": residue.id})
    return {"residues": residues, "cutoff_A": cutoff_A}


def ligand_parameters(ligand, forcefield="gaff-2.11", charge_method="am1bcc", store=None):
    result = call_tools("ligand_parameters", {"sdf": read_text_artifact(ligand["sdf_ref"]),
                        "forcefield": forcefield, "charge_method": charge_method})
    if abs(sum(result["charges"]) - ligand["formal_charge"]) > 1e-6:
        raise ValueError("Ligand parameter charges do not sum to its formal charge")
    result["parameters_ref"] = write_text_artifact(result.pop("ffxml"), ".xml")
    result["ligand"] = ligand
    return result


def mmgbsa(complex, parameters, frames=0, minimize=False, seed=0, store=None):
    if parameters["ligand"] != complex["ligand"]:
        raise ValueError("Ligand parameters do not match the complex")
    result = call_tools("mmgbsa", {"pdb": read_text_artifact(complex["pdb_ref"]),
                        "ffxml": read_text_artifact(parameters["parameters_ref"]), "frames": frames,
                        "minimize": minimize, "seed": seed})
    target = content_hash({"complex": complex, "parameters": parameters, "frames": frames,
                           "minimize": minimize, "seed": seed})
    return BindingEstimate(**result, value=Q(result["dG_kcal"], "kcal/mol"),
        kind="binding_free_energy", target=target, relation="equivalent",
        context=Context(keys={"kind": "binding_free_energy", "unit": "kcal/mol",
                              "hamiltonian_id": target}),
        provenance=Provenance(impl_id="mmgbsa", impl_version="1", backend_kind="cpu_numeric",
                              device="OpenMM Reference", seed=seed), cost=CostRecord(),
        diagnostics={"model": "Amber14/GBn2 single-trajectory MM/GBSA",
                     "experimental_relation": "model_change",
                     "protonation_method": complex["protonation_method"],
                     "sampling": "naive frame standard error; correlated frames not corrected",
                     "entropy": "No configurational entropy correction"})


def fragment_interaction(complex, ligand, residues, level=None, basis="sto-3g",
                         counterpoise=True, store=None):
    from pyscf import dft, gto, mp, scf

    level = level or {"method": "hf"}
    groups = _residue_atoms(complex)
    ligand_atoms = [(element, xyz) for residue, atoms in groups
                    if f"{residue.chain.id}:{residue.id}" == complex["ligand_residue"]
                    for _, element, xyz in atoms]
    if ligand != complex["ligand"]:
        raise ValueError("Fragment ligand differs from complex ligand")
    available = {f"{residue.chain.id}:{residue.id}": (residue, atoms) for residue, atoms in groups}
    if set(residues) - available.keys() or complex["ligand_residue"] in residues:
        raise ValueError("Unknown protein fragment residue")
    values, residue_atoms, charges = {}, {}, {}
    excluded = {"N", "CA", "C", "O", "H", "HA", "OXT", "H2", "H3"}
    for identifier in residues:
        residue, atoms = available[identifier]
        if residue.name in ("GLY", "PRO"):
            continue
        residue_atoms[identifier] = atoms
        names = {name: np.asarray(xyz) for name, _, xyz in atoms}
        if "CB" not in names:
            raise ValueError("Fragment lacks CB: consider an explicitly defined SQD active region")
        direction = names["CA"] - names["CB"]
        side = [(element, xyz) for name, element, xyz in atoms if name not in excluded]
        side.append(("H", (names["CB"] + 1.09 * direction / np.linalg.norm(direction)).tolist()))
        state = complex["protonation"].get(identifier, {}).get("state", residue.name)
        side_charge = {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1,
                       "HIP": 1, "CYM": -1, "TYM": -1}.get(state, 0)
        charges[identifier] = side_charge

        def energy(ligand_ghost, side_ghost, side=side, side_charge=side_charge):
            atoms = [("ghost-" + element if ghost else element, xyz)
                     for fragment, ghost in ((ligand_atoms, ligand_ghost), (side, side_ghost))
                     for element, xyz in fragment if counterpoise or not ghost]
            charge = (0 if ligand_ghost else ligand["formal_charge"]) + (
                      0 if side_ghost else side_charge)
            electrons = sum(gto.charge(element) for element, _ in atoms
                            if not element.startswith("ghost-")) - charge
            molecule = gto.M(atom=atoms, basis=basis, charge=charge, spin=electrons % 2, verbose=0)
            method = level.get("method", "hf").lower()
            if method == "dft":
                meanfield = dft.ROKS(molecule) if molecule.spin else dft.RKS(molecule)
                meanfield.xc = level.get("functional") or "pbe"
            else:
                meanfield = scf.ROHF(molecule) if molecule.spin else scf.RHF(molecule)
            value = meanfield.kernel()
            if not meanfield.converged:
                raise ValueError("Fragment SCF did not converge")
            if method == "mp2":
                value += mp.MP2(meanfield).kernel()[0]
            return float(value)

        values[identifier] = energy(False, False) - energy(False, True) - energy(True, False)
    return {"per_residue": values, "total": sum(values.values()), "unit": "hartree",
            "ligand_atoms": ligand_atoms, "residue_atoms": residue_atoms, "formal_charges": charges,
            "approximation": "Pairwise-additive capped side-chain approximation; no many-body, "
                             "backbone, solvation or entropy terms; not binding free energy."}


def resources(**kwargs):
    return {"wall_s": None, "cpu_core_s": None, "gpu_s": 0, "sim_shots": 0,
            "qpu_shots": 0, "qpu_s": 0}


def validate_parameters(operation, parameters):
    errors = []
    for name in ("frames", "seed"):
        if name in parameters and (type(parameters[name]) is not int or parameters[name] < 0):
            errors.append(f"{name} must be a nonnegative integer")
    for name in ("minimize", "remove_water", "counterpoise"):
        if name in parameters and type(parameters[name]) is not bool:
            errors.append(f"{name} must be a boolean")
    for name in ("ph", "cutoff_A"):
        if name in parameters:
            value = parameters[name]
            if (type(value) not in (int, float) or not np.isfinite(value) or
                    (name == "ph" and not 0 <= value <= 14) or
                    (name == "cutoff_A" and value <= 0)):
                errors.append(f"Invalid {name}")
    for name in ("keep", "chains", "residues"):
        if name in parameters and (not isinstance(parameters[name], list) or
                any(not isinstance(item, str) for item in parameters[name])):
            errors.append(f"{name} must be a list of strings")
    if "forcefield" in parameters and parameters["forcefield"] not in ("gaff-2.11", "openff-2.2.0"):
        errors.append("Use gaff-2.11 or openff-2.2.0")
    if "charge_method" in parameters and parameters["charge_method"] != "am1bcc":
        errors.append("Only AM1-BCC charges are supported")
    if "pdb_text" in parameters and not isinstance(parameters["pdb_text"], str):
        errors.append("pdb_text must be a string")
    if "source" in parameters:
        source = parameters["source"]
        if not isinstance(source, dict):
            errors.append("Ligand source must be a dictionary")
        elif (set(source) - {"pdb_text", "residue", "smiles", "sdf", "xyz", "charge"} or
              not any(isinstance(source.get(name), str) and source[name]
                      for name in ("pdb_text", "smiles", "sdf", "xyz"))):
            errors.append("Ligand source requires PDB, SMILES, SDF or XYZ")
    if "level" in parameters:
        level = parameters["level"]
        if (not isinstance(level, dict) or set(level) - {"method", "functional", "dispersion"} or
                level.get("method", "hf") not in ("hf", "dft", "mp2") or level.get("dispersion")):
            errors.append("Fragment level supports HF, DFT or MP2 without dispersion")
    if "basis" in parameters and not isinstance(parameters["basis"], str):
        errors.append("basis must be a string")
    return errors


def context_check(node, inputs, context, nodes):
    if node.op == "mmgbsa":
        context.update(kind="binding_free_energy", unit="kcal/mol",
                       hamiltonian_id=content_hash({"inputs": inputs, "params": node.params}),
                       relation="equivalent")
    return []


def base_plan(record, active=None):
    if active is not None:
        raise ValueError("MM/GBSA has no electronic active space")
    target = record.spec.target
    with build.program("binding-controls") as builder:
        protein = builder.node("protein_structure", **target.receptor,
                               ph=target.ph, keep=[], remove_water=True)
        ligand = builder.node("ligand_structure", source=target.ligand["source"])
        assembly = builder.node("complex", protein=protein, ligand=ligand)
        parameters = builder.node("ligand_parameters", ligand=ligand,
                                  forcefield=target.forcefield, charge_method="am1bcc")
        arguments = {"complex": assembly, "parameters": parameters, "frames": target.frames,
                     "minimize": target.minimize, "seed": target.seed}
        control = builder.node("mmgbsa", **arguments)
        reference = builder.node("mmgbsa", **arguments)
        builder.ir.targets = [control.id, reference.id]
        builder.ir.references = [reference.id]
        return builder.to_ir(), [control.id]


def identity(plan, record, contexts):
    expected, _ = base_plan(record)
    expected_signature = signature_tree(expected, expected.targets[0])
    issues = []
    for identifier in set(plan.targets + plan.references):
        actual = signature_tree(plan, identifier)
        valid = actual == expected_signature
        if not valid:
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                                     message="Binding target identity mismatch",
                                     fix_hints=identity_hints(actual, [expected_signature])))
    return issues


def register(registry):
    for name, adapter, inputs, output, parameters, description in (
        ("protein_structure", protein_structure, {}, "protein_structure",
         ("pdb_text", "ph", "keep", "remove_water", "chains"),
         "PDBFixer protein preparation with recorded pH-dependent protonation"),
        ("ligand_structure", ligand_structure, {}, "ligand_structure", ("source",),
         "RDKit ligand bond orders, formal charge and coordinate-bearing explicit hydrogens"),
        ("complex", complex, {"protein": "protein_structure", "ligand": "ligand_structure"},
         "protein_ligand_complex", (), "Assemble prepared protein and ligand without moving the pose"),
        ("pocket", pocket, {"complex": "protein_ligand_complex"}, "binding_pocket", ("cutoff_A",),
         "Protein residues within the declared angstrom cutoff of any ligand atom"),
        ("ligand_parameters", ligand_parameters, {"ligand": "ligand_structure"}, "ligand_parameters",
         ("forcefield", "charge_method"), "GAFF2 or OpenFF AM1-BCC ligand force-field parameters"),
        ("mmgbsa", mmgbsa, {"complex": "protein_ligand_complex", "parameters": "ligand_parameters"},
         "estimate", ("frames", "minimize", "seed"),
         "Model-defined binding_free_energy: Amber14/GBn2 MM/GBSA; model_change to experiment"),
        ("fragment_interaction", fragment_interaction,
         {"complex": "protein_ligand_complex", "ligand": "ligand_structure"}, "fragment_interaction",
         ("residues", "level", "basis", "counterpoise"),
         "Pairwise-additive ligand side-chain PySCF counterpoise interaction; not binding free energy"),
    ):
        register_adapter(registry, name, adapter, inputs, output, parameters, description)
        registry.implementations[name].resource_model = resources
    registry.register(RefinementRule(target="mmgbsa", template_id="binding_mmgbsa",
                                     relation="equivalent"))
    registry.register(RefinementRule(target="fragment_interaction", template_id="binding_fragment_qm",
                                     relation="approximate"))
    registry.context_checks.append(context_check)
