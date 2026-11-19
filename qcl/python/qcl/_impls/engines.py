"""Opaque engines: RDKit, tblite GFN2, OpenMM MM/GBSA, metabolism and solubility."""

from qcl._impls import implements
from qcl._impls._common import (
    _molecule_identity,
)


@implements("metabolism.sites")
def _op_metabolism_sites(ctx, node, identifier, operation, inputs):
    import pyscf

    values = ctx.values
    store = ctx.store
    from chemflow.domains.reactivity import site_reactivity

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Metabolic site scoring currently requires a CPU")
    params = node["params"]
    result = site_reactivity(store, inputs[0], params["model"],
                             {"method": params["level"]})
    values[identifier] = {**result, "provenance": {
        "question": {"property": "metabolic_sites", "molecule": inputs[0]},
        "method": {"family": "C-H BDE/FreeSASA site scoring",
                   "electronic_model": params["model"], "level": params["level"]},
        "implementation": "metabolism.sites/chemflow-reactivity",
        "device": device["id"], "versions": {"pyscf": pyscf.__version__},
        "validity_flags": ["Empirical site ranking only",
                           "No enzyme or metabolic-rate prediction"],
    }}


@implements("binding.mmgbsa")
def _op_binding_mmgbsa(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.domains import binding

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("MM/GBSA currently requires a CPU")
    params = node["params"]
    receptor = params["receptor"]
    ligand = binding.ligand_structure(params["ligand"]["source"])
    protein = binding.protein_structure(
        receptor["pdb_text"], ph=params["ph"], chains=receptor.get("chains"))
    combined = binding.complex(protein, ligand)
    parameters = binding.ligand_parameters(ligand, forcefield=params["forcefield"])
    result = binding.mmgbsa(combined, parameters, frames=params["frames"],
                            minimize=params["minimize"], seed=params["seed"])
    stderr = result.stderr
    values[identifier] = {
        "value": result.dG_kcal, "unit": "kcal/mol",
        "variance": stderr**2 if stderr is not None else 0.0,
        "provenance": {
            "question": {"property": "binding_free_energy",
                         "receptor": receptor, "ligand": params["ligand"]},
            "method": {"family": "Amber14/GBn2 MM/GBSA",
                       "forcefield": params["forcefield"],
                       "frames": params["frames"], "minimize": params["minimize"],
                       "seed": params["seed"], "ph": params["ph"]},
            "implementation": "binding.mmgbsa/chemflow-openmm",
            "device": device["id"], "versions": {},
            "components": result.components, "stderr": stderr,
            "validity_flags": ["No configurational entropy correction",
                               "Model-defined binding energy, not experimental affinity"],
        },
    }


@implements("rdkit.embed")
def _op_rdkit_embed(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from rdkit import Chem
    from rdkit.Chem import AllChem

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("RDKit embedding requires a CPU")
    params = node["params"]
    molecule = Chem.MolFromSmiles(params["smiles"])
    if (molecule is None or not molecule.GetNumHeavyAtoms() or
            len(Chem.GetMolFrags(molecule)) != 1 or
            any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())):
        raise ValueError("Embedding requires one closed-shell SMILES molecule")
    molecule = Chem.AddHs(molecule)
    options = AllChem.ETKDGv3()
    options.randomSeed = params["seed"]
    options.numThreads = 1
    conformer_ids = list(AllChem.EmbedMultipleConfs(
        molecule, numConfs=params["count"], params=options))
    if not conformer_ids:
        raise ValueError("ETKDG failed to embed conformers")

    def atoms_for(conf_id, molecule=molecule):
        coordinates = molecule.GetConformer(conf_id).GetPositions()
        return "; ".join(
            atom.GetSymbol() + " " + " ".join(str(float(x)) for x in coordinates[index])
            for index, atom in enumerate(molecule.GetAtoms()))

    values[identifier] = [{"atoms": atoms_for(conf_id), "basis": params["basis"],
                           "charge": Chem.GetFormalCharge(molecule), "spin": 0,
                           "unit": "angstrom", "rdkit_mol": molecule,
                           "conf_id": conf_id,
                           "conformer_provenance": {
                               "smiles": params["smiles"],
                               "embedding": "ETKDGv3",
                               "embedding_seed": params["seed"],
                               "conformer_id": conf_id}}
                          for conf_id in conformer_ids]


@implements("rdkit.mmff_optimize")
def _op_rdkit_mmff_optimize(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from rdkit.Chem import AllChem

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("MMFF optimization requires a CPU")
    conformer = inputs[0]
    molecule = conformer["rdkit_mol"]
    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    if properties is None:
        raise ValueError("MMFF parameters unavailable")
    field = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, confId=conformer["conf_id"])
    if field.Minimize(maxIts=1000) != 0:
        raise ValueError("MMFF optimization did not converge")
    coordinates = molecule.GetConformer(conformer["conf_id"]).GetPositions()
    atoms = "; ".join(
        atom.GetSymbol() + " " + " ".join(str(float(x)) for x in coordinates[index])
        for index, atom in enumerate(molecule.GetAtoms()))
    values[identifier] = {**conformer, "atoms": atoms,
                          "energy_hartree": field.CalcEnergy() / 627.5094740631,
                          "conformer_provenance": {
                              **conformer["conformer_provenance"],
                              "optimization": "MMFF94",
                              "mmff_energy_hartree": (
                                  field.CalcEnergy() / 627.5094740631)}}


@implements("rdkit.dedupe")
def _op_rdkit_dedupe(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("RDKit deduplication requires a CPU")
    conformers = sorted(inputs[0], key=lambda item: item["energy_hartree"])
    if not conformers:
        values[identifier] = []
    else:
        heavy = Chem.RemoveHs(conformers[0]["rdkit_mol"])
        retained = []
        for conformer in conformers:
            if any(rdMolAlign.GetBestRMS(
                    Chem.Mol(heavy), heavy, prbId=conformer["conf_id"],
                    refId=previous["conf_id"]) < node["params"]["rmsd"]
                   for previous in retained):
                continue
            retained.append({**conformer, "conformer_provenance": {
                **conformer["conformer_provenance"],
                "rmsd_threshold": node["params"]["rmsd"]}})
        values[identifier] = retained


@implements("solubility.esol", "solubility.gse")
def _op_solubility_esol(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.empirical import evaluate, model_card

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError("Solubility models currently require a CPU")
    model = operation.removeprefix("solubility.")
    params = node["params"]
    evaluated = evaluate(model, params["smiles"], params["melting_point_C"])
    card = model_card(model)
    values[identifier] = {
        "value": evaluated["value"], "unit": evaluated["unit"], "variance": 0.0,
        "provenance": {
            "question": {"property": "aqueous_solubility",
                         "smiles": params["smiles"]},
            "method": {"family": model.upper(),
                       "melting_point_C": params["melting_point_C"]},
            "implementation": operation + "/chemflow-empirical",
            "device": device["id"], "versions": card["software_versions"],
            "model_card": {"dataset_sha256": card["dataset"]["sha256"],
                           "train_rmse": card["train_rmse"],
                           "cv_rmse": card["cv_rmse"]},
            "validity_status": evaluated["validity_status"],
            "max_tanimoto": evaluated["max_tanimoto"],
        },
    }


@implements("gfn2.energy")
def _op_gfn2_energy(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from importlib.metadata import version

    from pyscf import gto
    from tblite.interface import Calculator

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"Unsupported GFN2 device: {device['kind']}")
    molecule = inputs[0]
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    charges = system.atom_charges()
    if any(charge == 0 for charge in charges):
        raise ValueError("GFN2 does not support ghost atoms")
    calculation = Calculator("GFN2-xTB", charges, system.atom_coords(),
                             charge=molecule["charge"], uhf=molecule["spin"])
    calculation.set("verbosity", 0)
    solvent = node["params"].get("solvent")
    if solvent is not None:
        calculation.add(f"{solvent['model']}-solvation", solvent["name"])
    energy = float(calculation.singlepoint().get("energy"))
    values[identifier] = {
        "value": energy, "unit": "hartree", "variance": 0.0,
        "provenance": {
            "question": {"atoms": molecule["atoms"],
                         "charge": molecule["charge"],
                         "spin": molecule["spin"], "unit": molecule["unit"],
                         "property": "electronic_energy"},
            "method": {"family": "GFN2-xTB",
                       **({"solvent": solvent} if solvent is not None else {})},
            "implementation": "gfn2.energy/chemflow-tblite",
            "device": device["id"],
            "versions": {"tblite": version("tblite")},
            "validity_flags": ["Semiempirical GFN2-xTB model"],
        },
    }


@implements("gfn2.optimize")
def _op_gfn2_optimize(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from importlib.metadata import version

    from ase import Atoms
    from ase.optimize import BFGS
    from ase.units import Hartree
    from pyscf import gto

    from chemflow.domains._path_calculator import isolated_tblite
    from chemflow.domains.paths import coordinates, molecular_result

    device = node["device"]
    if device["kind"] != "cpu":
        raise ValueError(f"Unsupported GFN2 device: {device['kind']}")
    molecule = _molecule_identity(inputs[0])
    system = gto.M(atom=molecule["atoms"], basis=molecule["basis"],
                   charge=molecule["charge"], spin=molecule["spin"],
                   unit=molecule["unit"], verbose=0)
    if any(charge == 0 for charge in system.atom_charges()):
        raise ValueError("GFN2 does not support ghost atoms")
    symbols, positions = coordinates(molecule)
    atoms = Atoms(symbols, positions=positions)
    solvent = node["params"].get("solvent")
    atoms.calc = isolated_tblite(molecule, solvent=solvent)
    try:
        if not BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500):
            raise ValueError("GFN2 optimization did not converge")
        geometry = molecular_result(molecule, atoms.get_chemical_symbols(),
                                    atoms.positions)
        energy = float(atoms.get_potential_energy() / Hartree)
    finally:
        atoms.calc.close()
    values[identifier] = {**_molecule_identity({**geometry,
                                                "basis": molecule["basis"]}),
                          "provenance": {
                              "question": {"property": "optimized_geometry",
                                           "starting_molecule": molecule},
                              "method": {"family": "GFN2-xTB", "kind": "minimum",
                                         "energy_hartree": energy,
                                         **({"solvent": solvent}
                                            if solvent is not None else {})},
                              "implementation": "gfn2.optimize/chemflow-tblite",
                              "device": device["id"],
                              "versions": {"tblite": version("tblite")},
                              "validity_flags": ["Semiempirical GFN2-xTB model"],
                          }}
