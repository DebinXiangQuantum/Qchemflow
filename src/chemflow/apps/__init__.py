"""Unsigned application drafts. Only a human may approve their gates and budget."""

import json
import math
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from chemflow.agent.task import TaskSpec

_ROWS = (
    ("ground_state_energy", "分子基态能量", "Molecular ground-state energy",
     ["molecular_ground_energy"], "Fixed-model ground energy; sampled VQE on MQ02 obeys R130."),
    ("stability_ranking", "多分子稳定性排序", "Isomer stability ranking", ["ranking"],
     "Rank same-formula isomers by electronic energy, not finite-temperature stability."),
    ("conformer_ranking", "多构象能量排序", "Conformer energy ranking", ["ranking"],
     "Deterministic ETKDG/MMFF inputs followed by electronic ranking; no draft accuracy claim."),
    ("pes_scan", "势能面扫描", "Potential-energy scan", ["pes_scan"],
     "Rigid or relaxed internal-coordinate scan with a signed validity interval."),
    ("metabolic_sites", "代谢位点预测", "Metabolic site screening", ["metabolic_sites"],
     "Empirical C-H BDE/SASA heuristic, not enzyme selectivity or metabolic kinetics."),
    ("bond_dissociation", "键解离能预测", "Bond dissociation energies", ["bond_dissociation"],
     "Rigid homolytic dissociation energies for all bonds or a selected filter."),
    ("solvation_scan", "溶剂化自由能扫描", "Solvation free-energy scan",
     ["solvation_free_energy"], "One fixed-geometry model energy difference per solvent."),
    ("frontier_orbitals", "前线轨道能量预测", "Frontier orbital energies",
     ["frontier_orbitals", "ionization_energy", "electron_affinity"],
     "HOMO/LUMO/orbital gap; optional vertical IP and EA define the fundamental gap IP-EA."),
    ("aqueous_solubility", "水溶度快速评估", "Aqueous solubility screening",
     ["aqueous_solubility"], "ESOL or GSE model-defined logS, not experimental accuracy."),
    ("reaction_barrier", "反应能垒估算", "Reaction barrier", ["reaction_barrier"],
     "Connected transition-state verification; optional RRHO Eyring rate, not kinetics validation."),
    ("dimer_binding_scan", "二聚体结合能(刚性扫描)", "Rigid dimer interaction scan",
     ["interaction_scan"], "Counterpoise-corrected rigid interaction curve, not binding free energy."),
    ("protein_ligand_binding", "蛋白质-配体结合能", "Protein-ligand binding",
     ["binding_free_energy"], "Model-defined MM/GBSA from PDB and ligand; no entropy correction."),
    ("lattice_superconductivity", "超导材料模拟(格点模型)", "Lattice superconductivity models",
     ["pairing_correlation", "lattice_ground_energy"],
     ("Finite-cluster s-wave pair and spin correlations and energy. Tc for unconventional "
      "superconductors is not predicted; Allen-Dynes is only for user/EPW inputs.")),
)
_DEFAULTS = {
    "basis": "sto-3g", "error": 0.0016, "wall_s": 1800,
}
_PARAMETERS = {
    "ground_state_energy": {"qpu": False, "qpu_shots": 100000, "qpu_s": 300},
    "stability_ranking": {},
    "conformer_ranking": {"count": 8, "rmsd": 0.05, "conformer_method": "mmff"},
    "pes_scan": {"mode": "rigid"},
    "metabolic_sites": {"method": "hf", "error": 1e-6},
    "bond_dissociation": {"bonds": "all"},
    "solvation_scan": {"solvents": [{"model": "alpb", "name": "water"},
                                      {"model": "alpb", "name": "octanol"},
                                      {"model": "pcm", "name": "water", "eps": 78.3553},
                                      {"model": "pcm", "name": "octanol", "eps": 9.8629}]},
    "frontier_orbitals": {"method": "hf", "physical_gap": False},
    "aqueous_solubility": {"error": 0.01},
    "reaction_barrier": {"rate_constant": False, "temperature_K": 298.15},
    "dimer_binding_scan": {"separations": [3.0, 4.0], "counterpoise": True},
    "protein_ligand_binding": {"frames": 0, "minimize": False, "ph": 7.0,
                                 "seed": 0, "forcefield": "gaff-2.11", "error": 0.01},
    "lattice_superconductivity": {"site_pair": [0, 1], "error": 1e-6},
}


def catalog():
    """Return independent catalogue records with declared input formats and defaults."""
    examples = json.loads(files(__package__).joinpath("examples/inputs.json").read_text())
    return [{"id": identifier, "name_zh": chinese, "name_en": english,
             "description": description, "target_kinds": list(kinds),
             "inputs": ({"smiles": "SMILES text"} if identifier in
                        ("conformer_ranking", "aqueous_solubility") else
                        {"lattice": "lattice", "sector": "particle sector"} if identifier ==
                        "lattice_superconductivity" else
                        {"receptor": "pdb", "ligand": "residue/smiles/sdf"} if identifier ==
                        "protein_ligand_binding" else
                        {"molecule": "xyz/sdf/smiles or atoms", "items": "optional molecules"}),
             "parameters": deepcopy({**_DEFAULTS, **_PARAMETERS[identifier]}),
             "example": deepcopy(examples[identifier])}
            for identifier, chinese, english, kinds, description in _ROWS]


def _molecule(value):
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, str):
        raise TypeError("Provide a molecule dictionary, XYZ/SDF path, or SMILES")
    path = Path(value)
    if path.suffix.lower() == ".xyz" and path.is_file():
        lines = path.read_text().splitlines()
        count = int(lines[0])
        if len(lines[2:]) != count:
            raise ValueError("XYZ atom count differs from its header")
        return {"atoms": "; ".join(lines[2:])}
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = (Chem.MolFromMolFile(str(path), removeHs=False) if path.suffix.lower() == ".sdf"
           and path.is_file() else Chem.MolFromSmiles(value))
    if mol is None:
        raise ValueError("Invalid molecule input")
    mol = Chem.AddHs(mol, addCoords=True)
    if not mol.GetNumConformers() and AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise ValueError("Could not embed the molecule")
    positions = mol.GetConformer().GetPositions()
    return {"atoms": "; ".join(atom.GetSymbol() + " " + " ".join(map(str, positions[index]))
                               for index, atom in enumerate(mol.GetAtoms())),
            "charge": Chem.GetFormalCharge(mol),
            "spin": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())}


def draft(app_id, inputs=None, parameters=None, example=False):
    """Build TaskSpec dictionaries only; no signing or execution authority is present."""
    entry = next((entry for entry in catalog() if entry["id"] == app_id), None)
    if entry is None:
        raise ValueError(f"Unknown application: {app_id}")
    if inputs is not None and not isinstance(inputs, dict):
        raise TypeError("Application inputs must be an object")
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError("Application parameters must be an object")
    supplied = deepcopy(inputs or {})
    data = {**(entry["example"] if example else {}), **supplied}
    if "smiles" in supplied and "molecule" not in supplied:
        data.pop("molecule", None)
    if not data:
        raise ValueError("Supply inputs or select example=True")
    optional = ({"functional", "dispersion", "method"} if app_id in
                ("frontier_orbitals", "solvation_scan", "pes_scan", "metabolic_sites") else set())
    if app_id == "frontier_orbitals":
        optional.add("final_spin")
    if app_id == "aqueous_solubility":
        optional.update(("melting_point_C", "model"))
    unknown = set(parameters or {}) - set(entry["parameters"]) - optional
    if unknown:
        raise ValueError(f"Unknown application parameters: {sorted(unknown)}")
    params = {**entry["parameters"], **(parameters or {})}
    for key in ("error", "wall_s", "qpu_shots", "qpu_s"):
        if key in params and (type(params[key]) not in (int, float) or
                              not math.isfinite(params[key]) or params[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("qpu", "physical_gap", "rate_constant", "counterpoise", "minimize"):
        if key in params and type(params[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    model = data.get("electronic_model", {"basis": params["basis"]})
    level = {"method": params.get("method", "hf"),
             "functional": params.get("functional"), "dispersion": params.get("dispersion")}
    def common():
        return {"molecule": _molecule(data.get("molecule", data.get("smiles"))),
                "electronic_model": model}
    targets = []
    validity = []
    unit = "hartree"
    relation = "approximate"
    decision = entry["description"] + " Human approval of gates and budget is required."
    budget = {"wall_s": params["wall_s"], "sim_shots": 100000, "qpu_s": 0, "qpu_shots": 0}
    effects = ["pure", "simulator"]
    if app_id == "ground_state_energy":
        targets = [{"kind": "molecular_ground_energy", **common(),
                    "active": data.get("active", "auto"), "sector": data.get("sector", {})}]
        relation = "variational_upper_bound"
        if params["qpu"]:
            budget.update(qpu_s=params["qpu_s"], qpu_shots=params["qpu_shots"])
            effects.append("qpu")
            decision += " Request vqe_sampled candidates deployable on MQ02; enforce R130."
    elif app_id in ("stability_ranking", "conformer_ranking"):
        if app_id == "conformer_ranking":
            from chemflow.domains.reactivity import conformer_search

            found = conformer_search(None, data["smiles"], {"method": params["conformer_method"]},
                                     count=params["count"], rmsd=params["rmsd"])
            from rdkit import Chem

            charge = Chem.GetFormalCharge(Chem.MolFromSmiles(data["smiles"]))
            items = [{"name": f"conformer-{index}", "molecule": {"atoms": item["atoms"],
                      "charge": charge}} for index, item in enumerate(found["conformers"])]
            decision += " Seed 42; deduplicated conformers can yield a singleton ranking."
        else:
            items = [{"name": item["name"], "molecule": _molecule(item["molecule"])}
                     for item in data["items"]]
        targets = [{"kind": "ranking", "items": items, "electronic_model": model}]
    elif app_id == "pes_scan":
        coordinate = data["coordinate"]
        targets = [{"kind": "pes_scan", **common(), "coordinate": coordinate,
                    "mode": params["mode"],
                    "relax_level": level if params["mode"] == "relaxed" else None}]
        low, high = min(coordinate["values"]), max(coordinate["values"])
        if low == high:
            low, high = low - 0.05, high + 0.05
        validity = [{"axis": "internal_coordinate", "range": [float(low), float(high)], "points": 3,
                     "resolution": (high - low) / 4}]
    elif app_id == "metabolic_sites":
        targets = [{"kind": "metabolic_sites", **common(), "level": level}]
        unit = "dimensionless"
    elif app_id == "bond_dissociation":
        targets = [{"kind": "bond_dissociation", **common(), "bonds": params["bonds"]}]
    elif app_id == "solvation_scan":
        for solvent in params["solvents"]:
            solvent = ({"model": "alpb", "name": solvent} if isinstance(solvent, str)
                       else solvent)
            solvent_level = ({"method": "gfn2"} if solvent["model"] in ("alpb", "gbsa")
                             else level)
            targets.append({"kind": "solvation_free_energy", **common(),
                            "level": solvent_level, "solvent": solvent})
        relation = "equivalent"
    elif app_id == "frontier_orbitals":
        targets = [{"kind": "frontier_orbitals", **common(), "level": level,
                    "quantity": quantity} for quantity in ("homo", "lumo", "gap")]
        if params["physical_gap"]:
            targets.extend({"kind": kind, **common(), "final_spin": params.get("final_spin", 0.5)}
                           for kind in ("ionization_energy", "electron_affinity"))
    elif app_id == "aqueous_solubility":
        melting = params.get("melting_point_C")
        targets = [{"kind": "aqueous_solubility", "molecule_smiles": data["smiles"],
                    "model": params.get("model", "gse" if melting is not None else "esol"),
                    "melting_point_C": melting}]
        unit, relation = "log10(mol/L)", "equivalent"
    elif app_id == "reaction_barrier":
        targets = [{"kind": "reaction_barrier", "electronic_model": model,
                    "species": data["species"], "product": _molecule(data["product"]),
                    "rate_constant": params["rate_constant"]}]
        if params["rate_constant"]:
            targets[0]["thermo"] = {"quantity": "gibbs", "temperature_K": params["temperature_K"]}
    elif app_id == "dimer_binding_scan":
        targets = [{"kind": "interaction_scan", "complex": _molecule(data["molecule"]),
                    "electronic_model": model, "fragments": data["fragments"],
                    "separations": params["separations"], "counterpoise": params["counterpoise"]}]
    elif app_id == "protein_ligand_binding":
        receptor = data["receptor"]
        if isinstance(receptor, str):
            receptor = {"pdb_text": Path(receptor).read_text()}
        elif "example_file" in receptor:
            receptor = {"pdb_text": files(__package__).joinpath(
                "examples/" + receptor["example_file"]).read_text()}
        source = deepcopy(data["ligand"])
        if "residue" in source:
            source.setdefault("pdb_text", receptor["pdb_text"])
        targets = [{"kind": "binding_free_energy", "receptor": receptor,
                    "ligand": {"source": source}, **{key: params[key] for key in
                    ("frames", "minimize", "ph", "seed", "forcefield")}}]
        unit, relation = "kcal/mol", "equivalent"
    elif app_id == "lattice_superconductivity":
        shared = {"lattice": data["lattice"], "sector": data["sector"]}
        targets = [{"kind": "pairing_correlation", **shared, "quantity": quantity,
                    "site_pair": params["site_pair"]} for quantity in ("pair_s_wave", "spin_spin")]
        targets.append({"kind": "lattice_ground_energy", **shared})
    drafts = []
    if not targets:
        raise ValueError("Application requires at least one target (check the solvent list)")
    for target in targets:
        target_unit = "dimensionless" if target["kind"] == "pairing_correlation" else unit
        spec = TaskSpec.model_validate({"question": entry["name_en"], "target": target,
            "gates": [{"kind": "abs_error_vs_exact", "threshold": params["error"],
                       "unit": target_unit}, {"kind": "relation_at_least", "relation": relation}],
            "budget": budget, "allowed_effects": effects, "decision": decision, "validity": validity})
        drafts.append(spec.model_dump(mode="json"))
    return drafts
