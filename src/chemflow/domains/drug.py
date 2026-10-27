"""RDKit screening and PySCF counterpoise interaction energies, not binding free energies."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from pydantic import BaseModel

from chemflow import build
from chemflow.diagnostics import AdapterError, Diagnostic
from chemflow.ids import content_hash
from chemflow.relations import combine_linear

from ._adapters import estimate, register_adapter


class LigandLibrary(BaseModel, frozen=True):
    ligands: dict[str, dict[str, str | int | bool]]


def fetch_pubchem(name: str) -> dict[str, str | int]:
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{quote(name, safe='')}/property/SMILES,InChIKey/JSON")
    with urlopen(url, timeout=15) as response:
        record = json.load(response)["PropertyTable"]["Properties"][0]
    return {"CID": record["CID"], "InChIKey": record["InChIKey"],
            "SMILES": next(record[field] for field in
                           ("SMILES", "IsomericSMILES", "CanonicalSMILES") if field in record)}


def _read_name_cache(path):
    try:
        records = json.loads(path.read_text())
        return records if isinstance(records, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_name_cache(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".name-cache-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(records, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _name_structure(name, record):
    from rdkit import Chem

    try:
        if (not isinstance(record, dict) or type(record.get("CID")) is not int or
                record["CID"] <= 0 or not isinstance(record.get("SMILES"), str) or
                not record["SMILES"].strip()):
            raise ValueError("Missing or invalid CID/SMILES")
        molecule = Chem.MolFromSmiles(record["SMILES"])
        if molecule is None:
            raise ValueError("Unparseable SMILES")
        smiles = Chem.MolToSmiles(molecule)
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
        if not inchikey or inchikey != record.get("InChIKey"):
            raise ValueError("InChIKey disagrees with SMILES")
        return smiles, inchikey
    except (ValueError, TypeError, RuntimeError) as exc:
        raise AdapterError("CF-E060", None, f"Invalid PubChem structure for {name}: {exc}",
                           "Check the PubChem CID and structure") from exc


def ligands_from_names(store, names):
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip()
                                          for name in names):
        raise AdapterError("CF-E060", None, "names must be a list of nonempty names",
                           "Provide molecule names")
    path = Path(os.environ.get("CHEMFLOW_NAME_CACHE", "~/.chemflow/name_cache.json")).expanduser()
    records = _read_name_cache(path)
    result, fetched = {}, {}
    for name in dict.fromkeys(names):
        record = records.get(name)
        cached = (isinstance(record, dict) and
                  all(key in record for key in ("CID", "SMILES", "InChIKey", "retrieved_at")))
        if cached:
            try:
                timestamp = datetime.fromisoformat(record["retrieved_at"])
                cached = timestamp.utcoffset() is not None
            except (ValueError, TypeError):
                cached = False
        if not cached:
            try:
                record = fetch_pubchem(name)
            except OSError as exc:
                raise AdapterError("CF-E080", None, f"Cannot resolve molecule {name}: {exc}",
                                   "Restore connectivity or populate the name cache") from exc
            except (ValueError, KeyError, IndexError, TypeError, StopIteration) as exc:
                raise AdapterError("CF-E060", None, f"Invalid PubChem response for {name}",
                                   "Check the PubChem CID and structure") from exc
        smiles, inchikey = _name_structure(name, record)
        if not cached:
            record = {**record, "retrieved_at": datetime.now(UTC).isoformat()}
            fetched[name] = record
        result[name] = {"smiles": smiles, "inchikey": inchikey, "source": "pubchem",
                        "cid": record["CID"], "retrieved_at": record["retrieved_at"],
                        "cached": cached}
    if fetched:
        _write_name_cache(path, {**records, **fetched})
    return LigandLibrary(ligands=result)


def name_resolution_resources(**kwargs):
    return {"wall_s": None, "cpu_core_s": 0, "gpu_s": 0, "sim_shots": 0,
            "qpu_s": 0, "qpu_shots": 0}


class FilterResult(BaseModel, frozen=True):
    passed: list[str]
    flagged: dict[str, list[str]]


def ligand_library(store, ligands):
    from rdkit import Chem

    result = {}
    for name, smiles in ligands.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid ligand SMILES: {name}")
        result[name] = {"smiles": Chem.MolToSmiles(molecule),
                        "inchikey": Chem.MolToInchiKey(molecule)}
    return LigandLibrary(ligands=result)


def descriptors(store, library):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    result = {}
    for name, entry in library.ligands.items():
        molecule = Chem.MolFromSmiles(entry["smiles"])
        result[name] = {"mw": Descriptors.MolWt(molecule),
                        "logp": Descriptors.MolLogP(molecule),
                        "hbd": Lipinski.NumHDonors(molecule),
                        "hba": Lipinski.NumHAcceptors(molecule),
                        "tpsa": Descriptors.TPSA(molecule)}
    return result


def filter_ligands(store, library, rules):
    from rdkit import Chem
    from rdkit.Chem import FilterCatalog

    if not rules or set(rules) - {"lipinski", "pains", "brenk"}:
        raise ValueError("Rules must be lipinski, pains, or brenk")
    catalogs = {}
    for rule in set(rules) - {"lipinski"}:
        parameters = FilterCatalog.FilterCatalogParams()
        parameters.AddCatalog(getattr(parameters.FilterCatalogs, rule.upper()
                                       if rule == "pains" else "BRENK"))
        catalogs[rule] = FilterCatalog.FilterCatalog(parameters)
    properties = descriptors(store, library)
    passed, flagged = [], {}
    for name, entry in library.ligands.items():
        reasons = []
        if "lipinski" in rules:
            reasons.extend(f"lipinski:{key}>{limit}" for key, limit in
                           (("mw", 500), ("logp", 5), ("hbd", 5), ("hba", 10))
                           if properties[name][key] > limit)
        molecule = Chem.MolFromSmiles(entry["smiles"])
        for rule, catalog in catalogs.items():
            reasons.extend(f"{rule}:{match.GetDescription()}"
                           for match in catalog.GetMatches(molecule))
        if reasons:
            flagged[name] = reasons
        else:
            passed.append(name)
    return FilterResult(passed=passed, flagged=flagged)


def conformers(store, library, seed=0, count=10):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Conformers require positive count and nonnegative integer seed")
    result = {}
    for name, entry in library.ligands.items():
        molecule = Chem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed, parameters.numThreads = seed, 1
        identifiers = list(AllChem.EmbedMultipleConfs(molecule, count, parameters))
        if not identifiers or not AllChem.MMFFHasAllMoleculeParams(molecule):
            raise ValueError(f"ETKDG/MMFF unavailable for {name}")
        energies = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=1, maxIters=1000)
        result[name] = {"seed": seed, "unit": "angstrom", "method": "ETKDGv3+MMFF94",
            "conformers": [{"positions": molecule.GetConformer(identifier).GetPositions().tolist(),
                            "energy_kcal_mol": energy, "converged": status == 0}
                           for identifier, (status, energy) in zip(identifiers, energies)]}
    return result


def component_specs(complex, fragments, electronic_model, counterpoise):
    from pyscf.gto import format_atom

    atoms = format_atom(complex["atoms"], unit=1)
    flattened = [index for fragment in fragments for index in fragment]
    if (not fragments or any(not fragment for fragment in fragments) or
            any(type(index) is not int for index in flattened) or
            sorted(flattened) != list(range(len(atoms)))):
        raise ValueError("Fragments must partition the complex atom indices")
    if complex.get("charge", 0) or complex.get("spin", 0):
        raise ValueError("Interaction adapter currently requires neutral closed-shell fragments")
    if set(electronic_model) - {"basis", "hamiltonian_class"} or electronic_model.get(
            "hamiltonian_class", "nonrel_bo") != "nonrel_bo":
        raise ValueError("Interaction adapter supports the all-electron nonrelativistic model")
    components = [complex]
    for fragment in fragments:
        fragment_atoms = [(symbol if index in fragment else "ghost-" + symbol, position)
                          for index, (symbol, position) in enumerate(atoms)
                          if counterpoise or index in fragment]
        text = "; ".join(symbol + " " + " ".join(str(value) for value in position)
                         for symbol, position in fragment_atoms)
        components.append({"atoms": text, "unit": complex.get("unit", "angstrom")})
    return components


def component_energy(store, molecule, electronic_model, method):
    from pyscf import ao2mo, cc, fci, gto, mp, scf

    from chemflow.types import ConstantLedger, Hamiltonian, Sector
    from chemflow.units import Q

    if molecule.get("charge", 0) or molecule.get("spin", 0):
        raise ValueError("Interaction components must be neutral and closed-shell")
    if set(electronic_model) - {"basis", "hamiltonian_class"} or electronic_model.get(
            "hamiltonian_class", "nonrel_bo") != "nonrel_bo":
        raise ValueError("Interaction adapter supports the all-electron nonrelativistic model")
    mol = gto.M(atom=molecule["atoms"], unit=molecule.get("unit", "angstrom"),
                basis=electronic_model["basis"], verbose=0)
    reference = scf.RHF(mol).run(conv_tol=1e-12)
    if not reference.converged:
        raise ValueError("Interaction reference SCF failed to converge")
    if method == "hf":
        energy, relation = reference.e_tot, "variational_upper_bound"
    elif method == "fci":
        calculation = fci.FCI(reference)
        calculation.conv_tol = 1e-12
        energy = calculation.kernel()[0]
        if not calculation.converged:
            raise ValueError("Interaction FCI failed to converge")
        relation = "equivalent"
    elif method == "mp2":
        energy, relation = mp.MP2(reference).run().e_tot, "approximate"
    elif method in ("ccsd", "ccsd_t"):
        calculation = cc.CCSD(reference).run(conv_tol=1e-11)
        if not calculation.converged:
            raise ValueError("Interaction CCSD failed to converge")
        energy = calculation.e_tot + (calculation.ccsd_t() if method == "ccsd_t" else 0)
        relation = "approximate"
    else:
        raise ValueError("Unsupported interaction method")
    orbitals = reference.mo_coeff
    hamiltonian = Hamiltonian(
        h1_ref=store.put(orbitals.T @ reference.get_hcore() @ orbitals),
        h2_ref=store.put(ao2mo.restore(1, ao2mo.kernel(mol, orbitals), orbitals.shape[1])),
        n_orb=orbitals.shape[1], space=store.put(orbitals),
        ledger=ConstantLedger().add("nuclear_repulsion", Q(float(mol.energy_nuc()), "hartree")))
    sector = Sector(n_alpha=mol.nelec[0], n_beta=mol.nelec[1], n_orb=orbitals.shape[1])
    return estimate(float(energy), "total_electronic_energy", relation,
        {"hamiltonian_id": content_hash((molecule, electronic_model))}, "pyscf",
        diagnostics={"hamiltonian_witness": hamiltonian.model_dump(mode="json"),
                     "molecule_witness": molecule,
                     "electronic_model_witness": electronic_model,
                     "sector_witness": sector.model_dump(mode="json")})


def component_matches(value, molecule, electronic_model):
    from chemflow.types import ElectronicModel
    from chemflow.types.model import model_protocol_id

    context = value.context.keys
    if context.get("hamiltonian_id") == content_hash((molecule, electronic_model)):
        return True
    geometry = {key: molecule.get(key, default) for key, default in
                (("atoms", None), ("unit", "angstrom"), ("charge", 0), ("spin", 0))}
    model_id = content_hash(ElectronicModel.model_validate(electronic_model))
    return (context.get("geometry_id") == content_hash(geometry) and
            context.get("model_id") == model_id and
            context.get("model_protocol_id") == model_protocol_id(model_id, None))


def interaction_energy(store, coefficients, target, **components):
    expected = component_specs(**target)
    ports = [f"component{index}" for index in range(len(expected))]
    wanted = dict(zip(ports, [1, *([-1] * (len(expected) - 1))]))
    if coefficients != wanted or set(components) != set(ports):
        raise ValueError("Interaction requires +1 complex and -1 per fragment")
    for port, molecule in zip(ports, expected):
        value = components[port]
        if (value.kind != "total_electronic_energy" or
                not component_matches(value, molecule, target["electronic_model"])):
            raise AdapterError("CF-E002", None, "Interaction component identity mismatch",
                               "Use electronic energies in the declared fragment bases")
    relation = combine_linear([(coefficients[port], value.relation)
                               for port, value in components.items()])
    return estimate(sum(coefficients[port] * value.value.to("hartree").value
                        for port, value in components.items()), "interaction_energy", relation,
                    {"hamiltonian_id": content_hash(target)}, "pyscf:counterpoise",
                    diagnostics={"counterpoise": target["counterpoise"]})


def dock(store, library, receptor_pdbqt, center, box, seed=0):
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina
    except ImportError as exc:
        raise AdapterError("CF-E080", None, "AutoDock Vina/meeko unavailable",
                           "Use an environment with the docking engines installed") from exc
    import tempfile

    from rdkit import Chem
    from rdkit.Chem import AllChem

    if len(library.ligands) != 1:
        raise ValueError("Dock one ligand per node to retain scalar score identity")
    name, ligand = next(iter(library.ligands.items()))
    molecule = Chem.AddHs(Chem.MolFromSmiles(ligand["smiles"]))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise ValueError("Docking ligand embedding failed")
    setups = MoleculePreparation().prepare(molecule)
    pdbqt, ok, message = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(message)
    text = receptor_pdbqt if "\n" in receptor_pdbqt else Path(receptor_pdbqt).read_text()
    with tempfile.TemporaryDirectory() as directory:
        receptor = Path(directory) / "receptor.pdbqt"
        receptor.write_text(text)
        engine = Vina(sf_name="vina", seed=seed, cpu=1, verbosity=0)
        engine.set_receptor(str(receptor))
        engine.set_ligand_from_string(pdbqt)
        engine.compute_vina_maps(center=center, box_size=box)
        engine.dock()
        score = float(engine.energies(n_poses=1)[0][0])
    return estimate(score, "docking_score", "approximate",
        {"hamiltonian_id": content_hash((ligand, text, center, box))}, "autodock-vina",
        diagnostics={"ligand": name}, seed=seed)


def context_check(node, inputs, context, nodes):
    if node.op == "component_energy":
        context.update(kind="total_electronic_energy", unit="hartree",
                       hamiltonian_id=content_hash((node.params.get("molecule", inputs.get("molecule")),
                                                   node.params["electronic_model"])))
    elif node.op == "interaction_energy":
        context.update(kind="interaction_energy", unit="hartree",
                       hamiltonian_id=content_hash(node.params["target"]))
        if any(value.get("kind") != "total_electronic_energy" for value in inputs.values()):
            return [Diagnostic(code="CF-E002", node=node.id,
                               message="Interaction requires electronic energies")]
    elif node.op == "dock":
        context.update(kind="docking_score", unit="kcal/mol")
    return []


def base_plan(record, active=None):
    if active is not None:
        raise ValueError("Interaction reference uses the full molecular orbital space")
    target = record.spec.target.model_dump(exclude={"kind", "operator"})
    species = component_specs(**target)
    with build.program("interaction-controls") as builder:
        controls = []
        for method in ("hf", "mp2", "ccsd", "ccsd_t", "fci"):
            components = {f"component{index}": builder.node("component_energy",
                molecule=molecule, electronic_model=target["electronic_model"], method=method)
                for index, molecule in enumerate(species)}
            result = builder.node("interaction_energy", **components, target=target,
                coefficients={port: 1 if port == "component0" else -1 for port in components})
            if method == "fci":
                builder.ir.references = [result.id]
            else:
                controls.append(result.id)
        builder.ir.targets = [*controls, *builder.ir.references]
        return builder.to_ir(), controls


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    target = record.spec.target.model_dump(exclude={"kind", "operator"})
    species = component_specs(**target)
    nodes = {node.id: node for node in plan.nodes}
    issues = []
    for identifier in set(plan.targets + plan.references):
        node = nodes.get(identifier)
        coefficients = {f"component{index}": 1 if index == 0 else -1
                        for index in range(len(species))}
        valid = (node is not None and node.op == "interaction_energy" and
                 node.params == {"target": target, "coefficients": coefficients} and
                 set(node.inputs) == set(coefficients))
        if valid:
            for index, molecule in enumerate(species):
                component = nodes.get(node.inputs[f"component{index}"])
                valid = valid and (component is not None and component.op == "component_energy"
                    and component.params.get("molecule") == molecule
                    and component.params.get("electronic_model") == target["electronic_model"])
        if not valid:
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                                     message="Interaction target identity mismatch",
                                     fix_hints=plan_hints(plan, base_plan(record)[0], identifier)))
    return issues


def register(registry):
    for name, adapter, inputs, output, parameters, semantics, effect in (
        ("ligands_from_names", ligands_from_names, {}, "ligand_library", ("names",),
         "PubChem names to RDKit-validated structures with cached provenance; check the CID",
         "network"),
        ("ligand_library", ligand_library, {}, "ligand_library", ("ligands",),
         "RDKit sanitized ligand names, canonical isomeric SMILES and InChIKeys", "pure"),
        ("filter_ligands", filter_ligands, {"library": "ligand_library"}, "filter_result", ("rules",),
         "Lipinski cutoffs and RDKit PAINS/Brenk alerts; flags are not efficacy predictions", "pure"),
        ("descriptors", descriptors, {"library": "ligand_library"}, "descriptors", (),
         "RDKit molecular weight, logP, hydrogen donors/acceptors and TPSA", "pure"),
        ("conformers", conformers, {"library": "ligand_library"}, "conformers", ("seed", "count"),
         "Seeded ETKDGv3 conformers relaxed with RDKit MMFF94", "random"),
        ("component_energy", component_energy, {}, "estimate",
         ("molecule", "electronic_model", "method"),
         "PySCF HF/MP2/CCSD/CCSD(T)/FCI component energy including ghost basis functions", "pure"),
        ("interaction_energy", interaction_energy, {}, "estimate", ("coefficients", "target"),
         "Complex minus fragments; counterpoise uses ghost atoms, combine_linear relations", "pure"),
        ("dock", dock, {"library": "ligand_library"}, "estimate",
         ("receptor_pdbqt", "center", "box", "seed"),
         "AutoDock Vina score with meeko ligand preparation; never an electronic energy", "random"),
    ):
        register_adapter(registry, name, adapter, inputs, output, parameters, semantics, effect)
    registry.implementations["ligands_from_names"].cacheable = False
    registry.implementations["ligands_from_names"].resource_model = name_resolution_resources
    registry.context_checks.append(context_check)
