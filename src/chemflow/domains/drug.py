"""RDKit screening and PySCF counterpoise interaction energies, not binding free energies."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from pydantic import BaseModel

from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash

from ._adapters import estimate


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
        records = json.loads(path.read_text(encoding="utf-8"))
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
    text = receptor_pdbqt if "\n" in receptor_pdbqt else Path(receptor_pdbqt).read_text(encoding="utf-8")
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
