"""Offline, deterministic solubility models and empirical applicability evidence."""

import csv
import hashlib
import math
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Literal, TypedDict

import numpy as np

SOURCE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ATTRIBUTION = ("Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000; via MoleculeNet. "
               "Redistributed research dataset; no separate dataset licence is asserted.")
DESCRIPTORS = ("clogp", "mw", "rb", "ap")
SEED = 42


class EmpiricalResult(TypedDict):
    value: float
    unit: Literal["log10(mol/L)"]
    model: Literal["esol", "gse"]
    validity_status: Literal["inside_domain", "outside_domain"]
    max_tanimoto: float


def molecule(smiles):
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("molecule_smiles must be a nonempty SMILES string")
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or not parsed.GetNumHeavyAtoms():
        raise ValueError("molecule_smiles must describe a molecule with heavy atoms")
    return parsed


def descriptors(parsed):
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return [Crippen.MolLogP(parsed), Descriptors.MolWt(parsed),
            rdMolDescriptors.CalcNumRotatableBonds(parsed),
            sum(atom.GetIsAromatic() for atom in parsed.GetAtoms()) / parsed.GetNumHeavyAtoms()]


@lru_cache(maxsize=1)
def _training():
    from rdkit.Chem import rdFingerprintGenerator

    path = files("chemflow").joinpath("data/esol/delaney-processed.csv")
    rows = list(csv.DictReader(path.read_text().splitlines()))
    molecules = [molecule(row["smiles"]) for row in rows]
    features = np.array([[1.0, *descriptors(parsed)] for parsed in molecules])
    target = np.array([float(row["measured log solubility in mols per litre"]) for row in rows])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = tuple(generator.GetFingerprint(parsed) for parsed in molecules)
    elements = sorted({atom.GetSymbol() for parsed in molecules for atom in parsed.GetAtoms()})
    return path, features, target, fingerprints, elements


@lru_cache(maxsize=2)
def _card(name):
    from rdkit import __version__ as rdkit_version

    if name not in ("esol", "gse"):
        raise ValueError(f"Unknown empirical model: {name}")
    path, features, target, _, elements = _training()
    coefficients = np.linalg.lstsq(features, target, rcond=None)[0]
    residuals = np.empty(len(target))
    for held_out in np.array_split(np.random.default_rng(SEED).permutation(len(target)), 5):
        training = np.ones(len(target), dtype=bool)
        training[held_out] = False
        fitted = np.linalg.lstsq(features[training], target[training], rcond=None)[0]
        residuals[held_out] = features[held_out] @ fitted - target[held_out]
    card = {"name": name, "dataset": {"path": "data/esol/delaney-processed.csv",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(target),
            "source": SOURCE, "licence_note": ATTRIBUTION}, "descriptors": list(DESCRIPTORS),
            "software_versions": {"rdkit": rdkit_version, "numpy": np.__version__},
            "coefficients": dict(zip(("intercept", *DESCRIPTORS), coefficients.tolist())),
            "train_rmse": float(np.sqrt(np.mean((features @ coefficients - target) ** 2))),
            "cv_rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "cross_validation": {"folds": 5, "seed": SEED, "aggregation": "pooled RMSE"},
            "applicability": {"elements": elements,
                "descriptor_ranges": {key: [float(features[:, index].min()),
                                             float(features[:, index].max())]
                                      for index, key in enumerate(DESCRIPTORS, 1)},
                "tanimoto_threshold": 0.3, "morgan_radius": 2, "fingerprint_bits": 2048}}
    if name == "gse":
        card.update(coefficients={"intercept": 0.5, "melting_point_offset": -0.01, "clogp": -1},
                    train_rmse=None, cv_rmse=None, cross_validation=None,
                    calibration_note="GSE is not fitted: Delaney lacks melting points. "
                    "Applicability is a structural ESOL-domain proxy, not GSE validation.",
                    reference="Yalkowsky and Valvani, J. Pharm. Sci. 1980, 69, 912–922")
    return card


def model_card(name):
    return deepcopy(_card(name))


def evaluate(name, molecule_smiles, melting_point_C=None) -> EmpiricalResult:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    card = _card(name)
    parsed = molecule(molecule_smiles)
    values = descriptors(parsed)
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(
        parsed)
    similarity = max(DataStructs.BulkTanimotoSimilarity(fingerprint, _training()[3]))
    domain = card["applicability"]
    inside = (all(atom.GetSymbol() in domain["elements"] for atom in parsed.GetAtoms()) and
              all(domain["descriptor_ranges"][key][0] <= value <=
                  domain["descriptor_ranges"][key][1]
                  for key, value in zip(DESCRIPTORS, values, strict=True)) and similarity >= 0.3)
    if name == "esol":
        value = float(np.dot([card["coefficients"][key] for key in ("intercept", *DESCRIPTORS)],
                             [1.0, *values]))
    else:
        if type(melting_point_C) not in (int, float) or not math.isfinite(melting_point_C):
            raise ValueError("GSE requires a finite melting_point_C")
        value = 0.5 - 0.01 * (max(25.0, melting_point_C) - 25.0) - values[0]
    return {"value": value, "unit": "log10(mol/L)", "model": name,
            "validity_status": "inside_domain" if inside else "outside_domain",
            "max_tanimoto": similarity}
