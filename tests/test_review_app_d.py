"""Reviewer tests for APP batch D: empirical models and aqueous solubility (written by Claude;
implementers must not edit).

Application covered: 水溶度快速评估 (aqueous solubility, LogS). Plan:
docs/research/09-application-operators.md, batch D.

Contract:
- **The empirical evidence type in V1.**
  - `chemflow.empirical.model_card(name)` returns `{name, dataset: {path, sha256, n,
    source, licence_note}, descriptors, coefficients, train_rmse, cv_rmse,
    applicability}`.
  - Dataset: ESOL (Delaney 2004), bundled as
    `src/chemflow/data/esol/delaney-processed.csv` (MoleculeNet copy, SHA-256 8c06a76f…).
  - Applicability: training-set elements, descriptor min/max, and a max-Tanimoto threshold
    of 0.3 on Morgan radius-2 2048-bit fingerprints.
  - Inputs outside any of these are `outside_domain`.
- **The ESOL model.** Refitted by ordinary least squares on the bundled dataset's measured
  logS, with RDKit descriptors:
  - clogP = Crippen MolLogP;
  - MW = Descriptors.MolWt;
  - RB = rdMolDescriptors.CalcNumRotatableBonds;
  - AP = aromatic heavy atoms / heavy atoms.

  `cv_rmse` is a seeded 5-fold cross-validation.
- **Operators.**
  - `logs_esol(molecule_smiles)` returns `{value, unit: "log10(mol/L)",
    validity_status, max_tanimoto, model: "esol"}`.
  - `logs_gse(molecule_smiles, melting_point_C)` returns the General Solubility Equation,
    logS = 0.5 − 0.01·(MP − 25) − clogP, with MP clamped below at 25 °C.
- **Target kind `aqueous_solubility`.**
  - Schema: `{kind, molecule_smiles, model: esol|gse, melting_point_C}`. Gates use unit
    "dimensionless". The quantity is model-defined, so a result computed as declared is
    "equivalent", proved.
  - The relation to the *experimental* solubility is model_change, and is stated in the
    report together with the card's RMSE.
  - Results carry `validity_status`, and the critic never escalates outside the domain.
  - The eval harness accepts `expected.value` with `expected.unit` for non-energy targets.
    Category "property".
"""
from __future__ import annotations

import csv
import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "chemflow" / "data" / "esol" / "delaney-processed.csv"
SHA = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"


def _descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    heavy = molecule.GetNumHeavyAtoms()
    aromatic = sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
    return [Crippen.MolLogP(molecule), Descriptors.MolWt(molecule),
            rdMolDescriptors.CalcNumRotatableBonds(molecule), aromatic / heavy]


def _refit():
    rows = list(csv.DictReader(DATA.open()))
    features = np.array([[1.0, *_descriptors(row["smiles"])] for row in rows])
    target = np.array([float(row["measured log solubility in mols per litre"]) for row in rows])
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    rmse = float(np.sqrt(np.mean((features @ coefficients - target) ** 2)))
    return coefficients, rmse, len(rows)


def test_d1_the_esol_card_matches_an_independent_refit() -> None:
    from chemflow.empirical import model_card

    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA
    card = model_card("esol")
    coefficients, rmse, n = _refit()
    assert card["dataset"]["sha256"] == SHA and card["dataset"]["n"] == n == 1128
    assert np.allclose([card["coefficients"][key] for key in
                        ("intercept", "clogp", "mw", "rb", "ap")], coefficients, atol=1e-8)
    assert card["train_rmse"] == pytest.approx(rmse, abs=1e-8)
    assert card["cv_rmse"] >= card["train_rmse"] - 1e-9
    assert card["applicability"]["tanimoto_threshold"] == 0.3


def _get(result, key):
    return result[key] if isinstance(result, dict) else getattr(result, key)


def test_d3_aqueous_solubility_through_the_agent(tmp_path) -> None:
    from chemflow.eval import runner

    coefficients, _, _ = _refit()
    smiles = "CC(=O)Nc1ccc(O)cc1"  # paracetamol
    expected = float(np.dot(coefficients, [1.0, *_descriptors(smiles)]))
    task = {"id": "paracetamol-logs", "category": "property", "spec": {
        "question": "Estimate the aqueous solubility of paracetamol",
        "target": {"kind": "aqueous_solubility", "molecule_smiles": smiles, "model": "esol",
                   "melting_point_C": None},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 1e-6, "unit": "dimensionless"},
                  {"kind": "relation_at_least", "relation": "equivalent"}],
        "budget": {"wall_s": 300}, "allowed_effects": ["pure"],
        "decision": "Report the model-defined estimate with its applicability"},
        "expected": {"value": expected, "unit": "dimensionless"},
        "reference_source": "Reviewer ESOL refit, independent of ChemFlow"}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "app-d", "tasks": [task]}))
    results = runner.run(Namespace(suite=None, suite_file=path, tasks=None, policy="scripted",
                                   model=None, roles="designer", jobs=1, out=tmp_path / "out"))
    row = results["rows"][0]
    assert row["correct"] and row["relation_ok"], row
    report = (tmp_path / "out" / "paracetamol-logs" / "report.md").read_text()
    assert "model_change" in report or "empirical" in report.lower()
    assert "RMSE" in report
