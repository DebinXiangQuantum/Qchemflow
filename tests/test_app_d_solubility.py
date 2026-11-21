import csv
from pathlib import Path

import numpy as np
import pytest

from chemflow.agent.critic import review
from chemflow.agent.task import TaskSpec
from chemflow.empirical import descriptors, evaluate, model_card, molecule
from chemflow.eval.runner import EvaluationTask, Expected, score_task


def spec(model="esol", smiles="CCO", melting=None):
    return TaskSpec.model_validate({"question": "Estimate logS", "target": {
        "kind": "aqueous_solubility", "model": model, "molecule_smiles": smiles,
        "melting_point_C": melting}, "gates": [
            {"kind": "abs_error_vs_exact", "threshold": 1e-6, "unit": "dimensionless"},
            {"kind": "relation_at_least", "relation": "equivalent"}],
        "decision": "Report estimate and applicability"})


def test_cv_refits_only_training_folds_and_cards_are_defensive():
    path = Path(__file__).parents[1] / "src/chemflow/data/esol/delaney-processed.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    features = np.array([[1, *descriptors(molecule(row["smiles"]))] for row in rows])
    target = np.array([float(row["measured log solubility in mols per litre"]) for row in rows])
    errors = []
    for fold in np.array_split(np.random.default_rng(42).permutation(len(rows)), 5):
        train = np.setdiff1d(np.arange(len(rows)), fold)
        weights = np.linalg.lstsq(features[train], target[train], rcond=None)[0]
        errors.extend((features[fold] @ weights - target[fold]) ** 2)
    card = model_card("esol")
    assert card["cv_rmse"] == pytest.approx(np.sqrt(np.mean(errors)), abs=1e-12)
    card["coefficients"]["intercept"] = 999
    assert model_card("esol")["coefficients"]["intercept"] != 999
    with pytest.raises(ValueError):
        model_card("unknown")


@pytest.mark.parametrize("melting", [-100, 0, 25])
def test_gse_clamps_melting_point(melting):
    result = evaluate("gse", "CCO", melting)
    assert result["value"] == pytest.approx(0.5 - descriptors(molecule("CCO"))[0])
    assert model_card("gse")["cv_rmse"] is None
    assert model_card("gse")["train_rmse"] is None


def test_applicability_all_three_checks():
    for smiles in ("CC(=O)Oc1ccccc1C(=O)O", "[Fe+2]", "C" * 100, "[He]"):
        parsed = molecule(smiles)
        values = descriptors(parsed)
        domain = model_card("esol")["applicability"]
        result = evaluate("esol", smiles)
        inside = (all(atom.GetSymbol() in domain["elements"] for atom in parsed.GetAtoms()) and
                  all(lower <= value <= upper for value, (lower, upper) in
                      zip(values, domain["descriptor_ranges"].values(), strict=True)) and
                  result["max_tanimoto"] >= 0.3)
        assert result["validity_status"] == ("inside_domain" if inside else "outside_domain")


def test_outside_domain_never_escalates():
    candidate = {"id": "estimate", "relation": "equivalent", "relation_status": "proved",
                 "gate_passed": True, "evidence": True, "validity_status": "outside_domain"}
    assert review([candidate], []).decision != "escalate"
    assert review([{**candidate, "validity_status": "inside_domain"}], []).decision == "escalate"


def test_expected_property_scoring_uses_signed_bar():
    task = EvaluationTask(id="solubility", category="property", spec=spec(),
                          expected=Expected(value=0, unit="dimensionless"),
                          reference_source="independent test")
    for value, correct in ((0, True), (1e-6, True), (1.00001e-6, False)):
        summary = {"candidate": "answer", "results": [{"node": "answer",
                   "value": f"{value} log10(mol/L)", "relation": "equivalent"}]}
        assert score_task(task, summary, [])["correct"] is correct
    with pytest.raises(ValueError):
        Expected(value=1)
    with pytest.raises(ValueError):
        Expected(value=1, unit="dimensionless", value_hartree=1)
