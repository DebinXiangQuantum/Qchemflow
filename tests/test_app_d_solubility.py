import csv
from pathlib import Path

import numpy as np
import pytest

from chemflow import build
from chemflow.agent.critic import review
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.check.static import check
from chemflow.domains import load_registry, pack_names
from chemflow.domains.solubility import base_plan, identity
from chemflow.empirical import descriptors, evaluate, model_card, molecule
from chemflow.eval.runner import EvaluationTask, Expected, score_task
from chemflow.runtime import Executor
from chemflow.verify.kernel import check_certificate


def spec(model="esol", smiles="CCO", melting=None):
    return TaskSpec.model_validate({"question": "Estimate logS", "target": {
        "kind": "aqueous_solubility", "model": model, "molecule_smiles": smiles,
        "melting_point_C": melting}, "gates": [
            {"kind": "abs_error_vs_exact", "threshold": 1e-6, "unit": "dimensionless"},
            {"kind": "relation_at_least", "relation": "equivalent"}],
        "decision": "Report estimate and applicability"})


def execute(operator, **parameters):
    registry = load_registry(pack_names())
    with build.program("empirical-test") as builder:
        node = builder.node(operator, **parameters)
    program = builder.to_ir()
    return registry, program, node.id, Executor(registry).run(program)


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


@pytest.mark.parametrize("parameters", [
    {"molecule_smiles": ""}, {"molecule_smiles": "not smiles"},
    {"molecule_smiles": "[H][H]"}, {"molecule_smiles": True},
    {"molecule_smiles": "CCO", "melting_point_C": True},
    {"molecule_smiles": "CCO", "melting_point_C": None},
    {"molecule_smiles": "CCO", "melting_point_C": float("inf")},
])
def test_invalid_operator_parameters_are_static_errors(parameters):
    with build.program("invalid") as builder:
        builder.node("logs_gse", **parameters)
    assert check(builder.to_ir(), load_registry(pack_names()))


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


def test_certificates_reject_tampered_model_value():
    registry, program, identifier, computed = execute("logs_esol", molecule_smiles="CCO")
    certificate = computed["relation_certificates"][identifier]
    assert certificate["relation"] == "equivalent"
    assert check_certificate(certificate, program, registry=registry, computed=computed).valid
    computed[identifier]["value"] += 1
    assert not check_certificate(certificate, program, registry=registry, computed=computed).valid


def test_runtime_empirical_evidence_roundtrip_and_tampering():
    from chemflow.runtime.evidence import decode_empirical_estimate

    _, _, identifier, computed = execute("empirical_estimate", molecule_smiles="CCO", model="esol")
    result = computed[identifier]
    assert result.evidence[0].status == "empirical_calibration"
    payload = result.model_dump(mode="json", fallback=dict)
    decoded = decode_empirical_estimate(payload, computed["_artifacts"])
    assert decoded.value == result.value
    with pytest.raises(ValueError):
        decode_empirical_estimate(payload, None)
    payload["value"]["value"] += 1
    with pytest.raises(ValueError):
        decode_empirical_estimate(payload, computed["_artifacts"])


def test_refinements_and_distinct_models_in_one_program():
    registry = load_registry(pack_names())
    with build.program("multiple-models") as builder:
        for smiles in ("CCO", "c1ccccc1"):
            node = builder.node("empirical_estimate", molecule_smiles=smiles, model="esol")
            builder.refine(node, "empirical_estimate")
    program = builder.to_ir()
    assert not check(program, registry)
    computed = Executor(registry).run(program)
    assert all(computed[node.id].diagnostics["relation_status"] == "proved"
               for node in program.nodes)


def test_target_identity_and_discoverability():
    registry = load_registry(pack_names())
    record = TaskStore("test").freeze(spec(), "reviewer")
    plan, controls = base_plan(record)
    assert controls and plan.references
    assert not identity(plan, record, {})
    plan.nodes[0].params["model"] = "gse"
    assert identity(plan, record, {})
    assert registry.search("aqueous solubility")
    from chemflow.cards import validate_cards

    cards = validate_cards(registry, Path(__file__).parents[1]).cards
    assert {"logs_esol", "logs_gse"} <= {card.id for card in cards}
    for name in ("logs_esol", "logs_gse", "empirical_estimate"):
        implementation = registry.implementations[name]
        assert implementation.execution_role == "classical"
        resources = implementation.resource_model()
        assert all(resources[key] == 0 for key in ("gpu_s", "sim_shots", "qpu_s", "qpu_shots"))


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


@pytest.mark.parametrize("model,smiles,melting", [
    ("gse", "c1ccc2ccccc2c1", 80.0), ("esol", "[Fe+2]", None),
])
def test_agent_gse_and_outside_domain_report(tmp_path, model, smiles, melting):
    from chemflow.agent.orchestrator import Orchestrator

    store = TaskStore("test")
    orchestrator = Orchestrator(store, lambda task: store.freeze(task, "reviewer"), tmp_path)
    result = orchestrator.run(spec(model, smiles, melting))
    assert result.decision.candidate is not None
    assert result.decision.decision == "stop"
    assert "model_change" in result.report and "RMSE" in result.report
    if smiles == "[Fe+2]":
        assert "outside_domain" in result.report
        assert all(item["validity_status"] == "outside_domain" for item in result.results)
