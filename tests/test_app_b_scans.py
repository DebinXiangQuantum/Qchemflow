from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemflow.agent.families import default_axes, instantiate
from chemflow.agent.task import InteractionScanTarget, PESScanTarget, TaskSpec, ValidityAxis
from chemflow.cards import validate_cards
from chemflow.check import check
from chemflow.domains import load_registry
from chemflow.domains.scans import base_plan, identity
from chemflow.eval.runner import EvaluationTask, score_task
from chemflow.runtime import Executor
from chemflow.types import EnergyCurve
from chemflow.types.curves import curve_error
from chemflow.verify import check_certificate, prove


def target():
    return PESScanTarget(kind="pes_scan", molecule={"atoms": "H 0 0 0; H 0 0 .74"},
                         electronic_model={"basis": "sto-3g"},
                         coordinate={"type": "bond", "atoms": [0, 1], "values": [.7, 1.2]})


def record(scan=None):
    return SimpleNamespace(spec=TaskSpec(question="Scan", target=scan or target(),
        gates=[{"kind": "abs_error_vs_exact", "threshold": .0016, "unit": "hartree"},
               {"kind": "relation_at_least", "relation": "approximate"}],
        budget={"wall_s": 1800}, allowed_effects=["pure"], decision="Choose a control"))


def test_curve_certificate_and_identity_tampering():
    registry = load_registry(["paths"])
    plan, controls = base_plan(record())
    assert not check(plan, registry)
    assert not identity(plan, record(), {})
    computed = Executor(registry).run(plan)
    reference = plan.references[0]
    curve = computed[reference]
    assert isinstance(curve, dict) and isinstance(curve, EnergyCurve)
    assert curve.relation == "equivalent"
    assert curve.diagnostics["relation_status"] == "proved"
    assert curve_error(computed[controls[2]], curve) < 1e-8
    certificate = prove(plan, reference, registry=registry, computed=computed)
    assert check_certificate(certificate, plan, registry=registry, computed=computed).valid
    altered = {**computed, reference: curve.model_copy(update={"values_hartree": [0., 0.]})}
    assert not check_certificate(certificate, plan, registry=registry, computed=altered).valid
    point_id = next(node for node in plan.nodes if node.id == reference).inputs["point0"]
    point = computed[point_id]
    altered[reference] = curve
    altered[point_id] = point.model_copy(update={"diagnostics": {
        **point.diagnostics, "molecule_witness": {"atoms": "H 0 0 0; H 0 0 .8"}}})
    assert not check_certificate(certificate, plan, registry=registry, computed=altered).valid
    next(node for node in plan.nodes if node.op == "geometry_transform").params["value"] = .8
    assert identity(plan, record(), {})


def test_validity_axes_and_instantiation():
    scan = target()
    assert default_axes(scan)[0].axis == "internal_coordinate"
    plan, _ = base_plan(record(scan))
    varied = instantiate(plan, "internal_coordinate", .9)
    assert not check(varied, load_registry(["paths"]))
    assert [node.params["value"] for node in varied.nodes
            if node.op == "geometry_transform"] == pytest.approx([.9, 1.4])
    dimer = InteractionScanTarget(kind="interaction_scan",
        complex={"atoms": "H 0 0 0; H 0 0 .74; H 3 0 0; H 3 0 .74"},
        electronic_model={"basis": "sto-3g"}, fragments=[[0, 1], [2, 3]], separations=[3., 4.])
    assert default_axes(dimer)[0].axis == "separation"
    plan, _ = base_plan(record(dimer))
    varied = instantiate(plan, "separation", 3.5)
    assert [node.params["value"] for node in varied.nodes
            if node.op == "geometry_transform"] == pytest.approx([3.5, 4.5])
    for axis in ("internal_coordinate", "separation"):
        assert ValidityAxis(axis=axis, range=(.5, 2.), points=3, resolution=.1)
    with pytest.raises(ValueError):
        TaskSpec.model_validate({**record().spec.model_dump(), "validity": [
            {"axis": "separation", "range": [.5, 2.], "points": 3, "resolution": .1}]})


def test_relaxed_scan_freezes_shared_geometry_protocol():
    scan = PESScanTarget(kind="pes_scan",
        molecule={"atoms": "O 0 0 0; H 0 .757 .587; H 0 -.757 .587"},
        electronic_model={"basis": "sto-3g"},
        coordinate={"type": "bond", "atoms": [0, 1], "values": [1.1]},
        mode="relaxed", relax_level={"method": "hf"})
    plan, _ = base_plan(record(scan))
    geometry = [node for node in plan.nodes if node.op == "constrained_optimize"]
    assert len(geometry) == 1
    assert {node.inputs["molecule"] for node in plan.nodes if node.op == "component_energy"} == {
        geometry[0].id}
    plan.targets = plan.references[:]
    results = Executor(load_registry(["paths"])).run(plan)
    assert results[plan.references[0]].relation == "equivalent"


def test_curve_scoring_requires_every_point_and_matching_length():
    task = EvaluationTask(id="curve", category="path", spec=record().spec,
        expected={"values_hartree": [-1., -2.]}, reference_source="Independent fixture")

    def scored(values):
        return score_task(task, {"candidate": "answer", "decision": "stop", "results": [
            {"node": "answer", "values_hartree": values, "relation": "approximate",
             "gate_verdict": "PASS"}]}, [])

    assert scored([-1., -2.])["correct"]
    assert not scored([-1., -1.9])["correct"]
    assert not scored([-1.])["correct"]
    assert not scored([-1., float("nan")])["completed"]
    with pytest.raises(ValidationError):
        EvaluationTask(id="curve", category="path", spec=record().spec,
            expected={"values_hartree": [float("inf")]}, reference_source="Fixture")


def test_pack_cards_validate_and_targets_forbid_extra_fields():
    report = validate_cards(load_registry(["paths"]), Path(__file__).resolve().parents[1])
    assert report.ok
    assert {card.id for card in report.cards} >= {"paths_pes", "paths_interaction", "paths_irc"}
    with pytest.raises(ValidationError):
        PESScanTarget.model_validate({**target().model_dump(), "typo": 1})
    with pytest.raises(ValidationError):
        PESScanTarget.model_validate({**target().model_dump(), "mode": "relaxed"})
