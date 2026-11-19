import json

import pytest
from scipy.stats import norm

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.design import DesignStore, instantiate_design, register_design, verify_design
from chemflow.diagnostics import AdapterError
from chemflow.domains import get_pack, load_registry
from chemflow.ir import AcceptanceGate
from chemflow.relations import TargetRelation
from chemflow.runtime import Executor
from chemflow.runtime.task import freeze_task
from chemflow.types import Uncertainty
from chemflow.units import Q

MODEL = {"U": 2.0, "eps_bath": [-1.0], "t_bath": [0.5], "unit": "hartree"}
ENERGY = {"kind": "ground_energy", "model": {"impurity_model": MODEL},
          "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}}
SPECTRAL = {**ENERGY, "kind": "spectral_function", "orbital": 0, "spin": "alpha",
            "omega": {"start": -4.0, "stop": 4.0, "n": 101, "unit": "hartree"}, "eta": 0.1}


def plan(target=ENERGY):
    with build.program("accuracy") as builder:
        model = builder.node("impurity_model", **MODEL)
        sector = builder.sector(**target["sector"])
        if target["kind"] == "spectral_function":
            result = builder.node("spectral_function", hamiltonian=model, sector=sector,
                                  **{key: target[key] for key in ("orbital", "spin", "omega", "eta")})
            builder.refine(result, "exact_lehmann")
            builder.bind(result, "spectral_exact")
        else:
            result = builder.ground_energy(model, sector)
            builder.refine(result, "fci")
            builder.bind(result, "sector_diag_numpy")
    return builder.to_ir(), result.id


def verify(program, node, *, target=ENERGY, registry=None, store=None, task=None):
    return verify_design(program, node, target=target,
                         registry=registry or load_registry(["molecular", "algorithms", "dynamics"]),
                         store=store or DesignStore(), instances=1, seed=11, task=task)


def perturb_energy(monkeypatch, *, offset=0.0, stat_err=None,
                   relation=TargetRelation.approximate):
    run = Executor.run

    def perturbed(executor, program, *args, **kwargs):
        results = run(executor, program, *args, **kwargs)
        for node in program.nodes:
            if node.op == "ground_energy":
                estimate = results[node.id]
                results[node.id] = estimate.model_copy(update={
                    "value": Q(estimate.value.to("hartree").value + offset, "hartree").to("eV"),
                    "uncertainty": Uncertainty(stat_err=None if stat_err is None else
                                               Q(stat_err, "hartree").to("eV").value),
                    "relation": relation})
        return results

    monkeypatch.setattr(Executor, "run", perturbed)


def forbid_adapters(monkeypatch, registry):
    def forbidden(*args, **kwargs):
        pytest.fail("No adapter may execute before design accuracy preflight")

    for implementation in registry.implementations.values():
        monkeypatch.setattr(implementation, "adapter", forbidden)


def test_energy_bar_converts_units_and_template_ignores_gates(monkeypatch):
    perturb_energy(monkeypatch, offset=0.001595)
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(0.0016, "hartree"))]
    loose = verify(program, node)
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(1.0, "kcal/mol"))]
    strict = verify(program, node)
    assert loose.passed and not strict.passed
    assert loose.instances[0]["metric"] == strict.instances[0]["metric"]
    assert loose.program_hash == strict.program_hash
    assert loose.template == strict.template and loose.template["gates"] == []
    assert loose.id != strict.id
    assert strict.accuracy == {"kind": "abs_error_vs_exact", "threshold": 1.0,
                               "unit": "kcal/mol", "source": "program", "confidence": 0.95}
    assert "tolerance" not in strict.model_dump()


@pytest.mark.parametrize("confidence,passed", [(0.5, True), (0.95, False)])
def test_statistical_margin_uses_confidence_and_estimate_units(monkeypatch, confidence, passed):
    perturb_energy(monkeypatch, offset=0.0002, stat_err=0.0005,
                   relation=TargetRelation.stochastic_estimate)
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(0.001, "hartree"),
                                    confidence=confidence)]
    report = verify(program, node)
    assert report.passed is passed
    assert report.accuracy["confidence"] == confidence
    assert report.instances[0]["metric"] < 0.001
    assert report.instances[0]["statistical_margin"] == pytest.approx(
        norm.ppf((1 + confidence) / 2) * 0.0005)


def test_stochastic_estimate_without_error_cannot_pass(monkeypatch):
    perturb_energy(monkeypatch, relation=TargetRelation.stochastic_estimate)
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(1.0, "hartree"))]
    assert not verify(program, node).passed


@pytest.mark.parametrize("required,passed", [("approximate", True), ("equivalent", False)])
def test_gate_relation_is_enforced(monkeypatch, required, passed):
    perturb_energy(monkeypatch)
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(1.0, "hartree"), relation=required)]
    report = verify(program, node)
    assert report.passed is passed
    assert report.accuracy["relation"] == required


@pytest.mark.parametrize("measure,unit", [("abs_error_le", "hartree"),
                                         ("l1_error_le", "dimensionless")])
@pytest.mark.parametrize("threshold", [-1.0, float("inf"), float("nan"), [0.1]])
def test_invalid_thresholds_are_refused(measure, unit, threshold):
    target = ENERGY if measure == "abs_error_le" else SPECTRAL
    program, node = plan(target)
    program.gates = [AcceptanceGate(node=node, **{measure: Q(threshold, unit)})]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        verify(program, node, target=target)


@pytest.mark.parametrize("measure,unit", [("abs_error_le", "dimensionless"),
                                         ("l1_error_le", "hartree")])
def test_wrong_threshold_dimensions_are_refused(measure, unit):
    target = ENERGY if measure == "abs_error_le" else SPECTRAL
    program, node = plan(target)
    program.gates = [AcceptanceGate(node=node, **{measure: Q(1.0, unit)})]
    with pytest.raises(ValueError, match="Incompatible units"):
        verify(program, node, target=target)


@pytest.mark.parametrize("other_node", [False, True])
def test_multiple_error_gates_or_gates_on_other_nodes_are_refused(other_node):
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(0.0016, "hartree")),
                    AcceptanceGate(node=program.nodes[0].id if other_node else node,
                                   abs_error_le=Q(1.0, "kcal/mol"))]
    with pytest.raises(ValueError, match="target node|exactly one"):
        verify(program, node)


@pytest.fixture
def registered(tmp_path):
    registry = load_registry(["molecular", "algorithms", "dynamics"])
    store = DesignStore(tmp_path / "designs.sqlite")
    program, node = plan()
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(0.0016, "hartree"))]
    report = verify(program, node, registry=registry, store=store)
    assert report.passed
    register_design(registry, store, report.id, "energy")
    return registry, store, report


def test_accuracy_persists_across_reopen_and_is_detached(registered, tmp_path):
    registry, store, report = registered
    store._connection.close()
    reopened = DesignStore(tmp_path / "designs.sqlite")
    assert reopened.get(report.id).accuracy == report.accuracy
    assert reopened.list()[0]["accuracy"] == report.accuracy
    reopened.list()[0]["accuracy"]["threshold"] = 99
    assert reopened._design("energy")["accuracy"] == report.accuracy
    program, node = instantiate_design(reopened, "energy", model_params=MODEL)
    estimate = Executor(registry).run(program)[node]
    assert estimate.diagnostics["design_accuracy"] == report.accuracy
    assert set(next(item for item in program.nodes if item.id == node).metadata) == {
        "design", "design_status", "design_evidence"}


@pytest.mark.parametrize("threshold,allowed", [(Q(0.0016, "hartree"), True),
                                              (Q(1.1, "kcal/mol"), True),
                                              (Q(1.0, "kcal/mol"), False)])
def test_reuse_compares_energy_bars_in_same_units(registered, monkeypatch, threshold, allowed):
    registry, store, _ = registered
    program, node = instantiate_design(store, "energy", model_params=MODEL)
    program.gates = [AcceptanceGate(node=node, abs_error_le=threshold)]
    if allowed:
        Executor(registry).run(program, freeze_task(program))  # gates need a frozen task
    else:
        forbid_adapters(monkeypatch, registry)
        with pytest.raises(AdapterError, match="CF-E120") as error:
            Executor(registry).run(program)
        assert "0.0016" in str(error.value) and "kcal/mol" in str(error.value)
        assert "register" in error.value.diagnostics[0].fix_hints[0]


@pytest.mark.parametrize("legacy,measure,unit", [(True, "abs_error_le", "hartree"),
    (True, "l1_error_le", "dimensionless"), (False, "l1_error_le", "dimensionless")])
def test_legacy_or_different_measure_is_refused_before_execution(
        registered, monkeypatch, legacy, measure, unit):
    registry, store, _ = registered
    if legacy:
        entry = registry.designs["design:energy"]
        entry.pop("accuracy")
        with store._connection:
            store._connection.execute("UPDATE designs SET payload=? WHERE name=?",
                                      (json.dumps(entry), "energy"))
        assert store.list()[0]["accuracy"] is None
    program, node = instantiate_design(store, "energy", model_params=MODEL)
    if legacy:
        assert Executor(registry).run(program)[node].diagnostics["design_accuracy"] is None
    program.gates = [AcceptanceGate(node=node, **{measure: Q(100, unit)})]
    forbid_adapters(monkeypatch, registry)
    with pytest.raises(AdapterError, match="CF-E120"):
        Executor(registry).run(program)


def test_preflight_checks_every_error_gate(registered, monkeypatch):
    registry, store, _ = registered
    program, node = instantiate_design(store, "energy", model_params=MODEL)
    program.gates = [AcceptanceGate(node=node, abs_error_le=Q(0.002, "hartree")),
                    AcceptanceGate(node=node, abs_error_le=Q(1.0, "kcal/mol"))]
    forbid_adapters(monkeypatch, registry)
    with pytest.raises(AdapterError, match="CF-E120"):
        Executor(registry).run(program)


def frozen(threshold):
    store = TaskStore("accuracy-test")
    spec = TaskSpec.model_validate({"question": "impurity spectrum", "target": SPECTRAL,
        "gates": [{"kind": "l1_error_vs_exact", "threshold": threshold,
                   "unit": "dimensionless"}], "decision": "verify"})
    return store.freeze(spec, "human"), store


def test_task_bar_records_provenance_and_forbids_program_gates():
    record, _ = frozen(0.1)
    program, node = plan(SPECTRAL)
    report = verify(program, node, target=SPECTRAL, task=record)
    assert report.passed
    assert report.accuracy == {"kind": "l1_error_vs_exact", "threshold": 0.1,
                               "unit": "dimensionless", "source": "task",
                               "task_hash": record.content_hash}
    next(item for item in program.nodes if item.id == node).params["orbital"] = 1
    mismatch = verify(program, node, target=SPECTRAL, task=record)
    assert not mismatch.passed and mismatch.accuracy == report.accuracy
    program.gates = [AcceptanceGate(node=node, l1_error_le=Q(0.1, "dimensionless"))]
    with pytest.raises(ValueError, match="must not contain"):
        verify(program, node, target=SPECTRAL, task=record)


def test_stricter_frozen_task_refuses_registered_design_before_any_execution(monkeypatch):
    registry = load_registry(["molecular", "algorithms", "dynamics"])
    store = DesignStore()
    program, node = plan(SPECTRAL)
    program.gates = [AcceptanceGate(node=node, l1_error_le=Q(0.1, "dimensionless"))]
    report = verify(program, node, target=SPECTRAL, registry=registry, store=store)
    register_design(registry, store, report.id, "spectral")
    record, task_store = frozen(0.001)
    instance, node = instantiate_design(store, "spectral", model_params=MODEL)
    base, _ = get_pack("dynamics").base_plan(record)
    base.nodes.extend(instance.nodes)
    base.edges.extend(instance.edges)
    base.targets.append(node)
    forbid_adapters(monkeypatch, registry)
    with pytest.raises(AdapterError, match="CF-E120") as error:
        Executor(registry).run_plan(base, record, task_store)
    assert "0.1" in str(error.value) and "0.001" in str(error.value)
