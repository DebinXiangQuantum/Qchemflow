from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemflow import build
from chemflow.agent.families import default_axes
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import PairingTarget, TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.cards import validate_cards
from chemflow.check import check
from chemflow.check.static import infer_context
from chemflow.domains import load_registry, pack_for_target, pack_names
from chemflow.domains.lattice_correlations import allen_dynes_tc, correlation_matrices
from chemflow.eval.runner import EvaluationTask, Expected, score_task
from chemflow.runtime import Executor
from chemflow.types import Sector
from chemflow.verify.kernel import check_certificate


def task_spec():
    return TaskSpec.model_validate({
        "question": "Pair correlations in a Hubbard dimer",
        "target": {"kind": "pairing_correlation",
                   "lattice": {"model": "hubbard", "geometry": "chain", "n_sites": 2,
                               "t": 1., "U": 4., "unit": "hartree"},
                   "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
                   "target_spin": 0, "site_pair": [0, 1], "quantity": "pair_s_wave"},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 1e-6,
                   "unit": "dimensionless"},
                  {"kind": "relation_at_least", "relation": "approximate"}],
        "budget": {"wall_s": 600, "sim_shots": 100000},
        "allowed_effects": ["pure", "simulator"], "decision": "Choose a supported method"})


def test_explicit_algebra_doubly_occupied_and_singlet():
    sector = Sector(n_alpha=1, n_beta=1, n_orb=2)
    double = correlation_matrices([1., 0., 0., 0.], sector)
    assert double["density_density"] == [[4., 0.], [0., 0.]]
    assert double["spin_spin"] == [[0., 0.], [0., 0.]]
    assert double["pair_s_wave"] == [[1., 0.], [0., 0.]]
    singlet = correlation_matrices(np.array([0., 1., 1., 0.]) / np.sqrt(2), sector)
    assert np.allclose(singlet["spin_spin"], [[0.75, -0.75], [-0.75, 0.75]])
    assert np.allclose(singlet["density_density"], np.ones((2, 2)))
    assert np.allclose(singlet["pair_s_wave"], 0)
    with pytest.raises(ValueError, match="normalized"):
        correlation_matrices([2., 0., 0., 0.], sector)


@pytest.mark.parametrize("binding", ["fci_pyscf", "sector_diag_numpy"])
def test_exported_fci_vectors_and_sum_rules(binding):
    registry = load_registry(pack_names())
    with build.program("sum-rules") as builder:
        lattice = builder.node("lattice_model", n_sites=4, geometry="ring", U=4.)
        sector = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
        ground = builder.ground_energy(lattice, sector, s=0)
        builder.refine(ground, "fci")
        builder.bind(ground, binding)
        matrices = builder.node("correlation_function", state=ground)
        scalar = builder.node("correlation_function", state=ground, site_pair=[0, 2])
    executor = Executor(registry)
    computed = executor.run(builder.to_ir())
    vector = executor.store.get(computed[ground.id].diagnostics["state_ref"], verify=True)
    assert vector.shape == (36,)
    assert np.vdot(vector, vector) == pytest.approx(1)
    correlations = computed[matrices.id]
    assert np.sum(correlations["density_density"]) == pytest.approx(16)
    assert np.sum(correlations["spin_spin"]) == pytest.approx(0, abs=1e-10)
    assert computed[scalar.id].value.value == pytest.approx(correlations["pair_s_wave"][0][2])
    assert computed[scalar.id].diagnostics["relation_status"] == "proved"
    assert computed[scalar.id].relation == "equivalent"


def test_sampled_correlations_remain_approximate_and_identity_is_frozen():
    registry = load_registry(pack_names())
    record = SimpleNamespace(spec=task_spec())
    pack = pack_for_target(record.spec.target.kind)
    plan, _ = pack.base_plan(record)
    trial, candidates = Orchestrator._candidate_plan(plan, [{"template": "sqd", "shots": 4096}])
    assert any(node.op == "sample_configurations" for node in trial.nodes)
    computed = Executor(registry).run(trial)
    identifier = next(iter(candidates.values()))
    result = computed[identifier]
    assert result.relation == "approximate"
    assert result.diagnostics["relation_status"] == "proved"
    assert result.value.value == pytest.approx(computed[plan.references[0]].value.value, abs=1e-8)
    certificate = deepcopy(computed["relation_certificates"][identifier])
    certificate["steps"][-1]["lemma"] = "expectation_evaluation"
    certificate["steps"][-1]["conclusion"]["relation"] = "equivalent"
    certificate["relation"] = "equivalent"
    assert not check_certificate(certificate, trial, registry=registry, computed=computed).valid
    changed = trial.model_copy(deep=True)
    next(node for node in changed.nodes if node.id == identifier).params["site_pair"] = [1, 1]
    contexts, _ = infer_context(changed, registry)
    assert pack.identity(changed, record, contexts)
    assert default_axes(record.spec.target)[0].axis == "hubbard_u"


@pytest.mark.parametrize("patch", [{"site_pair": [-1, 0]}, {"site_pair": [True, 1]},
                                   {"site_pair": [0, 2]}, {"quantity": "d_wave"},
                                   {"extra": 1}])
def test_pairing_target_validation(patch):
    with pytest.raises(ValidationError):
        PairingTarget.model_validate({**task_spec().target.model_dump(), **patch})


@pytest.mark.parametrize("params", [{"lambda_ep": 0}, {"lambda_ep": True},
                                    {"omega_log_K": -1}, {"mu_star": -0.1},
                                    {"mu_star": 1}, {"lambda_ep": float("nan")}])
def test_allen_dynes_rejects_invalid_inputs(params):
    values = {"lambda_ep": 1., "omega_log_K": 300., "mu_star": .1, **params}
    with pytest.raises(ValueError):
        allen_dynes_tc(**values)
    with build.program("invalid-tc") as builder:
        builder.node("allen_dynes_tc", **values)
    assert check(builder.to_ir(), load_registry(pack_names()))


def test_dimensionless_scoring_uses_the_task_bar():
    task = EvaluationTask(id="dimer-pair", category="lattice", spec=task_spec(),
                          expected=Expected(value=.125, unit="dimensionless"),
                          reference_source="Analytic test value")
    for value, correct in ((.125, True), (.1250005, True), (.125002, False)):
        result = {"node": "answer", "value": f"{value} dimensionless",
                  "relation": "approximate", "gate_verdict": "PASS"}
        row = score_task(task, {"candidate": "answer", "results": [result]}, [])
        assert row["correct"] is correct
    for expected in ({"value": 1}, {"unit": "dimensionless"},
                     {"value": 1, "unit": "dimensionless", "value_hartree": 1}):
        with pytest.raises(ValidationError):
            Expected(**expected)


@pytest.mark.parametrize("spin", [None, 0])
def test_dmrg_sz_and_su2_are_variational_mps_expectations(spin):
    pytest.importorskip("pyblock2")
    registry = load_registry(pack_names())
    with build.program("dmrg-dimer") as builder:
        lattice = builder.node("lattice_model", n_sites=2, U=4.)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        ground = builder.ground_energy(lattice, sector, **({"s": spin} if spin is not None else {}))
        builder.refine(ground, "dmrg", bond_dimension=16, sweeps=5, seed=23)
        builder.bind(ground, "dmrg_block2")
    result = Executor(registry).run(builder.to_ir())[ground.id]
    assert result.value.value == pytest.approx(2 - np.sqrt(8), abs=1e-9)
    assert result.relation == "variational_upper_bound"
    assert result.diagnostics["relation_status"] == "proved"
    assert result.diagnostics["discarded_weight"] >= 0
    assert result.diagnostics["mps_norm"] == pytest.approx(1)


def test_registry_cards_and_resource_zeros():
    registry = load_registry(pack_names())
    report = validate_cards(registry, Path(__file__).resolve().parents[1])
    assert {card.id for card in report.cards} >= {"pairing_sqd", "dmrg_block2", "allen_dynes_tc"}
    for name in ("correlation_function", "dmrg_block2", "allen_dynes_tc"):
        implementation = registry.implementations[name]
        assert implementation.execution_role == "classical"
        assert implementation.resource_model() == {
            "gpu_s": 0, "qpu_s": 0, "qpu_shots": 0, "sim_shots": 0}
    assert registry.search("pairing")
    assert registry.search("dmrg")
    assert registry.search("allen_dynes")


def test_pairing_validity_uses_dimensionless_errors():
    specification = task_spec().model_dump()
    specification["validity"] = [{"axis": "hubbard_u", "range": [0., 4.],
                                  "points": 3, "resolution": .1}]
    store = TaskStore("test")
    record = store.freeze(TaskSpec.model_validate(specification), "human")
    registry = load_registry(pack_names())
    tools = make_tools(registry, Executor(registry), record, store)
    empty = tools.call("designer", "plan.new", {"identifier": "pairing"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty})["data"]
    reference = tools.plans.get(base["plan_id"]).references[0]
    result = tools.call("critic", "algorithm.validity", {
        "plan_id": base["plan_id"], "node_id": reference})["data"]
    assert result["task_point"]["status"] == "sampled_valid"
    assert all(point["error"] == pytest.approx(0) for point in result["axes"][0]["points"])


def test_unavailable_hf_vector_does_not_abort_screening(monkeypatch):
    from chemflow.domains import lattice_correlations

    monkeypatch.setattr(lattice_correlations, "hf_state", lambda *args: None)
    record = SimpleNamespace(spec=task_spec())
    plan, controls = pack_for_target(record.spec.target.kind).base_plan(record)
    result = Executor(load_registry(pack_names())).run(plan)
    assert controls[0] in result["refused"]
    assert "sector vector" in result["refused"][controls[0]]
    assert result[plan.references[0]].relation == "equivalent"
