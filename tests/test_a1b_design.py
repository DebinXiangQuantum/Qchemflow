import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from test_review_a1a import ACCURACY, PACKS, _accept, _designed_plan, _enriched_plan
from test_review_a1b import TARGET

from chemflow.agent.critic import review
from chemflow.agent.report import result_headline, result_summary, write_recorded_report
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import ROLES, make_tools
from chemflow.backends.circuits import build_circuit, compile_native
from chemflow.design import (
    DesignStore,
    instantiate_design,
    register_design,
    verify_design,
)
from chemflow.domains import get_pack, load_registry
from chemflow.ids import content_hash
from chemflow.mcp import create_server
from chemflow.runtime import Executor
from chemflow.signing import TaskSigner, generate_keypair


def task_spec():
    return TaskSpec.model_validate({"question": "AIM response", "target": TARGET,
        "gates": [{"kind": "l1_error_vs_exact", "threshold": 0.1, "unit": "dimensionless"}],
        "budget": {"wall_s": 900}, "allowed_effects": ["pure", "simulator"], "decision": "d"})


@pytest.fixture
def verified(tmp_path):
    registry = load_registry(PACKS)
    store = DesignStore(tmp_path / "designs.sqlite")
    program, _, response = _designed_plan(full=True)
    _accept(program, response, {**ACCURACY, "threshold": 1e-8})
    report = verify_design(program, response, target=TARGET, registry=registry, store=store,
                           instances=2, seed=11)
    return registry, store, program, response, report


def test_report_is_immutable_content_addressed_and_persistent(verified, tmp_path):
    registry, store, program, response, report = verified
    payload = report.model_dump()
    payload.pop("id")
    assert report.id == content_hash(payload)
    with pytest.raises(FrozenInstanceError):
        report._payload = "{}"
    report.instances[0]["params"]["U"] = -100
    report.target["orbital"] = 99
    report.template["nodes"].clear()
    program.nodes.clear()
    assert store.get(report.id) == report
    assert report.target["orbital"] == 0
    assert report.template["nodes"]
    register_design(registry, store, report.id, "persisted")
    reopened = DesignStore(tmp_path / "designs.sqlite")
    assert reopened.get(report.id) == report
    assert reopened.list() == store.list()
    instance, target_node = instantiate_design(reopened, "persisted",
                                               model_params=TARGET["model"]["impurity_model"])
    assert target_node == response
    assert Executor(registry).run(instance)[target_node].diagnostics["design_evidence"] == report.id


def test_duplicate_names_and_unknown_reports_are_refused(verified):
    registry, store, _, _, report = verified
    with pytest.raises(KeyError):
        register_design(registry, store, "forged", "missing")
    register_design(registry, store, report.id, "unique")
    with pytest.raises(ValueError, match="Duplicate"):
        register_design(registry, store, report.id, "unique")
    with pytest.raises(ValueError, match="Duplicate"):
        register_design(load_registry(PACKS), store, report.id, "unique")


@pytest.mark.parametrize("count", [0, 2])
def test_refuse_zero_or_multiple_family_nodes(count):
    program, _, response = _designed_plan(full=True)
    family = next(node for node in program.nodes if node.op == "impurity_model")
    if count == 0:
        program.nodes.remove(family)
    else:
        program.nodes.append(family.model_copy(update={"id": "other_model"}, deep=True))
    with pytest.raises(ValueError, match="exactly one family node"):
        verify_design(_accept(program, response), response, target=TARGET,
                      registry=load_registry(PACKS), store=DesignStore(), instances=1, seed=5)


def test_missing_target_reference_is_refused():
    program, _, response = _designed_plan(full=True)
    with pytest.raises(ValueError, match="No exact reference"):
        verify_design(program, response, target={**TARGET, "kind": "excited_state"},
                      registry=load_registry(PACKS), store=DesignStore(),
                      instances=1, seed=5)


def test_fixed_seed_reproduces_draws_and_metrics_and_sums_resources():
    registry = load_registry(PACKS)
    program, response = _enriched_plan()
    _accept(program, response, {**ACCURACY, "threshold": 0.15})
    original = program.hash
    reports = [verify_design(program, response, target=TARGET, registry=registry,
                            store=DesignStore(), instances=3, seed=5)
               for _ in range(2)]
    assert program.hash == original
    assert reports[0].instances == reports[1].instances
    report = reports[0]
    assert report.resources["sim_shots"] == 60000
    assert report.resources["qpu_shots"] == 0
    assert report.resources["wall_s"] > 0
    assert report.resources["qubits"] == 8
    assert len(report.resources["circuits"]) == 3
    assert report.resources["native_2q_gates"] == sum(
        item["native_2q_gates"] for item in report.resources["circuits"])
    assert report.resources["native_depth"] == max(
        item["native_depth"] for item in report.resources["circuits"])
    instance = program.model_copy(deep=True)
    instance.gates = []  # rerun the design alone; its bar applies to verification
    next(node for node in instance.nodes if node.op == "impurity_model").params = (
        report.instances[0]["params"])
    computed = Executor(registry).run(instance)
    prepared = next(value for value in computed.values() if getattr(value, "circuit", None))
    native = compile_native(build_circuit(prepared.circuit))
    assert report.resources["circuits"][0]["native_2q_gates"] == sum(
        len(item.qubits) == 2 and item.operation.name != "barrier" for item in native.data)
    assert report.resources["circuits"][0]["native_depth"] == native.depth()


def test_random_family_ranges_and_fallback():
    draws = [get_pack("dynamics").random_instance("impurity_model", np.random.default_rng(seed),
                                                  n_bath=3) for seed in range(10)]
    assert all(1 <= draw["U"] <= 4 and len(draw["eps_bath"]) == 3 and
               all(-1.5 <= value <= 1.5 for value in draw["eps_bath"]) and
               all(0.3 <= value <= 0.8 for value in draw["t_bath"]) for draw in draws)
    assert get_pack("algorithms").random_instance("impurity_model", None) is NotImplemented
    assert get_pack("dynamics").random_instance("unknown", None) is NotImplemented


def test_energy_reference_and_relation_checks():
    registry = load_registry(PACKS)
    program, ground, _ = _designed_plan()
    target = {"kind": "ground_energy", "model": TARGET["model"], "sector": TARGET["sector"]}
    _accept(program, ground,
            {"kind": "abs_error_vs_exact", "threshold": 100, "unit": "hartree"})
    report = verify_design(program, ground, target=target, registry=registry, store=DesignStore(),
                           instances=2, seed=5)
    assert report.inferred_relation == "variational_upper_bound"
    assert report.passed
    assert all(item["metric_name"] == "abs_energy_error" and item["relation_holds"]
               for item in report.instances)
    program, _, response = _designed_plan(full=True)
    next(node for node in program.nodes if node.id == response).params["orbital"] = 1
    wrong = verify_design(_accept(program, response), response, target=TARGET,
                          registry=registry, store=DesignStore(),
                          instances=1, seed=11)
    assert wrong.max_metric < 100
    assert not wrong.passed
    assert not wrong.instances[0]["relation_holds"]


def test_instantiation_is_detached_and_rejects_forged_metadata(verified):
    registry, store, _, _, report = verified
    register_design(registry, store, report.id, "guarded")
    params = deepcopy(TARGET["model"]["impurity_model"])
    program, response = instantiate_design(store, "guarded", model_params=params)
    params["U"] = -999
    family = next(node for node in program.nodes if node.op == "impurity_model")
    assert family.params["U"] != -999
    node = next(node for node in program.nodes if node.id == response)
    node.params["orbital"] = 1
    with pytest.raises(ValueError, match="template was altered"):
        Executor(registry).run(program)
    program, response = instantiate_design(store, "guarded", model_params=params)
    next(node for node in program.nodes if node.id == response).metadata["design_evidence"] = "fake"
    with pytest.raises(ValueError, match="evidence metadata"):
        Executor(registry).run(program)


def test_registered_headlines_keep_evidence_and_critic_keeps_gates(verified):
    registry, design_store, _, _, report = verified
    register_design(registry, design_store, report.id, "reported")
    task_store = TaskStore("human")
    record = task_store.freeze(task_spec(), "human")
    base, _ = get_pack("dynamics").base_plan(record)
    program, response = instantiate_design(design_store, "reported",
                                           model_params=TARGET["model"]["impurity_model"])
    base.nodes.extend(program.nodes)
    base.edges.extend(program.edges)
    base.targets.append(response)
    computed = Executor(registry).run_plan(base, record, task_store)
    headline = result_headline(response, computed, {}, base.references[0], record.spec,
                               base, registry)
    assert headline["design_status"] == "experimental"
    assert report.id in headline["evidence"]
    summary = result_summary(response, computed, {}, base.references[0], record.spec, base)
    assert summary["gate_passed"]
    control = {**summary, "id": "classical", "relation": "approximate"}
    assert review([summary], [control]).decision == "stop"
    assert review([{**summary, "gate_passed": False}], []).decision == "revise"

    from chemflow.agent.memory import ResultsDB

    memory = ResultsDB()
    identifier = memory.add(task_id=record.content_hash, record_hash=record.hash,
                             plan_hash=base.hash, node_id=response, operator="spectral_function",
                             estimate=computed[response], headline=headline)
    text = write_recorded_report("# Design result", [{"result_id": identifier}], memory,
                                 record.hash)
    assert "experimental" in text and report.id in text


def test_roles_strict_schemas_persistence_and_frozen_target(tmp_path):
    registry = load_registry(PACKS)
    task_store = TaskStore("human")
    record = task_store.freeze(task_spec(), "human")
    tools = make_tools(registry, Executor(registry), record, task_store, plan_directory=tmp_path)
    for role in ROLES:
        names = {item["name"] for item in tools.schemas(role)}
        assert "algorithm.list" in names
        assert ("algorithm.verify" in names) == (role in {"designer", "orchestrator"})
        assert ("algorithm.register" in names) == (role in {"designer", "orchestrator"})
    schema = next(item for item in tools.schemas("designer") if item["name"] == "algorithm.verify")
    assert schema["parameters"]["additionalProperties"] is False
    assert set(schema["parameters"]["required"]) == {
        "plan_id", "target_node", "instances", "seed"}
    program, _, response = _designed_plan(full=True)
    plan_id = tools.plans.put(program)
    inputs = {"plan_id": plan_id, "target_node": response, "instances": 1,
              "seed": 11}
    with pytest.raises(ValueError):
        tools.call("designer", "algorithm.verify", {**inputs, "target": TARGET})
    report = tools.call("designer", "algorithm.verify", inputs)["data"]
    tools.call("designer", "algorithm.register", {"report_id": report["id"], "name": "tools"})
    reopened_registry = load_registry(PACKS)
    reopened = make_tools(reopened_registry, Executor(reopened_registry), record, task_store,
                           plan_directory=tmp_path)
    assert reopened.call("analyst", "algorithm.list")["data"][0]["id"] == "design:tools"
    assert reopened_registry.get("design:tools")["evidence"] == report["id"]
    searched = reopened.call("analyst", "registry.search", {"query": "design:"})["data"]
    assert searched[0]["id"] == "design:tools"
    retrieved = reopened.call("analyst", "registry.get", {"identifier": "design:tools"})["data"]
    assert retrieved["evidence"] == report["id"]


def test_mcp_exposes_design_tools_by_role(tmp_path):
    private, public = generate_keypair(b"human")
    key = tmp_path / "human.pub"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    task.write_text(TaskSigner(private, b"human").freeze(task_spec(), "human").model_dump_json())

    async def exercise():
        for profile in ("planner", "modeler", "designer", "critic", "analyst", "full"):
            server = create_server(profile, str(task), public_key=str(key), packs=",".join(PACKS))
            names = {item.name for item in await server.list_tools()}
            assert "algorithm_list" in names
            assert ("algorithm_verify" in names) == (profile in {"designer", "full"})
            assert ("algorithm_register" in names) == (profile in {"designer", "full"})

    asyncio.run(exercise())
