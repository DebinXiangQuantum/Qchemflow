"""Reviewer regression tests for the full-project review FULL-1 (R77–R82; written by Claude;
implementers must not edit). Findings came from a read-only Codex (gpt-6-astra) review and were
reproduced by the reviewer before being recorded. R79 revised for R1 (QPU as hardware)."""
from __future__ import annotations

import json

import pytest
from test_m3a_task import spec

from chemflow import build
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.ir.schema import Node
from chemflow.ops.molecular import molecular_registry
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ExecutionResult, Executor
from chemflow.types import Samples


def _tools(spec_data=None):
    store = TaskStore("reviewer")
    record = store.freeze(TaskSpec.model_validate(spec_data) if spec_data else spec(), "reviewer")
    registry = molecular_registry()
    return make_tools(registry, Executor(registry), record, store), record, store


def _base(tools):
    empty = tools.call("modeler", "plan.new", {})["data"]["plan_id"]
    return tools.call("modeler", "plan.base", {"plan_id": empty,
                                               "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]


def test_r77_node_parameters_cannot_change_what_a_binding_computes() -> None:
    tools, _, _ = _tools()
    base = _base(tools)
    reference = next(node.id for node in tools.plans.get(base).nodes
                     if node.binding == "fci_pyscf")
    exact = tools.call("designer", "run", {"plan_id": base})["data"]["estimates"][reference]
    try:
        edited = tools.call("designer", "plan.refine", {
            "plan_id": base, "node_id": reference, "template": "fci",
            "params": [{"key": "method", "value_json": json.dumps("hf")}]})["data"]
        if edited.get("diagnostics"):
            return  # refused at edit time
        estimate = tools.call("designer", "run", {"plan_id": edited["plan_id"]})["data"][
            "estimates"][reference]
    except (PermissionError, ValueError, CheckError, AdapterError):
        return  # refused
    assert estimate["value"]["value"] == pytest.approx(exact["value"]["value"], abs=1e-10), \
        "an fci-labelled binding computed something else"


def test_r78_reports_cannot_publish_unrecorded_results() -> None:
    tools, _, _ = _tools()
    invented = {"node": "invented", "value": "-100 hartree", "relation": "equivalent",
                "evidence": ["certified_bound"], "gate_verdict": "PASS",
                "cost": {"qpu_s": 0}, "backend_kind": "qpu", "advantage_evidence": True}
    for markdown in ("Summary of the screen.", "Quantum advantage and speedup observed."):
        with pytest.raises(Exception):  # noqa: B017 - any refusal is acceptable
            tools.call("analyst", "report.write", {"markdown": markdown,
                                                   "headlines_json": json.dumps([invented])})
    base = _base(tools)
    recorded = tools.call("designer", "run", {"plan_id": base})["data"]["results"]
    assert recorded
    forged = {**invented, "result_id": next(iter(recorded.values())), "value": "-100 hartree"}
    with pytest.raises(Exception):  # noqa: B017 - a real id must not launder a different value
        tools.call("analyst", "report.write", {"markdown": "Summary.",
                                               "headlines_json": json.dumps([forged])})


def _paid_plan(record, count):
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    hamiltonian = next(node.id for node in plan.nodes if node.op == "hamiltonian")
    sector = next(node.id for node in plan.nodes if node.op == "sector")
    plan.nodes.append(Node(id="prep", op="prepare_state", params={"ansatz": "hf"},
                           inputs={"hamiltonian": hamiltonian, "sector": sector}))
    for index in range(count):
        plan.nodes.append(Node(id=f"paid{index}", op="sample_configurations", binding="paid",
                               inputs={"prep": "prep"}, params={"shots": 4, "seed": index}))
    return plan


def test_r79_a_hardware_node_is_not_run_when_spend_so_far_plus_its_prediction_exceeds_budget():
    """Revised for R1: the limit is the task's budget, not an approval. Revised for R99: the
    refused node is reported and the run returns what it already used. Revised for D-R2: the
    limited resource is QPU device time (money is not a resource)."""
    charges = []

    def adapter(store, prep, shots, seed, **kwargs):
        charge = 15.0 if not charges else 10.0
        charges.append(charge)
        return ExecutionResult(Samples(counts={"5": shots}, shots=shots, encoding_id=prep.encoding_id,
                                       basis="computational", source="qpu", seed=seed),
                               device="fake", cost={"qpu_s": charge})

    data = spec().model_dump()
    data.update(budget={"qpu_s": 20.0, "wall_s": 600}, allowed_effects=("pure", "simulator"))
    store = TaskStore("reviewer")
    record = store.freeze(TaskSpec.model_validate(data), "reviewer")
    registry = molecular_registry()
    registry.register(ImplementationSpec(
        id="paid", version="1", target="sample_configurations", backend_kind="qpu",
        effects="random", adapter=adapter,
        resource_model=lambda **kwargs: {"qpu_s": 10.0}))
    plan = _paid_plan(record, 2)
    computed = Executor(registry).run_plan(plan, record, store)  # R99: the paid result is kept
    assert charges == [15.0], "second hardware job ran although 15 + 10 > 20"
    ran = [node for node in ("paid0", "paid1") if node in computed]
    assert len(ran) == 1 and ({"paid0", "paid1"} - set(ran)) <= set(computed["refused"])
    assert all("CF-E070" in str(reason) for reason in computed["refused"].values())


def test_r80_shot_based_sampling_is_counted_against_the_shot_budget() -> None:
    with build.program("shots") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                          builder.electronic_model(basis="sto-3g"))
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        sample = builder.node("sample", operator=encoding, prep=prep, shots=1000, seed=11)
        builder.budget(sim_shots=0)
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(molecular_registry()).run(builder.to_ir())
    assert "CF-E070" in {item.code for item in info.value.diagnostics}
    program = builder.to_ir()
    program.budget = {"sim_shots": 1_000_000}
    result = Executor(molecular_registry()).run(program)
    assert result["costs"][sample.id].sim_shots and result["costs"][sample.id].sim_shots >= 1000


def test_r81_task_budgets_are_cumulative_across_runs() -> None:
    data = spec().model_dump()
    data.update(budget={"sim_shots": 8, "wall_s": 600})
    tools, record, _ = _tools(data)
    plan = _paid_plan(record, 1)
    node = next(item for item in plan.nodes if item.id == "paid0")
    node.binding, node.params = "sample_configurations", {"shots": 8, "seed": 1}
    identifier = tools.plans.put(plan)
    tools.call("designer", "run", {"plan_id": identifier, "scale": "screen"})
    with pytest.raises(Exception) as info:
        tools.call("designer", "run", {"plan_id": identifier, "scale": "screen"})
    assert "CF-E070" in str(info.value) or "budget" in str(info.value).lower()


def test_r82_cz_noise_applies_in_both_qubit_orders() -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from chemflow.backends.aer_noise import noise_model

    config = {"properties": {
        "qubit_metrics": [{"id": 0, "measure_f0": 1, "measure_f1": 1},
                          {"id": 1, "measure_f0": 1, "measure_f1": 1}],
        "coupler_metrics": [{"qubits": [0, 1], "cz_fidelity": 0.0}]}}
    noise = noise_model(config, [0, 1])
    for pair in ((0, 1), (1, 0)):
        circuit = QuantumCircuit(2)
        circuit.cz(*pair)
        circuit.measure_all()
        counts = AerSimulator(noise_model=noise).run(circuit, shots=2000,
                                                      seed_simulator=3).result().get_counts()
        assert counts.get("00", 0) < 1000, f"CZ{pair} ran noiseless"
