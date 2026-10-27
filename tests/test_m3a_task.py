import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError
from chemflow.ir.schema import AcceptanceGate
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def spec():
    return TaskSpec.model_validate({
        "question": "H2 ground energy", "target": {
            "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
            "electronic_model": {"basis": "sto-3g"},
            "active": {"n_elec": 2, "n_orb": 2},
            "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
            "target_spin": 0}, "gates": [{"kind": "abs_error_vs_exact",
                "threshold": 0.0016, "unit": "hartree"}],
        "budget": {"wall_s": 120}, "allowed_effects": ["pure", "simulator"],
        "decision": "choose a method"})


def test_record_integrity_and_version():
    store = TaskStore("secret")
    record = store.freeze(spec(), "human")
    store.verify(record)
    edited = record.spec.model_copy(update={"budget": {"wall_s": 100}})
    changed_gate = record.spec.model_copy(update={"gates": (record.spec.gates[0].model_copy(
        update={"threshold": 0.01}),)})
    for forged in (record.model_copy(update={"spec": edited}),
                   record.model_copy(update={"spec": changed_gate}),
                   record.model_copy(update={"approver": "other"}),
                   record.model_copy(update={"signature": "0" * 64})):
        with pytest.raises(AdapterError, match="CF-E120"):
            store.verify(forged)
    with pytest.raises(AdapterError, match="CF-E120"):
        TaskStore("different").verify(record)
    successor = store.freeze(edited, "human", predecessor=record)
    assert successor.version == 2 and successor.predecessor_hash == record.hash


def test_plan_targets_and_gates():
    from chemflow.agent.orchestrator import Orchestrator

    store = TaskStore("secret")
    record = store.freeze(spec(), "human")
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    executor = Executor(molecular_registry())
    assert executor.run_plan(plan, record, store)["gates"][-1].passed
    plan.gates.append(AcceptanceGate(node=plan.targets[0], relation="equivalent"))
    with pytest.raises(AdapterError, match="CF-E120"):
        executor.run_plan(plan, record, store)
    plan.gates.clear()
    plan.targets = [plan.nodes[0].id]
    with pytest.raises(AdapterError, match="CF-E120"):
        executor.run_plan(plan, record, store)


def test_missing_exact_reference_fails_gate():
    from chemflow.agent.orchestrator import Orchestrator

    store = TaskStore("secret")
    record = store.freeze(spec(), "human")
    plan, _ = Orchestrator._base_plan(record, record.spec.target.active)
    plan.nodes[-1].params["s"] = 0
    plan.nodes[-1].refinement = "sci"
    plan.nodes[-1].binding = "sci_pyscf"
    plan.nodes[-1].params["select_cutoff"] = 0.01
    result = Executor(molecular_registry()).run_plan(plan, record, store)
    assert not result["gates"][0].passed
    assert "Missing exact reference" in result["gates"][0].reason
