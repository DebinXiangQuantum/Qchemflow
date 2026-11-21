import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError


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
