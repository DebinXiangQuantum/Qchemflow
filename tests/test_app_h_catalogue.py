import json

import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.apps import catalog, draft
from chemflow.check import check
from chemflow.domains import load_registry, pack_for_target, pack_names
from chemflow.runtime import Executor


def test_catalogue_is_detached_and_drafts_are_unsigned():
    first = catalog()
    first[0]["example"]["molecule"]["atoms"] = "changed"
    assert catalog()[0]["example"]["molecule"]["atoms"] != "changed"
    for entry in catalog():
        tasks = draft(entry["id"], example=True)
        for task in tasks:
            TaskSpec.model_validate(task)
            assert "signature" not in json.dumps(task)
            assert task["budget"]["wall_s"] > 0


def test_qpu_request_does_not_bind_or_execute():
    task = draft("ground_state_energy", parameters={"qpu": True}, example=True)[0]
    assert task["budget"]["qpu_shots"] > 0 and task["budget"]["qpu_s"] > 0
    assert all(text in task["decision"] for text in ("vqe_sampled", "MQ02", "R130"))
    assert "binding" not in task and "signature" not in task


def test_conformer_drafts_are_deterministic_and_allow_deduplication():
    first = draft("conformer_ranking", example=True)
    assert first == draft("conformer_ranking", example=True)
    assert first[0]["target"]["items"]
    assert "no draft accuracy claim" in first[0]["decision"]


def test_optional_properties_and_solvents():
    assert len(draft("frontier_orbitals", parameters={"physical_gap": True}, example=True)) == 5
    assert draft("aqueous_solubility", parameters={"melting_point_C": 40.},
                 example=True)[0]["target"]["model"] == "gse"
    pcm = draft("solvation_scan", parameters={"solvents": [{"model": "pcm", "eps": 78.3553},
                            {"model": "pcm", "eps": 9.8629}]}, example=True)
    assert len(pcm) == 2
    assert all(task["target"]["level"]["method"] == "hf" for task in pcm)
    rate = draft("reaction_barrier", parameters={"rate_constant": True}, example=True)[0]
    assert rate["target"]["thermo"]["quantity"] == "gibbs"
    assert rate["target"]["rate_constant"]


def test_metabolic_registry_identity_execution_and_relation():
    spec = TaskSpec.model_validate(draft("metabolic_sites", example=True)[0])
    record = TaskStore("unit-test-only").freeze(spec, "human-test-fixture")
    registry = load_registry(pack_names())
    pack = pack_for_target("metabolic_sites")
    plan, controls = pack.base_plan(record)
    assert controls and plan.references and not check(plan, registry)
    assert registry.search("metabolic")
    assert not pack.identity(plan, record, {})
    computed = Executor(registry).run(plan)
    result = computed[controls[0]]
    assert result.value.unit == "dimensionless"
    assert 0 < result.value.value < 1
    assert result.diagnostics["evidence"] == "empirical"
    assert result.diagnostics["relation_status"] == "proved"
    assert result.relation == "equivalent"
    plan.nodes[0].params["level"]["method"] = "gfn2"
    assert pack.identity(plan, record, {})
    assert any(item.code == "CF-E080" for item in check(plan, registry))


def test_missing_inputs_and_unknown_apps_are_errors():
    with pytest.raises(ValueError, match="inputs"):
        draft("ground_state_energy")
    with pytest.raises(ValueError, match="Unknown"):
        draft("missing", example=True)
