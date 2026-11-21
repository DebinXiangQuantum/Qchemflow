import json

import pytest

from chemflow.agent.task import TaskSpec
from chemflow.apps import catalog, draft


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


def test_missing_inputs_and_unknown_apps_are_errors():
    with pytest.raises(ValueError, match="inputs"):
        draft("ground_state_energy")
    with pytest.raises(ValueError, match="Unknown"):
        draft("missing", example=True)
