import asyncio
import json
from argparse import Namespace

import pytest

from chemflow.agent.task import TaskSpec
from chemflow.apps import draft
from chemflow.eval.runner import EvaluationTask, score_task
from chemflow.signing import TaskSigner, generate_keypair


def test_mcp_drafts_are_planner_only_and_unsigned(tmp_path):
    pytest.importorskip("mcp")
    from chemflow.mcp import create_server

    private, public = generate_keypair(b"test-human")
    key = tmp_path / "public.pem"
    key.write_bytes(public)
    task = tmp_path / "task.json"
    task.write_text(TaskSigner(private, b"test-human").freeze(
        TaskSpec.model_validate(draft("ground_state_energy", example=True)[0]),
        "human-test-fixture").model_dump_json())

    async def exercise():
        planner = create_server("planner", str(task), public_key=str(key))
        names = {tool.name for tool in await planner.list_tools()}
        assert {"app_list", "app_draft"} <= names
        result = await planner.call_tool("app_draft", {"app_id": "frontier_orbitals",
            "inputs": None, "parameters": '{"physical_gap": true}', "example": True})
        drafts = json.loads(result.content[0].text)["data"]
        assert len(drafts) == 5
        assert "signature" not in json.dumps(drafts)
        for item in drafts:
            TaskSpec.model_validate(item)
        designer = create_server("designer", str(task), public_key=str(key))
        assert "app_draft" not in {tool.name for tool in await designer.list_tools()}

    asyncio.run(exercise())


def test_metabolic_scoring_checks_each_site():
    spec = draft("metabolic_sites", example=True)[0]
    task = EvaluationTask(id="sites", category="property", spec=spec,
        expected={"value": 0.5, "unit": "dimensionless", "site_scores": {"1": 0.5, "2": 0.2}},
        reference_source="Synthetic scorer unit fixture")
    headline = {"node": "answer", "value": "0.5 dimensionless", "relation": "equivalent",
                "sites": [{"atom": 1, "score": 0.5}, {"atom": 2, "score": 0.2}]}
    summary = {"candidate": "answer", "decision": "stop", "results": [headline]}
    assert score_task(task, summary, [])["correct"]
    headline["sites"][1]["score"] = 0.3
    assert not score_task(task, summary, [])["correct"]


@pytest.mark.parametrize("parameters", [{"typo": True}, {"error": -1}, {"wall_s": float("inf")},
                                      {"qpu": "false"}, {"qpu_shots": 0}])
def test_invalid_resource_or_parameter_rejected(parameters):
    with pytest.raises(ValueError):
        draft("ground_state_energy", parameters=parameters, example=True)


def test_explicit_smiles_overrides_example_molecule():
    task = draft("frontier_orbitals", inputs={"smiles": "O"}, example=True)[0]
    assert task["target"]["molecule"]["atoms"].startswith("O ")


def test_empty_solvent_list_rejected():
    with pytest.raises(ValueError):
        draft("solvation_scan", parameters={"solvents": []}, example=True)


def test_missing_binding_environment_is_a_structured_refusal(tmp_path, monkeypatch):
    import yaml

    from chemflow.eval.runner import run

    monkeypatch.setenv("CHEMFLOW_TOOLS_PYTHON", str(tmp_path / "missing-python"))
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "missing-binding-tools", "tasks": [{
        "id": "binding", "category": "binding", "spec": draft("protein_ligand_binding",
                                                                 example=True)[0],
        "expected": {"value": -10.22, "unit": "kcal/mol"},
        "reference_source": "Independent fixture, expected unavailable environment"}]}))
    result = run(Namespace(suite=None, suite_file=path, tasks=None, policy="scripted",
        model=None, roles="designer", jobs=1, out=tmp_path / "output"))
    assert result["rows"][0]["decision"] == "refused"
    summary = json.loads((tmp_path / "output/binding/summary.json").read_text())
    assert summary["refusal"]["code"] == "CF-E080"


def test_optional_rate_is_reported_from_verified_gibbs_barrier(tmp_path):
    from chemflow.agent.orchestrator import Orchestrator
    from chemflow.agent.task import TaskStore

    spec = TaskSpec.model_validate(draft("reaction_barrier", parameters={"rate_constant": True},
                                        example=True)[0])
    store = TaskStore("test-human-only")
    agent = Orchestrator(store, lambda task: store.freeze(task, "human-test-fixture"), tmp_path,
                         max_rounds=0)
    result = agent.run(spec)
    assert "model Eyring k=" in result.report
    assert all(item["rate_constant"]["k_per_s"] > 0 for item in result.results)
