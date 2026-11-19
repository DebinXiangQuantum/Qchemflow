import json
from argparse import Namespace

import pytest
from test_m3a_task import spec
from test_m3b_llm import policy

from chemflow.agent import cli
from chemflow.agent.llm import MODELER_PROMPT, FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore
from chemflow.eval import runner


def response(name=None, arguments=None):
    return {"content": "Use this space; this is prose, not JSON.", "tool_calls": [] if name is None
            else [{"id": "choice", "name": name, "arguments": arguments}]}


def read_entries(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("task_id", [
    "h2-ground", "hubbard-ring6", "impurity-spectral", "h2-atomization", "h2-counterpoise",
])
def test_fixed_and_full_spaces_never_consult_llm_modeler(tmp_path, monkeypatch, task_id):
    clients = []

    def client_factory(*args):
        client = FakeLLMClient([])
        clients.append(client)
        return client

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test")
    monkeypatch.setattr(cli, "client_factory", client_factory)
    results = runner.run(Namespace(suite="core", suite_file=None, tasks=task_id, policy="glm",
                                   model=None, roles="modeler", jobs=1, out=tmp_path / "out"))
    assert results["rows"][0]["correct"]
    assert clients and all(not client.messages for client in clients)
    entries = read_entries(tmp_path / "out" / task_id / "journal.jsonl")
    model = next(entry for entry in entries if entry["state"] == "MODEL")
    controls = next(entry for entry in entries if entry["state"] == "CONTROLS")
    assert model["reason"].strip()
    assert model["role"] == "orchestrator"
    assert model["decision"]["active"] == controls["observation"]["active"]
    assert controls["observation"]["fallback"] is False


@pytest.mark.parametrize("choice,expected_fallback", [
    ({"n_elec": 2, "n_orb": 2}, False),
    ({"n_elec": 3, "n_orb": 2}, True),
    ({"n_elec": 6, "n_orb": 2}, True),
    ({"n_elec": 2, "n_orb": 11}, True),
    ({"n_elec": 0, "n_orb": 2}, True),
    ({"n_elec": 2, "n_orb": 0}, True),
    (None, True),
])
def test_auto_choice_or_diagnostics_fallback_never_parses_prose(tmp_path, choice,
                                                             expected_fallback):
    task = spec()
    task = task.model_copy(update={"target": task.target.model_copy(
        update={"active": "auto", "sector": {}})})
    reply = response("model.choose", {**choice, "reason": "Test choice"}) if choice else response()
    client = FakeLLMClient([reply])
    store = TaskStore("offline")
    modeler = LLMPolicy("modeler", client, None)
    agent = Orchestrator(store, lambda draft: store.freeze(draft, "human"), tmp_path,
                         max_rounds=1, policies={"modeler": modeler})
    suggested = {"n_elec": 2, "n_orb": 2}
    result = agent.run(task, diagnostics={"suggested_active": suggested})
    entries = read_entries(result.journal)
    controls = next(entry for entry in entries if entry["state"] == "CONTROLS")
    assert controls["observation"] == {"active": suggested, "fallback": expected_fallback}
    fallbacks = [entry for entry in entries if entry["state"] == "MODEL_FALLBACK"]
    assert len(fallbacks) == int(expected_fallback)
    assert all(entry["reason"].strip() for entry in fallbacks)
    assert len(client.messages) == 1
    assert client.messages[0][0]["content"] == MODELER_PROMPT
    assert json.loads(client.messages[0][1]["content"])["observation"]["diagnostics"] == {
        "suggested_active": suggested}


def test_model_choice_is_modeler_only_and_stops_same_response(tmp_path):
    arguments = {"n_elec": 2, "n_orb": 2, "reason": "Enough for H2"}
    reply = response("model.choose", arguments)
    reply["tool_calls"].extend(response("plan.new", {"identifier": "not-executed"})["tool_calls"])
    agent, client, _ = policy(tmp_path, "modeler", [reply])
    result = agent.decide({})
    assert result["outputs"][0]["result"]["data"] == arguments
    assert len(result["outputs"]) == len(client.messages) == 1
    assert result["end_reason"] == "model.choose"
    assert not agent.tools.plans._plans
    for role in ("planner", "designer", "critic", "analyst", "orchestrator"):
        assert "model.choose" not in {schema["name"] for schema in agent.tools.schemas(role)}
        with pytest.raises(PermissionError):
            agent.tools.call(role, "model.choose", arguments)


@pytest.mark.parametrize("active,spin,valid", [
    ({"n_elec": 1, "n_orb": 2}, 0.5, True),
    ({"n_elec": 2, "n_orb": 2}, 0.5, False),
    ({"n_elec": 1, "n_orb": 2}, 1.5, False),
    ({"n_elec": True, "n_orb": 2}, 0, False),
])
def test_active_space_feasibility_handles_half_integer_spin(active, spin, valid):
    assert (Orchestrator._active_error(active, spin) is None) is valid


def test_other_llm_roles_accept_prose_without_parsing_it(tmp_path, monkeypatch):
    clients = {}

    def client_factory(provider, role, model):
        clients[role] = FakeLLMClient([response()] * 4)
        return clients[role]

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test")
    monkeypatch.setattr(cli, "client_factory", client_factory)
    results = runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="glm",
                                   model=None, roles="planner,critic,analyst", jobs=1,
                                   out=tmp_path / "out"))
    assert results["rows"][0]["correct"]
    assert all(client.messages for client in clients.values())
