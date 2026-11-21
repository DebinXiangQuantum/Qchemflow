import json
from argparse import Namespace

import pytest

from chemflow.agent import cli
from chemflow.agent.llm import FakeLLMClient
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
    assert model["reason"].strip()
    assert model["role"] == "orchestrator"
    assert not any(entry["state"] == "MODEL_FALLBACK" for entry in entries)


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
