import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
from test_agent_cli_run import invoke, signed

from chemflow.agent import cli as agent_cli
from chemflow.agent.journal import Journal
from chemflow.agent.llm import CONTRACTS, LLMPolicy
from chemflow.agent.policies import ScriptedPolicy
from chemflow.agent.tools import ToolRegistry
from chemflow.cli import main

openai = pytest.importorskip("openai")


def fake_sdk(monkeypatch, error=None):
    configurations, calls = [], []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return openai.types.chat.ChatCompletion.model_validate({
            "id": "offline-glm", "object": "chat.completion", "created": 1,
            "model": kwargs["model"],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}]})

    def constructor(**kwargs):
        configurations.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-glm-secret")
    monkeypatch.delenv("CHEMFLOW_GLM_BASE_URL", raising=False)
    monkeypatch.setattr(openai, "OpenAI", constructor)
    return configurations, calls


@pytest.mark.parametrize("model", [None, "glm-4.7", "glm-5.3-flash"])
def test_factory_configuration(monkeypatch, model):
    configurations, _ = fake_sdk(monkeypatch)
    client = agent_cli.client_factory("glm", "designer", model)
    assert client.model == (model or "glm-5.3")
    assert configurations == [{"api_key": "offline-glm-secret",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/", "max_retries": 0, "timeout": 120}]


def test_selected_roles_and_journal(tmp_path, monkeypatch):
    configurations, calls = fake_sdk(monkeypatch)
    agents = []
    original = agent_cli.Orchestrator

    def orchestrator(*args, **kwargs):
        agent = original(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(agent_cli, "Orchestrator", orchestrator)
    _, task, key = signed(tmp_path)
    out = tmp_path / "run"
    invoke(monkeypatch, task, key, out, "--policy", "glm", "--roles", "designer,analyst")
    assert len(configurations) == 2
    for role in CONTRACTS:
        policy = agents[0].policies[role]
        if role in {"designer", "analyst"}:
            assert isinstance(policy, LLMPolicy)
            assert policy.provider == "glm" and policy.client.model == "glm-5.3"
        else:
            assert isinstance(policy, ScriptedPolicy)
    entries = [json.loads(line) for line in (out / "journal.jsonl").read_text().splitlines()]
    decisions = [entry["decision"] for entry in entries if entry["state"] == "LLM_CALL"]
    assert decisions and len(decisions) == len(calls)
    for decision in decisions:
        assert decision["provider"] == "glm" and decision["model"] == "glm-5.3"
        assert decision["usage"] == {"input_tokens": 8, "output_tokens": 3}
        assert decision["latency_s"] >= 0


@pytest.mark.parametrize("error_name,status,code,retryable", [
    ("AuthenticationError", 401, "authentication", False),
    ("RateLimitError", 429, "rate_limit", True),
])
def test_adapter_errors_and_redaction(tmp_path, monkeypatch, error_name, status, code, retryable):
    monkeypatch.setattr("chemflow.agent.llm.time.sleep", lambda seconds: None)
    response = SimpleNamespace(status_code=status, request=SimpleNamespace(), headers={})
    error = getattr(openai, error_name)("offline-glm-secret", response=response, body=None)
    fake_sdk(monkeypatch, error)
    journal = Journal(tmp_path / "journal.jsonl")
    policy = LLMPolicy("designer", agent_cli.client_factory("glm", "designer", None),
                       ToolRegistry(), journal=journal, provider="glm")
    policy.decide({})
    written = journal.path.read_text()
    assert "offline-glm-secret" not in written
    decision = json.loads(written.splitlines()[0])["decision"]
    assert decision["error"]["code"] == code
    assert decision["error"]["retryable"] is retryable
    assert "[REDACTED]" in decision["error"]["message"]
    assert decision["provider"] == "glm" and decision["latency_s"] >= 0


@pytest.mark.parametrize("key", [None, "", "   "])
def test_preflight_requires_glm_key(monkeypatch, key):
    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-glm-key")
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("GLM_API_KEY", key)
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        agent_cli.validate_options(Namespace(policy="glm", model=None, roles="designer", rounds=1))


@pytest.mark.parametrize("model", [None, "glm-5.3-flash"])
def test_evaluation_uses_shared_factory(tmp_path, monkeypatch, model):
    configurations, calls = fake_sdk(monkeypatch)
    out = tmp_path / "evaluation"
    arguments = ["chemflow", "eval", "run", "--suite", "core", "--tasks", "h2-ground",
                 "--policy", "glm", "--out", str(out)]
    if model:
        arguments.extend(["--model", model])
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    assert len(configurations) == 1 and calls
    assert all(call["model"] == (model or "glm-5.3") for call in calls)
    metadata = json.loads((out / "results.json").read_text())["metadata"]
    assert metadata["policy"] == "glm" and metadata["model"] == (model or "glm-5.3")
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            assert "offline-glm-secret" not in path.read_text()
