"""Reviewer regression tests for the GLM (Zhipu BigModel) policy (written by Claude;
implementers must not edit).

User request 2026-10-13: use GLM via BigModel for live agent runs. BigModel serves an
OpenAI-compatible chat-completions API. The reviewer verified that `glm-5.3` makes strict-schema
tool calls through ChemFlow's existing `OpenAIClient`.

Contract:
- `--policy glm` for `chemflow agent run` and `chemflow eval run`. Default model `glm-5.3`;
  `--model` overrides it.
- Base URL https://open.bigmodel.cn/api/paas/v4/, overridable with `CHEMFLOW_GLM_BASE_URL`.
- The key comes from `GLM_API_KEY`. Live use is opt-in: `CHEMFLOW_LIVE_LLM=1`, as for the other
  providers. It never falls back to scripted.
- The client is `OpenAIClient` with an injected `openai.OpenAI(api_key=..., base_url=...)`. The
  key is never written to summary.json, journals or reports.
- `chemflow.agent.cli.client_factory(provider, role, model)` stays the single factory the CLI
  and the harness use, so tests can inspect the configuration without a network call.
These tests are offline.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import test_review_agent_cli as agent

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")


def test_glm_is_opt_in_and_needs_its_key(tmp_path) -> None:
    task, key, _ = agent._signed(tmp_path)
    off = agent._agent("--task", str(task), "--public-key", str(key), "--out",
                       str(tmp_path / "off"), "--policy", "glm",
                       env={"CHEMFLOW_LIVE_LLM": "0", "GLM_API_KEY": "test-key"})
    assert off.returncode != 0 and "CHEMFLOW_LIVE_LLM" in off.stderr
    no_key = agent._agent("--task", str(task), "--public-key", str(key), "--out",
                          str(tmp_path / "nokey"), "--policy", "glm",
                          env={"CHEMFLOW_LIVE_LLM": "1", "GLM_API_KEY": ""})
    assert no_key.returncode != 0 and "GLM_API_KEY" in no_key.stderr
    assert not (tmp_path / "off" / "report.md").exists()
    assert not (tmp_path / "nokey" / "report.md").exists()


def test_the_glm_client_is_the_openai_adapter_pointed_at_bigmodel(monkeypatch) -> None:
    from chemflow.agent.cli import client_factory
    from chemflow.agent.llm_openai import OpenAIClient

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "test-key-not-real")
    monkeypatch.delenv("CHEMFLOW_GLM_BASE_URL", raising=False)
    client = client_factory("glm", "designer", None)
    assert isinstance(client, OpenAIClient) and client.model == "glm-5.3"
    assert str(client.client.base_url).rstrip("/") == "https://open.bigmodel.cn/api/paas/v4"
    monkeypatch.setenv("CHEMFLOW_GLM_BASE_URL", "https://example.invalid/v4/")
    other = client_factory("glm", "designer", "glm-5.3-flash")
    assert other.model == "glm-5.3-flash"
    assert str(other.client.base_url).rstrip("/") == "https://example.invalid/v4"


def test_the_key_never_reaches_outputs(tmp_path, monkeypatch) -> None:
    """Run the harness with a fake GLM client that only stops; the key must not leak."""
    from chemflow.agent import cli

    class Stop:
        model = "glm-5.3"

        def complete(self, messages, tools):
            return {"tool_calls": [], "content": "done", "stop_reason": "stop", "usage": {}}

    secret = "glm-secret-should-not-leak"
    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", secret)
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Stop())
    task, key, _ = agent._signed(tmp_path)
    out = tmp_path / "run"
    cli.run(Namespace(task=task, public_key=key, out=out, policy="glm", model=None,
                      roles="designer", rounds=1))
    written = "".join(path.read_text(errors="ignore") for path in out.rglob("*")
                      if path.is_file() and path.suffix in (".json", ".jsonl", ".md"))
    assert secret not in written
    assert json.loads((out / "summary.json").read_text())["policy"] == "glm"
