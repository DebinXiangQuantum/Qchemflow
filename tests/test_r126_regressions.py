from __future__ import annotations

import json
from argparse import Namespace

import pytest

from chemflow.eval.runner import EvaluationTask, load_suite, score_task


def _run_batch_designer(tmp_path, monkeypatch, terminal):
    from chemflow.agent import cli
    from chemflow.eval.runner import run

    class Designer:
        def complete(self, messages, tools):
            built = [json.loads(message["content"])["result"]["data"]
                     for message in messages if message["role"] == "tool"]
            if built:
                name = terminal
                arguments = {"plan_id": built[-1]["plan_id"]}
                if terminal == "design.decline":
                    arguments["reason"] = "Batch exceeds remaining shots"
            else:
                name = "plan.candidates"
                base = json.loads(messages[1]["content"])["observation"]["base_plan_id"]
                arguments = {"plan_id": base, "candidates": [
                    {"template": "sqd", "shots": shots} for shots in (1024, 2048)]}
            return {"content": "", "tool_calls": [{"name": name, "arguments": arguments}]}

    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    monkeypatch.setattr(cli, "client_factory", lambda provider, role, model: Designer())
    result = run(Namespace(suite="full", suite_file=None, tasks="refuse-sim-shots",
                           policy="glm", model=None, roles="designer", jobs=1, out=tmp_path))
    return result


@pytest.mark.parametrize("relation,expected", [
    ("equivalent", True), ("variational_upper_bound", False),
    ("stochastic_estimate", False), ("approximate", False), ("unknown", False),
])
def test_relation_scoring_requires_every_signed_relation_gate(relation, expected):
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"].append({"kind": "relation_at_least", "relation": "stochastic_estimate"})
    task = EvaluationTask.model_validate(data)
    summary = {"candidate": "answer", "results": [{"node": "answer", "relation": relation}]}
    assert score_task(task, summary, [])["relation_ok"] is expected


def test_relation_scoring_without_answer_or_minimum():
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"] = [gate for gate in data["spec"]["gates"]
                             if gate["kind"] != "relation_at_least"]
    task = EvaluationTask.model_validate(data)
    summary = {"candidate": "missing", "results": []}
    assert score_task(task, summary, [])["relation_ok"] is None
    summary["results"] = [{"node": "missing", "relation": "model_change"}]
    assert score_task(task, summary, [])["relation_ok"] is True
