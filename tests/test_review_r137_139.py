"""Reviewer tests for R137-R139 (written by Claude; implementers must not edit).

Found by the live glm-5.3 run on the `apps` suite (2026-10-15, 64039e4). Both GLM
configurations answered 13/13 correctly. In 7 of 13 applications, however, the answer came
from controls because GLM could not build a candidate.

- R137: the protein–ligand task's signed target carries the PDB text twice (receptor and
  ligand source, about 150k characters each). The LLM context inlined it: the first designer
  call used 150k input tokens, every later call about 300k, and the task as a whole 7.8M.
  `_summary` shortens lists but not long strings, and `plan.render` prints parameters
  verbatim.
- R138: `reaction_barrier`. `plan.candidates` built a candidate plan that `run`, `prove`
  and `falsify` all rejected with "CF-E120: Path geometry, evaluation or reference identity
  differs". The template expansion and the target's identity check disagree.
- R139: ranking, pes_scan, interaction_scan and bond_dissociation have no candidate
  templates, so a designer must compose every component by hand. GLM hit turn limits or
  declined. These composite targets combine ground energies, as reaction_energy already
  does, so `plan.candidates` can expand a template per component.

Contract:
- **R137.** Every LLM-facing payload caps strings at 2048 characters. A longer string is
  replaced by `{"text_chars", "text_sha256", "head"}`, with head ≤ 200 characters. This
  covers the task context in LLMPolicy messages, the observations, and every tool result
  sent to the model, `plan.render` included. Signed tasks and the executor keep the full
  text.
- **R138.** For the reaction_barrier application example, `plan.candidates` with
  `{"template": "sci", "select_cutoff": 1e-4}` yields a plan that screen-runs without
  CF-E120. Its candidate results have proved relations.
- **R139.** For the application examples of stability_ranking, pes_scan,
  dimer_binding_scan and bond_dissociation, the same SCI template yields:
  - at least one candidate id;
  - a plan that screen-runs;
  - candidate results for the target quantity, within the task's own bar of the screening
    reference;
  - proved relations.
"""
from __future__ import annotations

import hashlib
import json

import pytest


def _tools_for(app_id, index=0):
    from chemflow.agent.task import TaskSpec, TaskStore
    from chemflow.agent.tools import make_tools
    from chemflow.apps import draft
    from chemflow.domains import load_registry, pack_names
    from chemflow.runtime import Executor

    spec = TaskSpec.model_validate(draft(app_id, example=True)[index])
    store = TaskStore("reviewer")
    registry = load_registry(pack_names())
    tools = make_tools(registry, Executor(registry), store.freeze(spec, "reviewer"), store)
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty, "active": None})["data"]
    return spec, tools, base


def _long_strings(value, limit=2048):
    if isinstance(value, str):
        return [len(value)] if len(value) > limit else []
    if isinstance(value, dict):
        return [n for item in value.values() for n in _long_strings(item, limit)]
    if isinstance(value, list):
        return [n for item in value for n in _long_strings(item, limit)]
    return []


def test_r137_large_texts_never_reach_the_model_verbatim() -> None:
    from chemflow.agent.llm import LLMPolicy

    spec, tools, base = _tools_for("protein_ligand_binding")

    class Capture:
        model = "capture"

        def __init__(self):
            self.messages = []
            self.calls = 0

        def complete(self, messages, schemas):
            self.messages.append(json.loads(json.dumps(messages, default=str)))
            self.calls += 1
            if self.calls == 1:
                return {"tool_calls": [{"id": "c1", "name": "plan.render",
                                        "arguments": {"plan_id": base["plan_id"]}}],
                        "content": "", "stop_reason": "tool_calls", "usage": {}}
            return {"tool_calls": [], "content": "done", "stop_reason": "stop", "usage": {}}

    client = Capture()
    LLMPolicy("designer", client, tools).decide({"base_plan_id": base["plan_id"]})
    for messages in client.messages:
        for message in messages:
            content = message.get("content") or ""
            assert len(content) < 40000, (message.get("role"), len(content))
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                continue
            assert not _long_strings(payload), message.get("role")
    pdb = spec.target.receptor["pdb_text"]
    digest = hashlib.sha256(pdb.encode()).hexdigest()
    assert any(digest in json.dumps(messages) for messages in client.messages)


def _candidate_results(app_id):
    spec, tools, base = _tools_for(app_id)
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    assert built["ids"], app_id
    data = tools.call("designer", "run", {"plan_id": built["plan_id"], "scale": "screen"})["data"]
    return spec, base, built, data


@pytest.mark.parametrize("app_id", ["reaction_barrier", "stability_ranking", "pes_scan",
                                    "dimer_binding_scan", "bond_dissociation"])
def test_r138_r139_candidate_templates_work_for_composite_targets(app_id) -> None:
    _spec, _base, built, data = _candidate_results(app_id)
    headlines = data.get("headlines") or {}
    for node in built["ids"].values():
        assert node in headlines, (app_id, node)
        assert headlines[node].get("relation_status") == "proved", (app_id, node)
        gates = [gate for gate in data.get("gates", []) if gate.get("node") == node]
        assert gates and all(gate.get("passed") for gate in gates), (app_id, node, gates)
