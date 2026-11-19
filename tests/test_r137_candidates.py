import hashlib
import json

import pytest

from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy, _cap_strings
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.apps import draft
from chemflow.check.deployment import deployment
from chemflow.check.static import check, infer_context
from chemflow.domains import load_registry, pack_for_target, pack_names


def test_caps_nested_strings_without_mutation():
    text = "λ" * 2049
    payload = {"nested": [{"text": text}], "boundary": "x" * 2048}
    capped = _cap_strings(payload)
    assert capped["nested"][0]["text"] == {
        "text_chars": 2049, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "head": text[:200],
        "ref": "pass this object back as the parameter value to reuse the frozen text"}
    assert capped["boundary"] == payload["boundary"]
    assert payload["nested"][0]["text"] == text
    assert _cap_strings(capped) == capped


@pytest.mark.parametrize("tool_name", ["plan.render", "registry.cards", "results.get"])
def test_all_conversation_payloads_are_capped(tool_name, tmp_path):
    text = "payload" * 1000

    class Tools:
        def schemas(self, role):
            return [{"name": tool_name, "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}}}]

        def call(self, role, name, args):
            assert args == {"query": text}
            return {"data": {"nested": [text]}}

    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, messages, schemas):
            self.calls.append(json.loads(json.dumps(messages)))
            return {"content": text, "tool_calls": [{"id": "call", "name": tool_name,
                "arguments": {"query": text}}] if len(self.calls) == 1 else []}

    client = Client()
    policy = LLMPolicy("designer", client, Tools(), journal=Journal(tmp_path / "journal.jsonl"))
    policy.decide({"nested": [text]})
    assert len(client.calls) == 2
    expected = _cap_strings(text)
    messages = client.calls[-1]
    assert json.loads(messages[1]["content"])["observation"]["nested"] == [expected]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert json.loads(assistant["content"]) == expected
    assert assistant["tool_calls"][0]["arguments"]["query"] == expected
    tool = next(message for message in messages if message["role"] == "tool")
    assert json.loads(tool["content"])["result"]["data"]["nested"] == [expected]


@pytest.mark.parametrize("app", ["reaction_barrier", "stability_ranking", "pes_scan",
                                 "dimer_binding_scan", "bond_dissociation"])
@pytest.mark.parametrize("template", ["sci", "sqd"])
def test_composite_candidates_preserve_identity_and_components(app, template):
    spec = TaskSpec.model_validate(draft(app, example=True)[0])
    record = TaskStore("test").freeze(spec, "test")
    registry = load_registry(pack_names())
    base, _ = pack_for_target(spec.target.kind).base_plan(record)
    plan, identifiers = Orchestrator._candidate_plan(base, [{"template": template}])
    assert identifiers
    assert not check(plan, registry, purpose="deploy")
    pack = pack_for_target(spec.target.kind)
    contexts, _ = infer_context(plan, registry)
    assert not pack.identity(plan, record, contexts)
    original = {node.id: node for node in base.nodes}
    added = [node for node in plan.nodes if node.id not in original]
    solvers = [node for node in added if node.op == "ground_energy"]
    assert solvers and all(node.refinement == template for node in solvers)

    def component_count(identifier):
        node = original[identifier]
        if node.op in ("ground_energy", "component_energy", "app_component_energy"):
            return 1
        return sum(component_count(source) for port, source in node.inputs.items()
                   if not port.startswith("thermo"))

    assert len(solvers) == component_count(base.references[0])
    result = next(node for node in plan.nodes if node.id in identifiers.values())
    assert result.params == original[base.references[0]].params
    assert plan.references == base.references
    if template == "sqd":
        assert len([node for node in added if node.op == "sample_configurations"]) == len(solvers)
    deploy, issues = deployment(plan, registry, targets=list(identifiers.values()))
    assert not issues and deploy["deployable"]
    assert bool(deploy["deployment"]) == (template == "sqd")
    if template == "sqd":
        assert any(issue.code == "CF-E140" for issue in check(plan, registry, purpose="production"))
    changed = plan.model_copy(deep=True)
    sector_id = solvers[0].inputs["sector"]
    next(node for node in changed.nodes if node.id == sector_id).params["n_alpha"] += 1
    assert pack.identity(changed, record, contexts)
    changed = plan.model_copy(deep=True)
    original_energy = next(node for node in changed.nodes if node.id in original and
                           (node.refinement == "fci" or node.params.get("method") == "fci") and
                           node.op in ("ground_energy", "component_energy", "app_component_energy"))
    if original_energy.op == "ground_energy":
        original_energy.refinement, original_energy.binding = "sci", "sci_pyscf"
    else:
        original_energy.params["method"] = "hf"
    assert pack.identity(changed, record, contexts)
    result.params = {**result.params, "tampered": True}
    assert any(issue.code == "CF-E120" for issue in pack.identity(plan, record, contexts))


@pytest.mark.parametrize("template", ["sci", "sqd"])
@pytest.mark.parametrize("app", ["reaction_barrier", "stability_ranking", "pes_scan",
                                 "dimer_binding_scan", "bond_dissociation"])
def test_composite_solvers_have_replayable_certificates(template, app):
    from chemflow.runtime import Executor
    from chemflow.types import EnergyCurve
    from chemflow.units import Q
    from chemflow.verify import check_certificate, prove

    spec = TaskSpec.model_validate(draft(app, example=True)[0])
    record = TaskStore("test").freeze(spec, "test")
    registry = load_registry(pack_names())
    base, _ = pack_for_target(spec.target.kind).base_plan(record)
    plan, identifiers = Orchestrator._candidate_plan(base, [{"template": template, "shots": 1024}])
    computed = Executor(registry).run(plan)
    identifier = next(iter(identifiers.values()))
    result = computed[identifier]
    assert result.diagnostics["relation_status"] == "proved"
    certificate = prove(plan, identifier, registry=registry, computed=computed)
    assert check_certificate(certificate, plan, registry=registry, computed=computed).valid
    if app == "reaction_barrier":
        from chemflow.relations import combine_linear

        node = next(node for node in plan.nodes if node.id == identifier)
        terms = [(node.params["coefficients"][port], computed[source])
                 for port, source in node.inputs.items()]
        assert result.value.to("hartree").value == pytest.approx(sum(
            coefficient * energy.value.to("hartree").value for coefficient, energy in terms))
        assert result.relation == combine_linear([(coefficient, energy.relation)
                                                  for coefficient, energy in terms])
    else:
        updates = ({"values_hartree": [0.] * len(result.values_hartree)}
                   if isinstance(result, EnergyCurve) else {"value": Q(1000, "hartree")})
        altered = {**computed, identifier: result.model_copy(update=updates)}
        assert not check_certificate(certificate, plan, registry=registry, computed=altered).valid
    changed = plan.model_copy(deep=True)
    next(node for node in changed.nodes if node.id == identifier).params["tampered"] = True
    assert not check_certificate(certificate, changed, registry=registry, computed=computed).valid
