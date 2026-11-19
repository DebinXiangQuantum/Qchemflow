import random
from types import SimpleNamespace

import pytest

from chemflow.check.static import check, infer_relations
from chemflow.domains import load_registry, pack_for_target, pack_names
from chemflow.eval.runner import load_suite
from chemflow.ir.schema import Node, Program
from chemflow.verify.semantics import facts


@pytest.fixture(scope="module")
def registry():
    return load_registry(pack_names())


@pytest.mark.parametrize("method", [None, {}, [], 1, True, "unknown"])
def test_component_method_is_diagnosed(registry, method):
    node = Node(id="energy", op="component_energy", params={
        "molecule": {"atoms": "H 0 0 0; H 0 0 .74"},
        "electronic_model": {"basis": "sto-3g"}, "method": method})
    program = Program(name="bad-method", nodes=[node])
    assert any("method" in issue.message for issue in check(program, registry))
    assert infer_relations(program, registry).get(node.id) != "equivalent"
    assert not facts(node, registry, {node.id: {}})["known_denotation"]


def test_random_json_parameters_never_crash(registry):
    randomizer = random.Random(135136)

    def json_value(depth=0):
        values = [None, True, False, 0, -1, 1.25, "", "invalid"]
        if depth < 2:
            values.extend([[json_value(depth + 1)], {"invalid": json_value(depth + 1)}])
        return randomizer.choice(values)

    for implementation in registry.implementations.values():
        for _ in range(20):
            node = Node(id="subject", op=implementation.target, binding=implementation.id,
                        refinement=implementation.implements,
                        params={name: json_value() for name in sorted(implementation.params)})
            program = Program(name="fuzz", nodes=[node], targets=[node.id])
            assert isinstance(check(program, registry), list), node
            assert isinstance(infer_relations(program, registry), dict), node
            assert isinstance(facts(node, registry, {node.id: {}}), dict), node


@pytest.mark.parametrize("coefficients", [None, [], {"left": None}, {"left": {}}])
def test_linear_coefficients_never_crash(registry, coefficients):
    source = Node(id="source", op="component_energy", params={
        "molecule": {}, "electronic_model": {}, "method": "hf"})
    node = Node(id="sum", op="reaction_energy", inputs={"left": source.id},
                params={"coefficients": coefficients})
    program = Program(name="linear", nodes=[source, node])
    assert any("coefficients" in issue.message for issue in check(program, registry))
    infer_relations(program, registry)


def test_thermal_none_is_diagnosed(registry):
    node = Node(id="thermal", op="thermochemistry", params={"temperature_K": None})
    assert any("temperature_K" in issue.message for issue in
               check(Program(name="thermal", nodes=[node]), registry))


def test_json_mutations_in_connected_domain_plans():
    seen = set()
    randomizer = random.Random(136)
    for task in load_suite("full").tasks:
        kind = task.spec.target.kind
        if kind in seen:
            continue
        seen.add(kind)
        pack = pack_for_target(kind)
        registry = load_registry(["molecular", pack.name])
        plan, _ = pack.base_plan(SimpleNamespace(spec=task.spec, content_hash="fuzz"))
        for index, node in enumerate(plan.nodes):
            for name in node.params:
                values = [None, {}, [], {"bad": []}, [None], True, "invalid", -1, .5]
                randomizer.shuffle(values)
                for value in values:
                    mutated = plan.model_copy(deep=True)
                    mutated.nodes[index].params[name] = value
                    assert isinstance(check(mutated, registry), list), (kind, name, value)
                    assert isinstance(infer_relations(mutated, registry), dict)
