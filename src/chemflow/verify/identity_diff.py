"""Bounded structural diagnostics; never used to relax an identity predicate."""

import json
from functools import cache

from chemflow.ids import canonical_json
from chemflow.text import cap_strings

GENERIC_HINT = "Use plan.candidates to obtain identity-valid candidates for this target"
MISSING = "<missing>"


@cache
def identity_registry():
    from chemflow.domains import load_registry, pack_names

    return load_registry(pack_names())


def normalized_params(node):
    operator = identity_registry().operators.get(f"chemflow.{node.op}@1")
    defaults = operator.parameter_defaults if operator else {}
    parameters = json.loads(canonical_json({**defaults, **node.params}))
    if node.op == "protein_structure" and parameters.get("keep") is None:
        parameters["keep"] = []
    return parameters


def signature_tree(plan, identifier):
    nodes = {node.id: node for node in plan.nodes}

    def visit(identifier, visiting):
        if identifier not in nodes:
            return MISSING
        if identifier in visiting:
            return "<cycle>"
        node = nodes[identifier]
        return {"op": node.op, "params": normalized_params(node),
                "inputs": {port: visit(source, visiting | {identifier})
                           for port, source in sorted(node.inputs.items())}}

    return visit(identifier, set())


def _value(value, sensitive=False):
    if value == MISSING:
        return MISSING
    text = value if isinstance(value, str) else canonical_json(value)
    capped = cap_strings(text, limit=0 if sensitive or "pdb_text" in text else 70, head=0)
    if isinstance(capped, dict):
        return f"<{capped['text_chars']} chars sha256:{capped['text_sha256'][:12]}…>"
    return repr(value)


def _tree(value):
    if isinstance(value, tuple) and len(value) in (3, 5):
        keys = ("op", "spin", "inputs") if len(value) == 3 else (
            "op", "params", "binding", "refinement", "inputs")
        return {key: _tree(item) for key, item in zip(keys, value, strict=True)}
    if isinstance(value, dict):
        return {key: _tree(item) for key, item in value.items()}
    return value


def _differences(actual, expected, path="target"):
    if actual == expected:
        return []
    if actual == MISSING and isinstance(expected, dict) and "op" in expected:
        label = f"input port {path.rsplit('.', 1)[-1]}" if ".inputs." in path else path
        return [f"{label}: expected node with op {_value(expected['op'])}, got <missing>"]
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in dict.fromkeys([*expected, *actual]):
            differences.extend(_differences(actual.get(key, MISSING), expected.get(key, MISSING),
                                            f"{path}.{key}"))
        return differences
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return [difference for index in range(max(len(actual), len(expected)))
                for difference in _differences(actual[index] if index < len(actual) else MISSING,
                    expected[index] if index < len(expected) else MISSING, f"{path}[{index}]")]
    if path.endswith(".op"):
        return [f"expected op {_value(expected)} at {path[:-3]}, got {_value(actual)}"]
    label = f"param {path.split('.params.', 1)[1]}" if ".params." in path else path
    sensitive = "pdb_text" in path
    return [f"{label}: expected {_value(expected, sensitive)}, got {_value(actual, sensitive)}"]


def identity_hints(actual, expected):
    differences = [_differences(_tree(actual), _tree(item)) for item in expected]
    closest = min(differences, key=len) if differences else ["Expected target is missing"]
    return [hint[:200] for hint in closest[:1]] + [GENERIC_HINT]


def plan_hints(plan, expected, identifier):
    return identity_hints(signature_tree(plan, identifier),
                          [signature_tree(expected, target) for target in
                           dict.fromkeys(expected.targets + expected.references)])
