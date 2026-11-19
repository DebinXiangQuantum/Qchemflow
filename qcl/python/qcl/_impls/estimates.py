"""Arithmetic, unit conversion and ranking of Estimate values."""

import math

from qcl._impls import implements
from qcl._impls._common import _linear_uncertainty


@implements("estimate.add", "estimate.sub")
def _op_estimate_add(ctx, node, identifier, operation, inputs):
    values = ctx.values
    left, right = inputs
    sign = 1 if operation == "estimate.add" else -1
    components, variance = _linear_uncertainty(
        (1.0, left, node["inputs"][0]),
        (float(sign), right, node["inputs"][1]))
    values[identifier] = {
        "value": left["value"] + sign * right["value"],
        "unit": left["unit"],
        "variance": variance, "_uncertainty_components": components,
        "provenance": {"question": {"property": "estimate_expression"},
                       "method": {"family": "host_arithmetic"},
                       "implementation": operation + "/qcl-host",
                       "device": "host", "operands": [left["provenance"],
                                                      right["provenance"]]},
    }


@implements("estimate.scale")
def _op_estimate_scale(ctx, node, identifier, operation, inputs):
    values = ctx.values
    original = inputs[0]
    factor = node["params"]["factor"]
    components, variance = _linear_uncertainty(
        (factor, original, node["inputs"][0]))
    values[identifier] = {
        "value": factor * original["value"], "unit": original["unit"],
        "variance": variance, "_uncertainty_components": components,
        "provenance": {"question": original["provenance"]["question"],
                       "method": {"family": "scalar_arithmetic",
                                  "factor": factor},
                       "implementation": "estimate.scale/qcl-host",
                       "device": "host", "operand": original["provenance"]},
    }


@implements("estimate.mean")
def _op_estimate_mean(ctx, node, identifier, operation, inputs):
    values = ctx.values
    estimates = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], list) else inputs
    if not estimates:
        raise ValueError("Mean requires at least one estimate")
    units = {item["unit"] for item in estimates}
    if len(units) != 1:
        raise ValueError("Mean estimate units differ")
    count = len(estimates)
    source_ids = (node["inputs"] if len(node["inputs"]) == count else
                  [f"{identifier}:{index}" for index in range(count)])
    components, variance = _linear_uncertainty(*(
        (1 / count, item, source)
        for item, source in zip(estimates, source_ids, strict=True)))
    questions = [item["provenance"]["question"] for item in estimates]
    values[identifier] = {
        "value": sum(item["value"] for item in estimates) / count,
        "unit": estimates[0]["unit"], "variance": variance,
        "_uncertainty_components": components,
        "provenance": {"question": questions[0] if all(
            item == questions[0] for item in questions) else
            {"property": "mean_estimate", "inputs": questions},
            "method": {"family": "host_mean", "count": count},
            "implementation": "estimate.mean/qcl-host", "device": "host",
            "input_implementations": [item["provenance"]["implementation"]
                                      for item in estimates]},
    }


@implements("estimate.compare")
def _op_estimate_compare(ctx, node, identifier, operation, inputs):
    values = ctx.values

    left, right = inputs
    if left["unit"] != right["unit"]:
        raise ValueError("Comparison estimate units differ")
    _components, variance = _linear_uncertainty(
        (1.0, left, node["inputs"][0]),
        (-1.0, right, node["inputs"][1]))
    sigma = math.sqrt(max(0.0, variance))
    difference = left["value"] - right["value"]
    threshold = node["params"]["threshold_sigma"] * sigma
    relation = ("less" if difference < -threshold else
                "greater" if difference > threshold else "unresolved")
    values[identifier] = {
        "relation": relation, "difference": difference,
        "combined_sigma": sigma, "unit": left["unit"],
        "provenance": {"question": {"property": "estimate_comparison"},
                       "method": {"family": "uncertainty_aware_comparison",
                                  "threshold_sigma": node["params"]["threshold_sigma"]},
                       "implementation": "estimate.compare/qcl-host",
                       "device": "host", "operands": [left["provenance"],
                                                       right["provenance"]]},
    }


@implements("estimate.in_units")
def _op_estimate_in_units(ctx, node, identifier, operation, inputs):
    values = ctx.values
    from chemflow.units import Q

    original = inputs[0]
    converted = Q(original["value"], original["unit"]).to(node["params"]["unit"])
    factor = Q(1, original["unit"]).to(node["params"]["unit"]).value
    components, variance = _linear_uncertainty(
        (factor, original, node["inputs"][0]))
    values[identifier] = {
        "value": converted.value, "unit": node["params"]["unit"],
        "variance": variance, "_uncertainty_components": components,
        "provenance": {**original["provenance"],
                       "unit_conversion": {"from": original["unit"],
                                           "to": node["params"]["unit"]}},
    }


@implements("rank")
def _op_rank(ctx, node, identifier, operation, inputs):
    values = ctx.values

    if len(inputs) == 1 and isinstance(inputs[0], list):
        inputs = inputs[0]
    if not inputs:
        raise ValueError("Ranking requires at least one computed estimate")
    source_ids = (node["inputs"] if len(node["inputs"]) == len(inputs) else
                  [f"{identifier}:{index}" for index in range(len(inputs))])
    order = sorted(range(len(inputs)), key=lambda index: inputs[index]["value"])
    unresolved = [
        [left, right] for left in range(len(inputs))
        for right in range(left + 1, len(inputs))
        if abs(inputs[left]["value"] - inputs[right]["value"]) <=
        2 * math.sqrt(max(0.0, _linear_uncertainty(
            (1.0, inputs[left], source_ids[left]),
            (-1.0, inputs[right], source_ids[right]))[1]))
    ]
    values[identifier] = {
        "order": order, "unresolved": unresolved, "values": inputs,
        "provenance": {"question": {"property": "energy_ranking"},
                       "method": {"family": "uncertainty_aware_ranking",
                                  "threshold_sigma": 2},
                       "implementation": "rank/qcl-host", "device": "host",
                       "input_implementations": [item["provenance"]["implementation"]
                                                 for item in inputs]},
    }
