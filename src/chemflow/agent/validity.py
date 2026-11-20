"""Validity maps of screened methods over the signed validity domain.

A method is evaluated against the exact reference at sampled points of each signed axis
(geometry scale, Hubbard U, an internal coordinate, a separation, pH). All points and
methods of one sampling level compile into one QCL plan, so the scheduler runs them side
by side; intervals whose end points disagree are bisected, one plan per level, down to the
signed resolution. Regions interpolate their bounding samples: they assume no unsampled
status reversal and are not interval proofs.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np

from chemflow.agent import screen as screens
from chemflow.agent.task import task_axis_point
from chemflow.units import Q

ASSUMPTION = ("Regions interpolate bounding samples, assuming no unsampled status reversal. "
              "They are not interval proofs.")


def _scaled(molecule: dict[str, Any], factor: float) -> dict[str, Any]:
    """Coordinates scaled about the first atom, in the molecule's own unit."""
    from pyscf.gto import format_atom

    # unit="Bohr" tells PySCF not to convert, so the numbers keep the molecule's unit.
    atoms = format_atom(molecule["atoms"], unit="Bohr")
    origin = np.asarray(atoms[0][1])
    return {**molecule, "atoms": "; ".join(
        symbol + " " + " ".join(repr(float(value)) for value in
                                origin + factor * (np.asarray(position) - origin))
        for symbol, position in atoms)}


def instantiate(task_spec: Any, axis: str, value: float) -> Any:
    """The signed task at another point of a validity axis (the same gates and model)."""
    target = task_spec.target.model_dump(mode="json")
    if axis == "geometry_scale":
        for key in ("molecule", "complex", "product"):
            if isinstance(target.get(key), dict) and "atoms" in target[key]:
                target[key] = _scaled(target[key], value)
        for collection in ("species", "items"):
            for item in target.get(collection) or []:
                item["molecule"] = _scaled(item["molecule"], value)
    elif axis == "hubbard_u":
        target["lattice"]["U"] = value
    elif axis == "impurity_u":
        target["model"]["impurity_model"]["U"] = value
    elif axis == "internal_coordinate":
        values = target["coordinate"]["values"]
        offset = value - values[0]
        target["coordinate"]["values"] = [point + offset for point in values]
    elif axis == "separation":
        offset = value - target["separations"][0]
        target["separations"] = [point + offset for point in target["separations"]]
    elif axis == "binding_ph":
        target["ph"] = value
    else:
        raise ValueError(f"Unsupported validity axis {axis}")
    return task_spec.model_copy(update={"target": type(task_spec.target).model_validate(
        target), "validity": ()})


def bar(task_spec: Any) -> float:
    gate = next((item for item in task_spec.gates if item.kind in (
        "abs_error_vs_exact", "l1_error_vs_exact")), None)
    if gate is None:
        raise ValueError("Validity requires a signed accuracy bar")
    unit = ("dimensionless" if gate.kind == "l1_error_vs_exact" or
            task_spec.target.kind == "pairing_correlation" else "hartree")
    return Q(gate.threshold, gate.unit).to(unit).value


def map_methods(task_spec: Any, cluster: Any, entries: list[Any], *, active: Any,
                execute: Any) -> dict[str, dict[str, Any]]:
    """Validity maps for screen entries; ``execute(built)`` runs one batched plan.

    Model-defined targets are equivalent to their model at every point of the domain, so
    their outputs are reported ``inside_domain`` without sampling.
    """
    import qcl

    if task_spec.target.kind in screens.MODEL_DEFINED:
        return {entry.name: {"node": entry.name, "method": entry.method,
                             "task_point": {"status": "inside_domain"}, "axes": [],
                             "assumption": "Model-defined target: the plan computes the "
                                           "signed model at every point of the domain."}
                for entry in entries}
    limit = bar(task_spec)
    candidates = tuple(entry for entry in entries if entry.role == "candidate")
    names = {entry.name for entry in entries}
    maps: dict[str, dict[str, Any]] = {entry.name: {"node": entry.name, "method": entry.method,
                                                    "bar": limit, "axes": [],
                                                    "assumption": ASSUMPTION}
                                       for entry in entries}
    for axis in task_spec.validity:
        samples: dict[str, dict[float, dict]] = {name: {} for name in names}

        def evaluate(values: list[float], axis: Any = axis,
                     samples: dict = samples) -> None:
            values = [value for value in dict.fromkeys(values)
                      if any(value not in samples[name] for name in names)]
            if not values:
                return
            instances = []
            for value in values:
                try:
                    point = instantiate(task_spec, axis.axis, value)
                    built = screens.build(point, cluster, active=active, proposed=candidates,
                                          only=names | {"fci", "ed"})
                except (ValueError, TypeError) as error:
                    for name in names:
                        samples[name][value] = {"value": value, "status": "unknown",
                                                "reason": str(error)}
                    continue
                instances.append((value, point, built))
            if not instances:
                return

            @qcl.workflow
            def family(*_: Any) -> Any:
                outputs = {}
                for index, (_value, _point, built) in enumerate(instances):
                    values_at = built.workflow.function(*built.args)
                    for name, value in values_at.items():
                        outputs[f"{name}@{index}"] = value
                return outputs

            combined = screens.Screen(family, (), tuple(
                screens.Entry(f"{entry.name}@{index}", entry.role, entry.method, entry.params)
                for index, (_value, _point, built) in enumerate(instances)
                for entry in built.entries), None,
                sim_shots=sum(built.sim_shots for _value, _point, built in instances))
            results = execute(combined)
            for index, (value, point, built) in enumerate(instances):
                reference = results.get(f"{built.reference}@{index}")
                for name in names:
                    result = results.get(f"{name}@{index}")
                    if result is None or reference is None:
                        samples[name][value] = {"value": value, "status": "unknown",
                                                "reason": "method or reference failed"}
                        continue
                    error = screens.error(result, reference, screens.unit_for(point))
                    samples[name][value] = {
                        "value": value, "error": error if math.isfinite(error) else None,
                        "status": ("unknown" if not math.isfinite(error) else
                                   "invalid" if error > limit else "sampled_valid")}

        task_value = task_axis_point(task_spec.target, axis.axis)
        evaluate([*map(float, np.linspace(*axis.range, axis.points)), float(task_value)])
        while True:
            pending = []
            for name in names:
                known = sorted(samples[name])
                for low, high in pairwise(known):
                    if (samples[name][low]["status"] != samples[name][high]["status"] and
                            high - low > axis.resolution):
                        pending.append((low + high) / 2)
            if not pending:
                break
            evaluate(sorted(set(pending)))
        for name in names:
            points = [samples[name][value] for value in sorted(samples[name])]
            regions: list[dict[str, Any]] = []
            for left, right in pairwise(points):
                status = left["status"] if left["status"] == right["status"] else "unknown"
                if regions and regions[-1]["status"] == status:
                    regions[-1]["hi"] = right["value"]
                else:
                    regions.append({"lo": left["value"], "hi": right["value"], "status": status})
            maps[name]["axes"].append({
                **axis.model_dump(mode="json"), "points": points, "regions": regions,
                "counterexamples": [{"value": point["value"], "error": point.get("error")}
                                    for point in points if point["status"] == "invalid"]})
            point = samples[name].get(float(task_value), {"status": "unknown"})
            previous = maps[name].get("task_point")
            if previous is None or previous["status"] != "invalid":
                maps[name]["task_point"] = point
    return maps


def section(maps: list[dict[str, Any]]) -> list[str]:
    """The report's validity section (the legacy format)."""
    if not maps:
        return []
    lines = ["", "## Validity", ""]
    for mapped in maps:
        bar_text = f", accuracy bar {mapped['bar']:.4g}" if "bar" in mapped else ""
        lines.append(f"{mapped['method']} ({mapped['node']}){bar_text}: task point "
                     f"{mapped['task_point']['status']}.")
        lines.append(mapped["assumption"])
        for axis in mapped["axes"]:
            lines.append(f"Axis {axis['axis']}:")
            lines.extend(f"- {region['status']}: [{region['lo']:.6g}, {region['hi']:.6g}]"
                         for region in axis["regions"])
            lines.append("Counterexamples: " + (", ".join(
                f"{item['value']:.6g}" for item in axis["counterexamples"]) or "none"))
    return lines
