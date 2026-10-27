"""Pointwise electronic-energy curves with shared, signed geometry protocols."""

from chemflow import build
from chemflow.diagnostics import Diagnostic
from chemflow.domains.drug import component_matches, component_specs
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.types import EnergyCurve, Provenance
from chemflow.verify.identity_diff import identity_hints, normalized_params, plan_hints

from .paths import resources


def scan_fragment(store, molecule, electronic_model, fragments, counterpoise, index):
    return component_specs(molecule, fragments, electronic_model, counterpoise)[index]


def scan_interaction(store, coefficients, target, **components):
    from chemflow.domains.drug import interaction_energy

    molecule = components["component0"].diagnostics["molecule_witness"]
    actual = {key: target[key] for key in ("electronic_model", "fragments", "counterpoise")}
    actual["complex"] = molecule
    result = interaction_energy(store, coefficients, actual, **components)
    return result.model_copy(update={"diagnostics": {**result.diagnostics,
        "molecule_witness": molecule, "electronic_model_witness": actual["electronic_model"],
        "interaction_target": actual}})


def collect_scan(store, target, method, **points):
    values = (target["coordinate"]["values"] if "coordinate" in target else target["separations"])
    if set(points) != {f"point{index}" for index in range(len(values))}:
        raise ValueError("Scan requires exactly one electronic energy per declared point")
    kind = "total_electronic_energy" if "coordinate" in target else "interaction_energy"
    if any(value.kind != kind for value in points.values()):
        raise ValueError("Scan point energy kind differs from the declared target")
    if not all(point_matches_target(points[f"point{index}"], target, value)
               for index, value in enumerate(values)):
        raise ValueError("Scan point geometry or electronic model differs from the declared target")
    return EnergyCurve(values_hartree=[points[f"point{index}"].value.to("hartree").value
                                      for index in range(len(values))], coordinates=values,
        context={"keys": {"hamiltonian_id": content_hash(target)}},
        provenance=Provenance(impl_id="scan", impl_version="1", backend_kind="cpu_numeric",
                              device="pyscf"), diagnostics={"method": method})


def register(registry):
    specs = (
        ("scan_fragment", {"molecule": "molecule"}, "molecule", scan_fragment,
         {"electronic_model", "fragments", "counterpoise", "index"},
         "Counterpoise ghost-basis component of the transformed scan geometry"),
        ("scan_interaction", {}, "estimate", scan_interaction, {"coefficients", "target"},
         "Complex minus fragment electronic energies at one counterpoise scan point"),
        ("pes_scan", {}, "energy_curve", collect_scan, {"target", "method"},
         "Ordered rigid or constrained-relaxed internal-coordinate PES energy curve"),
        ("interaction_scan", {}, "energy_curve", collect_scan, {"target", "method"},
         "Ordered rigid dimer centroid-separation counterpoise interaction energy curve"),
    )
    for name, inputs, output, adapter, parameters, description in specs:
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
            output=output, parameters=dict.fromkeys(parameters, "JSON"), semantics=description))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
            backend_kind="cpu_numeric", execution_role="classical", params=frozenset(parameters),
            resource_model=resources))
        registry.register(RefinementRule(target=name, template_id=name, relation="approximate"))
    registry.context_checks.append(context_check)


def context_check(node, inputs, context, nodes):
    if node.op in ("geometry_transform", "constrained_optimize", "scan_fragment"):
        context.update(inputs.get("molecule", {}))
        context["geometry_id"] = content_hash((inputs, node.params))
    if node.op in ("pes_scan", "interaction_scan"):
        context.update(kind="energy_curve", unit="hartree",
                       hamiltonian_id=content_hash(node.params.get("target")))
    if node.op == "scan_interaction":
        context.update(kind="interaction_energy", unit="hartree",
                       hamiltonian_id=content_hash(inputs))
    return []


def validate_parameters(operation, parameters):
    from chemflow.agent.task import InteractionScanTarget, PESScanTarget

    if "target" in parameters:
        target = parameters["target"]
        if not isinstance(target, dict):
            raise ValueError("Scan target must be an object")
        schema = PESScanTarget if operation == "pes_scan" else InteractionScanTarget
        schema.model_validate({**target, "kind": "pes_scan" if operation == "pes_scan"
                               else "interaction_scan"})
    if "method" in parameters and parameters["method"] not in ("hf", "mp2", "ccsd", "ccsd_t", "fci"):
        raise ValueError("Unknown scan evaluation method")
    if operation == "scan_fragment":
        fragments = parameters.get("fragments")
        if (not isinstance(fragments, list) or len(fragments) != 2 or
                any(not isinstance(fragment, list) or not fragment or
                    any(type(index) is not int or index < 0 for index in fragment)
                    for fragment in fragments)):
            raise ValueError("Scan fragments require two nonempty atom-index lists")
        flattened = [index for fragment in fragments for index in fragment]
        if len(set(flattened)) != len(flattened):
            raise ValueError("Scan fragment indices must not overlap")
        if type(parameters.get("counterpoise")) is not bool:
            raise ValueError("counterpoise must be a boolean")
        if type(parameters.get("index")) is not int or not 0 <= parameters["index"] <= 2:
            raise ValueError("Component index must be 0, 1 or 2")


def base_plan(record, active=None):
    if active is not None:
        raise ValueError("Scan references use the full orbital space")
    target = record.spec.target
    payload = target.model_dump(exclude={"kind", "operator"})
    with build.program("scan-controls") as builder:
        geometry = builder.molecule(**(target.molecule if target.kind == "pes_scan"
                                       else target.complex))
        model = builder.electronic_model(**target.electronic_model)
        points = []
        values = target.coordinate.values if target.kind == "pes_scan" else target.separations
        for value in values:
            if target.kind == "pes_scan":
                moved = builder.node("geometry_transform", molecule=geometry,
                    operation=f"set_{target.coordinate.type}", atoms=target.coordinate.atoms,
                    value=value)
                if target.mode == "relaxed":
                    moved = builder.node("constrained_optimize", molecule=moved, model=model,
                        level=target.relax_level.model_dump(), constraints=[{
                            "type": target.coordinate.type, "atoms": target.coordinate.atoms,
                            "value": value}])
            else:
                moved = builder.node("geometry_transform", molecule=geometry,
                    operation="translate_fragment", atoms=[], fragment=target.fragments[1], value=value)
            points.append(moved)
        controls = []
        for method in ("hf", "mp2", "ccsd", "ccsd_t", "fci"):
            energies = {}
            for index, point in enumerate(points):
                if target.kind == "pes_scan":
                    energy = builder.node("component_energy", molecule=point,
                        electronic_model=target.electronic_model, method=method)
                else:
                    components = {}
                    for component in range(len(target.fragments) + 1):
                        fragment = builder.node("scan_fragment", molecule=point,
                            electronic_model=target.electronic_model, fragments=target.fragments,
                            counterpoise=target.counterpoise, index=component)
                        components[f"component{component}"] = builder.node("component_energy",
                            molecule=fragment, electronic_model=target.electronic_model, method=method)
                    energy = builder.node("scan_interaction", **components, target=payload,
                        coefficients={port: 1 if port == "component0" else -1 for port in components})
                energies[f"point{index}"] = energy
            curve = builder.node(target.kind, **energies, target=payload, method=method)
            if method == "fci":
                builder.ir.references = [curve.id]
            else:
                controls.append(curve.id)
        builder.ir.targets = [*controls, *builder.ir.references]
        return builder.to_ir(), controls


def point_matches_target(point, target, value):
    import numpy as np

    from .paths import coordinates, geometry_transform

    molecule = point.diagnostics.get("molecule_witness")
    model = point.diagnostics.get("electronic_model_witness")
    if molecule is None or model != target["electronic_model"]:
        return False
    if "coordinate" in target:
        coordinate = target["coordinate"]
        source = {"charge": 0, "spin": 0, "unit": "angstrom", **target["molecule"]}
        expected = geometry_transform(None, source, f"set_{coordinate['type']}",
                                      coordinate["atoms"], value)
        matching_identity = component_matches(point, molecule, model)
        if target["mode"] == "relaxed":
            from chemflow.types import ElectronicModel

            protocol = molecule.get("relaxation", {})
            return (protocol == {"initial_molecule": expected, "level": target["relax_level"],
                "constraints": [{"type": coordinate["type"], "atoms": coordinate["atoms"],
                                 "value": value}], "electronic_model":
                ElectronicModel.model_validate(model).model_dump(mode="json")} and
                    matching_identity)
    else:
        expected = geometry_transform(None, target["complex"], "translate_fragment", [], value,
                                      target["fragments"][1])
        actual = point.diagnostics.get("interaction_target", {})
        if actual != {"complex": molecule, "electronic_model": model,
                      "fragments": target["fragments"], "counterpoise": target["counterpoise"]}:
            return False
        matching_identity = point.context.keys.get("hamiltonian_id") == content_hash(actual)
    wanted_symbols, wanted_positions = coordinates(expected)
    symbols, positions = coordinates(molecule)
    return (symbols == wanted_symbols and np.allclose(positions, wanted_positions, atol=1e-9, rtol=0)
            and all(molecule.get(key, 0) == expected.get(key, 0) for key in ("charge", "spin"))
            and matching_identity)


def identity(plan, record, contexts):
    expected, _ = base_plan(record)
    return plan_identity(plan, expected)


def signature(nodes, identifier, candidate=False):
    node = nodes[identifier]
    if candidate and node.op in ("component_energy", "app_component_energy"):
        from .component_candidates import ground_reference

        additions, node = ground_reference(nodes, node)
        nodes = {**nodes, **{item.id: item for item in additions}, node.id: node}
    if candidate and node.op == "ground_energy":
        return (node.op, node.params.get("s", node.params.get("total_spin")),
                {port: signature(nodes, node.inputs[port])
                 for port in ("hamiltonian", "sector")})
    return (node.op, normalized_params(node), node.binding or node.op, node.refinement,
            {port: signature(nodes, source, candidate) for port, source in node.inputs.items()})


def plan_identity(plan, expected):
    wanted = {node.id: node for node in expected.nodes}
    actual = {node.id: node for node in plan.nodes}

    allowed = [signature(wanted, identifier, True) for identifier in expected.targets]
    issues = []
    for identifier in set(plan.targets + plan.references):
        hints = None
        try:
            candidate_signature = signature(actual, identifier, True)
            valid = candidate_signature in allowed
            hints = identity_hints(candidate_signature, allowed)
            if identifier in plan.references:
                reference = signature(wanted, expected.references[0])
                actual_reference = signature(actual, identifier)
                if actual_reference != reference:
                    valid = False
                    hints = identity_hints(actual_reference, [reference])
        except (KeyError, RecursionError):
            valid = False
        if not valid:
            issues.append(Diagnostic(code="CF-E120", node=identifier,
                                     message="Path geometry, evaluation or reference identity differs",
                                     fix_hints=hints or plan_hints(plan, expected, identifier)))
    return issues
