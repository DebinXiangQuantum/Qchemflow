from __future__ import annotations

import math
import re

from chemflow.agent.memory import ResultsDB
from chemflow.agent.task import TaskSpec
from chemflow.ir.schema import Program
from chemflow.registry import Registry
from chemflow.relations import satisfies
from chemflow.runtime.budget import RESOURCE_DIMENSIONS, resource_defaults
from chemflow.types import EnergyCurve, SpectralEstimate
from chemflow.units import Q


class ReportLintError(ValueError):
    pass


ENERGY_NUMBER = re.compile(
    r"(?<![\w.])([+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s*"
    r"(hartree|mHa|Ha|Eh|eV|kcal/mol|kJ/mol|cm\^?-1)\b", re.IGNORECASE)


def _selection_work(node_id: str, plan: Program | None) -> int | None:
    """Reproducible asymptotic solver-work proxy, not a resource estimate or speedup claim."""
    if plan is None:
        return None
    nodes = {node.id: node for node in plan.nodes}
    node = nodes[node_id]
    if node.op != "ground_energy":
        visited = set()
        pending = list(node.inputs.values())
        work = []
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            source = nodes[dependency]
            if source.op == "ground_energy":
                estimate = _selection_work(dependency, plan)
                if estimate is None:
                    return None
                work.append(estimate)
            pending.extend(source.inputs.values())
        return sum(work) if work else None
    sector = nodes.get(node.inputs.get("sector"))
    if sector is None:
        return None
    size = sector.params.get("n_orb")
    if type(size) is not int or size < 1:
        return None
    powers = {"hf": 3, "mp2": 5, "ccsd": 6, "ccsd_t": 7}
    if node.refinement in powers:
        return size ** powers[node.refinement]
    if node.refinement in ("sci", "sqd", "fci"):
        counts = [sector.params.get(key) for key in ("n_alpha", "n_beta")]
        if any(type(count) is not int or not 0 <= count <= size for count in counts):
            return None
        dimension = math.prod(math.comb(size, count) for count in counts)
        return dimension ** 2
    return None


def _energy_values(text):
    units = {"ha": "hartree", "eh": "hartree", "hartree": "hartree", "ev": "eV",
             "kcal/mol": "kcal/mol", "kj/mol": "kJ/mol", "cm-1": "cm^-1",
             "cm^-1": "cm^-1", "mha": "hartree"}
    return [Q(float(number.replace("−", "-")) * (0.001 if unit.lower() == "mha" else 1),
              units[unit.lower()]).to("hartree").value
            for number, unit in ENERGY_NUMBER.findall(text)]


def result_method(node_id: str, plan: Program | None) -> str | None:
    if plan is None:
        return None
    nodes = {node.id: node for node in plan.nodes}
    node = nodes[node_id]
    if node.refinement:
        return node.refinement
    if node.op in ("component_energy", "app_component_energy", "pes_scan", "interaction_scan"):
        return node.params.get("method")
    methods = {nodes[source].refinement or nodes[source].params.get("method")
               for source in node.inputs.values()
               if nodes[source].op in ("ground_energy", "component_energy", "app_component_energy")}
    return methods.pop() if len(methods) == 1 else None


def result_summary(node_id: str, computed: dict, gates: dict, reference: str,
                   spec: TaskSpec, plan: Program | None = None) -> dict:
    from chemflow.verify.integration import relation_fields

    visited = {node_id}
    if plan is not None:
        nodes = {node.id: node for node in plan.nodes}
        pending = list(nodes[node_id].inputs.values())
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            pending.extend(nodes[dependency].inputs.values())
    budget_overruns = [item for item in computed.get("budget_overruns", [])
                       if item["node"] in visited]
    if node_id in computed.get("refused", {}):
        return {"id": node_id, "method": result_method(node_id, plan),
                "relation": "unknown", "evidence": False,
                "gate_passed": False, "cost": {}, "error": None,
                "reason": computed["refused"][node_id], "budget_overruns": budget_overruns}
    estimate = computed[node_id]
    cost = computed["costs"][node_id].model_dump(mode="json")
    amounts = {dimension: [] for dimension in RESOURCE_DIMENSIONS}
    for identifier in visited:
        recorded = computed["costs"][identifier]
        value = computed.get(identifier)
        kind = recorded.diagnostics.get("backend_kind")
        if kind is None:
            kind = getattr(getattr(value, "provenance", None), "backend_kind",
                           getattr(value, "source", "cpu_numeric"))
        resources = resource_defaults(recorded.model_dump(exclude_unset=True), kind)
        for dimension in RESOURCE_DIMENSIONS:
            amounts[dimension].append(resources.get(dimension))
    for dimension, values in amounts.items():
        cost[dimension] = None if None in values else sum(values)
    spectral = isinstance(estimate, SpectralEstimate)
    curve = isinstance(estimate, EnergyCurve)
    comparison_unit = ("dimensionless" if spectral or curve else "hartree"
                       if estimate.value.dimension == "energy" else estimate.value.unit)
    error = (abs(estimate.value.to(comparison_unit).value -
                 computed[reference].value.to(comparison_unit).value)
             if reference in computed and not spectral and not curve and
             not isinstance(computed[reference], EnergyCurve) and
             estimate.kind == computed[reference].kind else float("inf"))
    if curve and isinstance(computed.get(reference), EnergyCurve):
        from chemflow.types.curves import curve_error

        error = curve_error(estimate, computed[reference])
    threshold = next((gate for gate in spec.gates if gate.kind == "abs_error_vs_exact"), None)
    verdicts = [gate for gate in computed.get("gates", []) if gate.node == node_id]
    if spectral:
        error = next((gate.measured.value for gate in verdicts if gate.measured is not None),
                     float("inf"))
    if "values_hartree" in estimate.diagnostics and reference in computed:
        values = estimate.diagnostics["values_hartree"]
        expected = computed[reference].diagnostics.get("values_hartree", [])
        error = max((abs(left - right) for left, right in zip(values, expected)),
                    default=float("inf")) if len(values) == len(expected) else float("inf")
    passed = (all(gate.passed for gate in verdicts) if verdicts else
              gates[node_id].passed if node_id in gates else
              threshold is not None and math.isfinite(error) and
              error <= Q(threshold.threshold, threshold.unit).to(comparison_unit).value)
    passed = passed and (getattr(spec.target, "target_spin", None) is None or
                         (estimate.context.keys.get("target_spin") is not None and
                          float(estimate.context.keys["target_spin"]) == spec.target.target_spin))
    passed = passed and all(estimate.diagnostics.get("relation_status") == "proved" and
                            satisfies(estimate.relation, gate.relation)
                            for gate in spec.gates if gate.kind == "relation_at_least")
    return {"id": node_id, "method": result_method(node_id, plan),
            **({"validity_status": estimate.diagnostics.get("validity", {}).get(
                "task_point", {}).get("status", "unknown"),
                "validity": estimate.diagnostics.get("validity")} if spec.validity else {}),
            **relation_fields(estimate),
            **({"validity_status": estimate.diagnostics["validity_status"]}
               if "validity_status" in estimate.diagnostics else {}),
            "relation": estimate.relation.value,
            "evidence": bool(estimate.provenance) if spectral or curve else bool(estimate.evidence),
            "gate_passed": passed, "cost": cost, "error": error,
            "selection_work": _selection_work(node_id, plan),
            "budget_overruns": budget_overruns}


def result_headline(node_id: str, computed: dict, gates: dict, reference: str,
                    spec: TaskSpec, plan: Program | None, registry: Registry) -> dict:
    from chemflow.check.deployment import deployment

    estimate = computed[node_id]
    deployment_plan = plan
    if plan is not None and getattr(estimate, "kind", None) in (
            "frontier_orbitals", "solvation_free_energy"):
        deployment_plan = plan.model_copy(update={
            "references": [identifier for identifier in plan.references if identifier != node_id]})
    item = result_summary(node_id, computed, gates, reference, spec, plan)
    kinds = []
    if plan is not None:
        nodes = {node.id: node for node in plan.nodes}
        visited = set()

        def visit(identifier: str) -> None:
            if identifier in visited:
                return
            visited.add(identifier)
            for dependency in nodes[identifier].inputs.values():
                visit(dependency)
            binding = nodes[identifier].binding or nodes[identifier].op
            kind = registry.implementations[binding].backend_kind.value
            if not kinds or kind != kinds[-1]:
                kinds.append(kind)

        visit(node_id)
    else:
        kinds.append(estimate.provenance.backend_kind.value)
    failures = [gate.reason for gate in computed.get("gates", [])
                if gate.node == node_id and not gate.passed]
    if any(estimate.diagnostics.get("relation_status") != "proved" or
           not satisfies(estimate.relation, gate.relation)
           for gate in spec.gates if gate.kind == "relation_at_least"):
        failures.append("Relation does not match required relation")
    spectral = isinstance(estimate, SpectralEstimate)
    curve = isinstance(estimate, EnergyCurve)
    evidence = ("diagnostic_only" if spectral or curve else
                ", ".join(evidence.status.value for evidence in estimate.evidence))
    design = {key: estimate.diagnostics[key] for key in
              ("design", "design_status", "design_evidence") if key in estimate.diagnostics}
    if design:
        evidence += f"; experimental {design['design_evidence']}"
    label = None
    rate = {}
    if (spec.target.kind == "reaction_barrier" and spec.target.rate_constant and
            estimate.kind == "reaction_gibbs_energy"):
        from chemflow.domains.paths import rate_constant

        rate = {"rate_constant": rate_constant(None, estimate.value.to("hartree").value,
                spec.target.thermo.temperature_K)}
    if not spectral and estimate.kind == "metabolic_sites":
        evidence = "empirical; diagnostic_only; uncalibrated"
        label = "model-defined C-H reactivity score; not enzyme specificity or metabolic kinetics"
    if not spectral and estimate.kind in ("frontier_orbitals", "solvation_free_energy"):
        model = estimate.diagnostics.get("declared_model", {})
        level = model.get("level", {})
        method = level.get("functional") or level.get("method", "SCF")
        if estimate.kind == "frontier_orbitals":
            label = (f"model-defined {method} {model.get('quantity')} "
                     f"({model.get('spin')}); not a physical IP/EA")
        else:
            solvent = model.get("solvent", {})
            label = f"model-defined {method}/{solvent.get('model')} solvation"
            if solvent.get("model") == "pcm":
                label += f"; IEF-PCM eps={solvent.get('eps')}; electrostatic only"
    if not spectral and estimate.kind in ("reaction_enthalpy", "reaction_gibbs_energy"):
        temperature = float(estimate.context.keys["temperature_K"])
        pressure = float(estimate.context.keys["pressure_Pa"])
        label = (f"ΔH({temperature:g} K)" if estimate.kind == "reaction_enthalpy" else
                 f"ΔG({temperature:g} K, {pressure / 101325:g} atm)")
    if rate:
        label = f"{label}; model Eyring k={rate['rate_constant']['k_per_s']:.6g} s^-1"
    return {"node": node_id, **rate,
            **({"validity_status": item["validity_status"],
                "empirical": estimate.diagnostics["model_card"]}
               if "model_card" in estimate.diagnostics else {}),
            **{key: estimate.diagnostics[key] for key in
               ("values_hartree", "ranking", "certified_pairs", "error_bounds_hartree",
                "certification_basis", "sites", "top_site", "scoring_formula",
                "experimental_relation") if key in estimate.diagnostics},
            **(deployment(deployment_plan, registry, targets=[node_id])[0] if plan else {}),
            **({"quantity_label": label} if label else {}),
            **({"validity": item["validity"], "validity_status": item["validity_status"]}
               if spec.validity else {}),
            "relation_status": item["relation_status"],
            "relation_certificate": item["relation_certificate"],
            **({"values_hartree": estimate.values_hartree,
                "coordinates": estimate.coordinates} if curve else {}),
            "value": (f"spectral density ({len(estimate.omega)} points)" if spectral else
                      f"energy curve ({len(estimate.values_hartree)} points)" if curve else
                      f"{estimate.value.value:.9f} {estimate.value.unit}"
                      if estimate.value.dimension == "energy" else
                      f"{estimate.value.value:.17g} {estimate.value.unit}"),
            "relation": estimate.relation.value,
            "evidence": evidence, **design,
            "gate_verdict": "PASS" if item["gate_passed"] else
            f"FAIL ({'; '.join(failures) or 'threshold exceeded'})",
            "cost": item["cost"], "backend_kind": "→".join(kinds)}


def resolve_headlines(headlines: list[dict], results: ResultsDB, record_hash: str) -> list[dict]:
    if not isinstance(headlines, list):
        raise ReportLintError("Headlines must be a list of recorded result references")
    resolved = []
    for supplied in headlines:
        if not isinstance(supplied, dict) or not isinstance(supplied.get("result_id"), str):
            raise ReportLintError("Every headline requires a valid result_id from run.results")
        record = results.get(supplied["result_id"])
        stored = results.headline(supplied["result_id"])
        if record is None or stored is None or record["record_hash"] != record_hash:
            raise ReportLintError("Headline result_id is not a recorded result of this task")
        headline = stored["headline"]
        if any(key not in headline or value != headline[key] for key, value in supplied.items()):
            raise ReportLintError("Headline fields differ from the recorded result")
        resolved.append(headline)
    return resolved


def lint_recorded_report(markdown: str, headlines: list[dict], results: ResultsDB,
                         record_hash: str) -> list[dict]:
    resolved = resolve_headlines(headlines, results, record_hash)
    claims = set()
    for headline in resolved:
        verdict = results.headline(headline["result_id"])["critic"]
        if verdict:
            claims.update(verdict.get("supported_claims", []))
    lint_report(markdown, resolved, supported_claims=claims)
    values = [value for headline in resolved for value in _energy_values(headline["value"])]
    for numerical in _energy_values(markdown):
        if not any(math.isclose(numerical, value, rel_tol=1e-9, abs_tol=1e-8)
                   for value in values):
            raise ReportLintError("Numeric headline differs from the recorded result")
    return resolved


def write_recorded_report(markdown: str, headlines: list[dict], results: ResultsDB,
                          record_hash: str) -> str:
    resolved = lint_recorded_report(markdown, headlines, results, record_hash)
    missing = [_headline_line(item) for item in resolved if _headline_line(item) not in markdown]
    if missing:
        markdown += ("\n\n| Node | Value | Relation | Evidence | Gate | Screening cost | "
                     "Target cost interval | Backend kind |\n"
                     "| --- | --- | --- | --- | --- | --- | --- | --- |\n" +
                     "\n".join(missing) + "\n")
    return markdown


def lint_report(markdown: str, headlines: list[dict], *,
                supported_claims: set[str] | None = None) -> None:
    required = {"value", "relation", "evidence", "gate_verdict", "cost", "backend_kind"}
    if any(required - set(item) for item in headlines):
        raise ReportLintError("Every headline needs relation, evidence, gates, cost and backend")
    if ENERGY_NUMBER.search(markdown) and not headlines:
        raise ReportLintError("Numeric headline lacks relation, evidence, gate, cost and backend")
    remainder = markdown.replace(_validity_section(headlines), "", 1)
    for term in ("quantum advantage", "speedup"):
        if re.search(r"\b" + term + r"\b", markdown, re.IGNORECASE) and term not in (
                supported_claims or set()):
            raise ReportLintError(f"Unsupported claim: {term}")
    for item in headlines:
        line = _headline_line(item)
        if line not in remainder:
            continue
        if (re.search(r"\bexact\b", line, re.IGNORECASE) and
                item["relation"] != "equivalent"):
            raise ReportLintError("Unsupported exact claim in headline")
        if (re.search(r"\bcertified\b", line, re.IGNORECASE) and
                "certified_bound" not in item["evidence"]):
            raise ReportLintError("Unsupported certification in headline")
        if (re.search(r"\bqpu result\b", line, re.IGNORECASE) and
                "qpu" not in item["backend_kind"].split("→")):
            raise ReportLintError("Unsupported QPU claim in headline")
        remainder = remainder.replace(line, "", 1)
    lowered = remainder.lower()
    if re.search(r"\bcertified\b", lowered):
        raise ReportLintError("Unsupported certification claim")
    if re.search(r"\bexact\b", lowered):
        raise ReportLintError("Unsupported exact claim")
    if re.search(r"\bqpu result\b", lowered):
        raise ReportLintError("Unsupported QPU claim")


def _headline_line(item: dict) -> str:
    costs = ", ".join(f"{key}={value if value is not None else 'unknown'}"
                      for key, value in item["cost"].items())
    label = item["node"]
    if item.get("quantity_label"):
        label += " — " + item["quantity_label"]
    if item.get("screening_reference"):
        if item.get("empirical"):
            label += " — model-evaluation screening reference, not experimental data"
        elif "model-defined" in item.get("quantity_label", ""):
            label += " — declared-model reference; gate verdicts measured against it"
        else:
            label += (" — exact reference available only at screening scale; "
                      "gate verdicts measured against it")
    target = item.get("target_cost")
    target_text = (f"[{target['lower']:.3g}, {target['upper']:.3g}] {target['dimension']} "
                   f"at {target['target_size']:g} {target['feature']}"
                   if target else "unknown (<3 distinct sizes)")
    return (f"| {label} | {item['value']} | {item['relation']} | "
            f"{item['evidence']} | {item['gate_verdict']} | {costs} | "
            f"{target_text} | {item['backend_kind']} |")


def _validity_section(headlines):
    maps = [item["validity"] for item in headlines if item.get("validity")]
    lines = ["", "## Validity", ""] if maps else []
    for mapped in maps:
        lines.append(f"{mapped['method']} ({mapped['node']}), accuracy bar {mapped['bar']}: "
                     f"task point {mapped['task_point']['status']}.")
        lines.append(mapped["assumption"])
        for axis in mapped["axes"]:
            lines.append(f"Axis {axis['axis']}:")
            for region in axis["regions"]:
                lines.append(f"- {region['status']}: [{region['lo']}, {region['hi']}]")
            lines.append(f"Counterexamples: {axis['counterexamples']}")
    return "\n".join(lines)


def write_report(question: str, decision: str, reason: str,
                 headlines: list[dict], orbital_provenance: str | None = None, *,
                 results: ResultsDB | None = None, record_hash: str | None = None,
                 answer_node: str | None = None) -> str:
    if results is not None:
        annotations = {"screening_reference", "target_cost"}
        recorded = resolve_headlines([{key: value for key, value in item.items()
                                       if key not in annotations} for item in headlines],
                                     results, record_hash)
        headlines = [{**saved, **{key: item[key] for key in annotations if key in item}}
                     for saved, item in zip(recorded, headlines)]
    lines = ["# ChemFlow decision report", "", f"Question: {question}", "",
             f"Decision: {decision} — {reason}", "",
             ("Screening-scale dominance does not establish target-scale dominance "
              "when predicted cost intervals overlap or are unknown."), "",
             "| Node | Value | Relation | Evidence | Gate | Screening cost | Target cost interval | Backend kind |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if orbital_provenance:
        provenance_label = ("Model provenance" if any(item.get("empirical") for item in headlines)
                            else "Active-space provenance")
        lines[6:6] = [f"{provenance_label}: {orbital_provenance}", ""]
    for item in headlines:
        lines.append(_headline_line(item))
    lines.extend(["", "## Deployment", "",
                  ("Any emulators used during screening were for verification, "
                   "not production quantum execution.")])
    if answer_node is not None:
        lines.append(f"Selected answer: {answer_node}; deployment steps listed below.")
    for item in headlines:
        if item.get("screening_reference") and item["node"] != answer_node:
            continue
        if item.get("deployable") is False:
            lines.append(f"- {item['node']}: not deployable — {item['deployment_reason']}")
        elif "deployment" in item:
            quantum = ", ".join(f"{node} → {device}" for node, device in
                                item["deployment"].items()) or "none"
            classical = ", ".join(item["classical_nodes"]) or "none"
            lines.append(f"- {item['node']}: QPU steps: {quantum}; classical nodes: {classical}.")
    if section := _validity_section(headlines):
        lines.append(section)
    text = "\n".join(lines) + "\n"
    lint_report(text, headlines)
    return text
