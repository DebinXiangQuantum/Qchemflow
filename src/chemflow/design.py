"""Server-recorded, small-instance evidence for experimental algorithm templates."""

import json
import math
import re
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from chemflow.diagnostics import AdapterError
from chemflow.domains import get_pack
from chemflow.domains.fock_reference import MAX_ORBITALS
from chemflow.domains.fock_reference import reference as fock_reference
from chemflow.ids import canonical_json, content_hash
from chemflow.ir.schema import Edge, Node, Program
from chemflow.relations import TargetRelation, compose, satisfies
from chemflow.runtime import Executor
from chemflow.types import Estimate, SpectralEstimate, Spectrum
from chemflow.units import Q


@dataclass(frozen=True)
class VerificationReport:
    """Immutable canonical payload; nested values are detached on every access."""

    _payload: str

    @property
    def id(self):
        return content_hash(json.loads(self._payload))

    def __getattr__(self, name):
        payload = json.loads(self._payload)
        if name not in payload:
            raise AttributeError(name)
        return payload[name]

    def model_dump(self, **kwargs):
        return {"id": self.id, **json.loads(self._payload)}


class DesignStore:
    def __init__(self, path=None):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path) if path is not None else ":memory:")
        self._connection.execute("CREATE TABLE IF NOT EXISTS verification_reports "
                                 "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._connection.execute("CREATE TABLE IF NOT EXISTS designs "
                                 "(name TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def _record(self, report):
        with self._connection:
            self._connection.execute("INSERT OR IGNORE INTO verification_reports VALUES (?,?)",
                                     (report.id, report._payload))

    def get(self, report_id):
        row = self._connection.execute("SELECT payload FROM verification_reports WHERE id=?",
                                       (report_id,)).fetchone()
        if row is None:
            raise KeyError(report_id)
        report = VerificationReport(row[0])
        if report.id != report_id:
            raise ValueError("Verification record content hash mismatch")
        return report

    def list(self):
        return [{"accuracy": None, **json.loads(row[0])} for row in self._connection.execute(
            "SELECT payload FROM designs ORDER BY name")]

    def falsifications(self, task_hash):
        reports = [self.get(row[0]).model_dump() for row in self._connection.execute(
            "SELECT id FROM verification_reports")]
        return [report for report in reports if report.get("task_hash") == task_hash and
                "verdict" in report]

    def _design(self, name):
        row = self._connection.execute("SELECT payload FROM designs WHERE name=?",
                                       (name.removeprefix("design:"),)).fetchone()
        if row is None:
            raise KeyError(name)
        return {"accuracy": None, **json.loads(row[0])}


def candidate_program(program, target_node):
    """Select a candidate's dependencies, excluding separate task controls/references."""
    nodes = {node.id: node for node in program.nodes}
    selected = set()
    pending = [target_node]
    while pending:
        identifier = pending.pop()
        if identifier not in nodes:
            raise ValueError(f"Unknown design node: {identifier}")
        if identifier not in selected:
            selected.add(identifier)
            pending.extend(nodes[identifier].inputs.values())
    result = program.model_copy(deep=True)
    result.nodes = [node for node in result.nodes if node.id in selected]
    result.edges = [edge for edge in result.edges
                    if edge.source in selected and edge.target in selected]
    result.targets, result.references, result.gates = [target_node], [], []
    return result


def _family_node(program):
    family_nodes = [node for node in program.nodes if node.op == "impurity_model"]
    if len(family_nodes) != 1:
        raise ValueError("Design requires exactly one family node (impurity_model)")
    return family_nodes[0]


def _validate_target(target):
    kind = target.get("kind")
    if kind not in ("spectral_function", "ground_energy", "molecular_ground_energy",
                    "lattice_ground_energy"):
        raise ValueError(f"No exact reference for target kind: {kind}")
    if set(target.get("model", {})) != {"impurity_model"}:
        raise ValueError("No impurity_model exact reference for this target")


def supports_verification(target):
    try:
        _validate_target(target)
    except ValueError:
        return False
    return True


def _metric(value, exact):
    if isinstance(exact, Estimate) and "values_hartree" in exact.diagnostics:
        expected = exact.diagnostics["values_hartree"]
        actual = getattr(value, "diagnostics", {}).get("values_hartree", [])
        if not expected or len(actual) != len(expected):
            raise ValueError("Energy collection dimensions differ")
        error = max(abs(left - right) for left, right in zip(actual, expected))
        return error, "max_component_energy_error", error
    from chemflow.types import EnergyCurve
    from chemflow.types.curves import curve_error

    if isinstance(exact, EnergyCurve):
        return curve_error(value, exact), "max_pointwise_energy_error", None
    if isinstance(exact, SpectralEstimate):
        if not isinstance(value, SpectralEstimate):
            raise TypeError("Target requires a spectral estimate")
        if value.omega != exact.omega or value.unit != exact.unit:
            raise ValueError("Spectral verification requires a shared grid and unit")
        return float(np.trapezoid(np.abs(np.asarray(value.spectral) -
                                        np.asarray(exact.spectral)), exact.omega)), "l1", None
    unit = "hartree"
    if isinstance(value, Spectrum):
        energy = value.energies[0]
    elif isinstance(value, Estimate):
        unit = "hartree" if value.value.dimension == "energy" else value.value.unit
        energy = value.value.to(unit).value
    else:
        raise TypeError("Target requires an energy estimate or Ritz spectrum")
    difference = energy - (exact if isinstance(exact, float) else exact.value.to(unit).value)
    return abs(difference), "abs_energy_error" if unit == "hartree" else "abs_error", difference


def _verification_accuracy(program, target_node, target, task):
    spectral = target["kind"] == "spectral_function"
    kind = "l1_error_vs_exact" if spectral else "abs_error_vs_exact"
    if task is not None:
        if program.gates:
            raise ValueError("Task-bound designs must not contain program gates")
        gates = [gate for gate in task.spec.gates if gate.kind == kind]
        if len(gates) != 1:
            raise ValueError("Frozen task requires exactly one matching error gate")
        gate = gates[0]
        threshold = Q(gate.threshold, gate.unit)
        accuracy = {"source": "task", "task_hash": task.content_hash}
        confidence = 0.95
    else:
        if any(gate.node != target_node for gate in program.gates):
            raise ValueError("Verification gates must be on the target node")
        gates = [gate for gate in program.gates
                 if gate.l1_error_le is not None or gate.abs_error_le is not None]
        if len(gates) != 1:
            raise ValueError("Program requires exactly one error gate on the target node")
        gate = gates[0]
        if gate.reference is not None:
            raise ValueError("Verification gate must not name a reference")
        threshold = gate.l1_error_le if spectral else gate.abs_error_le
        other = gate.abs_error_le if spectral else gate.l1_error_le
        if threshold is None or other is not None:
            raise ValueError("Error gate measure does not fit the target kind")
        accuracy = {"source": "program"}
        confidence = gate.confidence
    unit = "dimensionless" if spectral else "hartree"
    converted = threshold.to(unit).value
    if not isinstance(converted, (int, float)) or not math.isfinite(converted) or converted < 0:
        raise ValueError("Accuracy threshold must be finite and nonnegative")
    accuracy.update(kind=kind, threshold=threshold.value, unit=threshold.unit)
    if not spectral:
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Accuracy confidence must be between zero and one")
        accuracy["confidence"] = confidence
    # Relation requirements are part of the bar: the task's relation_at_least gates, or any
    # program gate on the target node, not only the error gate's own relation.
    required = {item.relation for item in (task.spec.gates if task is not None else program.gates)
                if item.relation is not None and (task is None or
                                                  item.kind in ("relation_at_least", kind))}
    if len(required) > 1:
        raise ValueError("Conflicting relation requirements for verification")
    if required:
        accuracy["relation"] = TargetRelation(required.pop()).value
    return accuracy


def verify_design(program, target_node, *, target, registry, store, instances, seed,
                  ledger=None, task=None):
    if type(instances) is not int or instances < 1:
        raise ValueError("instances must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _validate_target(target)
    accuracy = _verification_accuracy(program, target_node, target, task)
    template = program.model_copy(deep=True)
    template.gates = []
    family = _family_node(template)
    if template.scans or template.loops:
        raise ValueError("Verification requires a single acyclic design")
    candidate = candidate_program(template, target_node)
    _family_node(candidate)
    target = deepcopy(target)
    n_bath = len(target["model"]["impurity_model"]["eps_bath"])
    if n_bath + 1 > MAX_ORBITALS:
        raise ValueError(f"Independent verification supports at most {MAX_ORBITALS} orbitals")
    if len(family.params["eps_bath"]) != n_bath:
        raise ValueError("Design family size differs from target")
    rng = np.random.default_rng(seed)
    observations = []
    resources = {"sim_shots": 0, "qpu_shots": 0, "wall_s": 0.0, "qubits": 0,
                 "native_2q_gates": 0, "native_depth": 0, "circuits": []}
    from chemflow.check.static import infer_context

    contexts, issues = infer_context(candidate, registry)
    if issues:
        raise ValueError("Invalid design parameters: " + "; ".join(issue.message for issue in issues))
    node = next(node for node in candidate.nodes if node.id == target_node)
    context = contexts[target_node]
    sector = next((item.params for item in candidate.nodes if item.id == node.inputs.get("sector")),
                  context.get("sector_params"))
    spectral = target["kind"] == "spectral_function"
    threshold = Q(accuracy["threshold"], accuracy["unit"]).to(
        "dimensionless" if spectral else "hartree").value
    mismatch = (family.params != target["model"]["impurity_model"] or
                context.get("hamiltonian_id") != content_hash(family.params) or
                sector != target["sector"] or
                context.get("target_spin") != target.get("target_spin", target["sector"].get(
                    "s", target["sector"].get("total_spin"))) or
                node.op not in (("spectral_function",) if spectral else
                                ("ground_energy", "rayleigh_ritz")) or
                spectral and any(node.params.get(key) != target.get(key) for key in
                                 ("orbital", "spin", "omega", "eta")))
    if mismatch:
        reason = "Candidate target identity differs in model, sector, spin, or observable/grid"
        report = VerificationReport(canonical_json({
            "program_hash": template.hash, "template": template.model_dump(mode="json"),
            "target_node": target_node, "target": target, "family": "impurity_model",
            "seed": seed, "instances": [{"metric": 1.0, "metric_name": "identity_mismatch",
                "relation_holds": False, "reason": reason}], "max_metric": 1.0,
            "inferred_relation": "model_change", "resources": resources,
            "accuracy": accuracy, "passed": False, "reason": reason}))
        store._record(report)
        return report
    fock_reference(target, target["model"]["impurity_model"])
    inferred = TargetRelation.equivalent
    for index in range(instances):
        params = get_pack("dynamics").random_instance("impurity_model", rng, n_bath=n_bath)
        instance = template.model_copy(deep=True)
        _family_node(instance).params = deepcopy(params)
        computed = Executor(registry).run(instance, ledger=ledger)
        for cost in computed["costs"].values():
            for key in ("sim_shots", "qpu_shots", "wall_s"):
                resources[key] += getattr(cost, key) or 0
        for identifier, prepared in computed.items():
            circuit = getattr(prepared, "circuit", None)
            if circuit is not None:
                from chemflow.backends.circuits import build_circuit, compile_native

                native = compile_native(build_circuit(circuit))
                count = sum(len(item.qubits) == 2 and item.operation.name != "barrier"
                            for item in native.data)
                depth = native.depth()
                resources["circuits"].append({"instance": index, "node": identifier,
                    "qubits": native.num_qubits, "native_2q_gates": count,
                    "native_depth": depth})
                resources["qubits"] = max(resources["qubits"], native.num_qubits)
                resources["native_2q_gates"] += count
                resources["native_depth"] = max(resources["native_depth"], depth)
        if target_node in computed["refused"]:
            observations.append({"params": params, "metric": 1.0,
                                 "metric_name": "refused", "relation_holds": False,
                                 "accuracy_holds": False,
                                 "reason": computed["refused"][target_node]})
            continue
        exact = fock_reference(target, params)
        value = computed[target_node]
        diagnostics = deepcopy(getattr(value, "diagnostics", {}))
        sectors = {}
        for role, identifier in node.inputs.items():
            spectrum = computed.get(identifier)
            if isinstance(spectrum, Spectrum):
                sector = spectrum.sector
                sectors[role] = {"size": len(spectrum.subspace.determinants),
                    "full_size": math.comb(sector.n_orb, sector.n_alpha) *
                                 math.comb(sector.n_orb, sector.n_beta),
                    "fraction": spectrum.coverage}
        diagnostics["sectors"] = sectors
        if spectral:
            diagnostics["exact_ground_energy"] = exact.diagnostics["ground_energy"]
            diagnostics["exact_sum_rule"] = float(np.trapezoid(exact.spectral, exact.omega))
            if "ground_energy" in diagnostics:
                diagnostics["ground_energy_error"] = (diagnostics["ground_energy"] -
                                                       diagnostics["exact_ground_energy"])
        resources.setdefault("subspaces", []).append({"instance": index, "sectors": sectors})
        metric, metric_name, difference = _metric(value, exact)
        margin = 0.0
        accuracy_holds = math.isfinite(metric)
        if not spectral and isinstance(value, Estimate):
            stat_err = value.uncertainty.stat_err
            if stat_err is not None:
                margin = float(norm.ppf((1 + accuracy["confidence"]) / 2)) * Q(
                    stat_err, value.value.unit).to("hartree").value
                accuracy_holds = accuracy_holds and math.isfinite(margin) and margin >= 0
            elif value.relation == TargetRelation.stochastic_estimate:
                accuracy_holds = False
        accuracy_holds = accuracy_holds and metric + margin <= threshold
        relation = value.relation
        holds = math.isfinite(metric)
        if relation == TargetRelation.equivalent:
            holds = holds and metric <= 1e-8
        elif relation == TargetRelation.variational_upper_bound:
            holds = holds and difference is not None and difference >= -1e-9
        elif relation == TargetRelation.model_change:
            holds = False
        inferred = compose(inferred, relation)
        observations.append({"params": params, "metric": metric, "metric_name": metric_name,
                             "relation": relation.value, "relation_holds": bool(holds),
                             "diagnostics": diagnostics,
                             "accuracy_holds": bool(accuracy_holds), "statistical_margin": margin})
    payload = {"program_hash": template.hash, "template": template.model_dump(mode="json"),
               "target_node": target_node, "target": target, "family": "impurity_model",
               "seed": seed, "instances": observations,
               "max_metric": max(item["metric"] for item in observations),
               "inferred_relation": inferred.value, "resources": resources,
               "accuracy": accuracy,
               "passed": all(item["accuracy_holds"] and item["relation_holds"]
                             for item in observations) and (
                                 "relation" not in accuracy or satisfies(
                                     inferred, TargetRelation(accuracy["relation"])))}
    report = VerificationReport(canonical_json(payload))
    store._record(report)
    return report


MOVES = ("more_shots", "enriched_ground", "doubles_closure", "more_ritz_roots", "full_sector")
"""Ordered transformations; each is tried at most once per loop (shots cap 40,000)."""

MOVE_RULES = (
    ("more_shots", ("incomplete_ground",), "Low ground coverage: double sampling first."),
    ("enriched_ground", ("ground_energy_error", "incomplete_ground"),
     "Improve the ground state in a ladder-enriched N-sector space."),
    ("doubles_closure", ("incomplete_ground", "incomplete_response", "sum_rule_error"),
     "Missing determinant support: add a second excitation layer."),
    ("more_ritz_roots", ("truncated_resolution",),
     "Missing spectral roots: increase reported Ritz roots."),
    ("full_sector", ("accuracy_failure", "relation_failure"),
     "Cheaper applicable repairs exhausted: enumerate complete sectors."),
)
"""First applicable rule wins; triggers are ORed, never used as acceptance bars."""


@dataclass(frozen=True)
class DesignLoopResult:
    outcome: str
    trajectory: list[dict]
    design: dict | None


def apply_move(program, target_node, move):
    """Return a detached transformed candidate, or None when a move is inapplicable.

    Only algorithm dependencies change. Target parameters, sectors, Hamiltonian
    and declared gates are retained. Unused dependencies are pruned.
    """
    if move not in MOVES:
        raise ValueError(f"Unknown design move: {move}")
    result = candidate_program(program, target_node)
    result.gates = deepcopy(program.gates)
    nodes = {node.id: node for node in result.nodes}
    changed = False
    if move == "more_shots":
        for node in result.nodes:
            if node.op == "sample_configurations" and node.params.get("shots", 0) < 40000:
                node.params["shots"] = min(40000, 2 * node.params["shots"])
                changed = True
    elif move == "enriched_ground":
        ground = nodes.get(nodes[target_node].inputs.get("ground"))
        if ground is None or ground.op != "rayleigh_ritz":
            return None
        source = nodes[ground.inputs["subspace"]]
        if source.op != "subspace_from_samples":
            return None
        sector_id = source.inputs["sector"]
        enriched = next((node for node in reversed(result.nodes)
                         if node.op == "extend_subspace" and
                         node.inputs["sector"] == sector_id), None)
        if enriched is None:
            response = nodes[target_node]
            particle = nodes.get(response.inputs.get("particle"))
            if particle is None or particle.op != "rayleigh_ritz":
                return None
            particle_space = nodes[particle.inputs["subspace"]]
            identifier = f"{ground.id}_enriched"
            while identifier in nodes:
                identifier += "_"
            enriched = Node(id=identifier, op="extend_subspace",
                inputs={"subspace": particle_space.id, "sector": sector_id},
                params={"operator": "annihilate", "orbital": response.params["orbital"],
                        "spin": response.params["spin"], "closure": "singles"})
            result.nodes.append(enriched)
        ground.inputs["subspace"] = enriched.id
        changed = True
    elif move == "doubles_closure":
        for node in result.nodes:
            if node.op == "extend_subspace" and node.params.get("closure") != "doubles":
                node.params["closure"] = "doubles"
                changed = True
    elif move == "more_ritz_roots":
        for node in result.nodes:
            if node.op == "rayleigh_ritz":
                sector = nodes[nodes[node.inputs["subspace"]].inputs["sector"]].params
                limit = math.comb(sector["n_orb"], sector["n_alpha"]) * math.comb(
                    sector["n_orb"], sector["n_beta"])
                old = node.params.get("n_roots", 1)
                if old < limit:
                    node.params["n_roots"] = min(limit, 2 * old)
                    changed = True
    elif move == "full_sector":
        for node in result.nodes:
            if node.op in ("subspace_from_samples", "extend_subspace"):
                node.op = "full_subspace"
                node.inputs = {"sector": node.inputs["sector"]}
                node.params = {}
                node.binding = node.refinement = None
                changed = True
    if not changed:
        return None
    gates = result.gates
    result = candidate_program(result, target_node)
    result.gates = gates
    result.edges = [Edge(source=source, target=node.id, port=port)
                    for node in result.nodes for port, source in node.inputs.items()]
    return result


def diagnose_design(report):
    """Diagnose recorded evidence, without rerunning candidates or relaxing gates."""
    flags = set()
    if not report.passed:
        if any(not item.get("accuracy_holds", False) for item in report.instances):
            flags.add("accuracy_failure")
        if (any(not item.get("relation_holds", False) for item in report.instances) or
                "relation" in report.accuracy and not satisfies(
                    TargetRelation(report.inferred_relation),
                    TargetRelation(report.accuracy["relation"]))):
            flags.add("relation_failure")
    for item in report.instances:
        if item.get("metric_name") in ("identity_mismatch", "refused"):
            flags.add("invalid_candidate")
        diagnostics = item.get("diagnostics", {})
        coverage = diagnostics.get("coverage", {})
        if coverage.get("ground", 1) < 1:
            flags.add("incomplete_ground")
        if any(coverage.get(role, 1) < 1 for role in ("particle", "hole")):
            flags.add("incomplete_response")
        if abs(diagnostics.get("ground_energy_error", 0)) > 1e-8:
            flags.add("ground_energy_error")
        if abs(diagnostics.get("sum_rule", 0) - diagnostics.get("exact_sum_rule", 0)) > 1e-8:
            flags.add("sum_rule_error")
        if diagnostics.get("resolution") == "truncated":
            flags.add("truncated_resolution")
    return {"flags": sorted(flags), "instances": report.instances,
            "resources": report.resources}


def select_move(program, target_node, diagnosis, available):
    """Choose the cheapest plausible applicable transformation from MOVE_RULES."""
    flags = set(diagnosis["flags"])
    if "invalid_candidate" in flags:
        return None, None, None
    for move, triggers, rationale in MOVE_RULES:
        if move in available and flags.intersection(triggers):
            candidate = apply_move(program, target_node, move)
            if candidate is not None:
                return move, candidate, rationale
    return None, None, None


def design_loop(program, target_node, *, target, registry, store, instances, seed,
                max_rounds, moves=None, register_as=None, ledger=None, task=None):
    """Verify → diagnose → apply one move; never change the declared accuracy bar.

    max_rounds bounds verified candidates, not unverified transformations. Each
    move is used at most once. A terminal (passing or exhausted) entry has no move.
    Fixed random instances are reused across rounds for comparable evidence.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    available = set(MOVES if moves is None else moves)
    if not available <= set(MOVES):
        raise ValueError("Unknown design moves")
    candidate = program.model_copy(deep=True)
    trajectory = []
    for round_number in range(1, max_rounds + 1):
        execution = verify_design(candidate, target_node, target=target, registry=registry,
            store=store, instances=instances, seed=seed, ledger=ledger, task=task)
        payload = execution.model_dump()
        payload.pop("id")
        payload["resources"]["wall_s"] = None
        payload["timing"] = "Measured timing is in the trajectory's execution_report_id"
        report = VerificationReport(canonical_json(payload))
        store._record(report)
        diagnosis = diagnose_design(report)
        entry = {"round": round_number, "report_id": report.id, "passed": report.passed,
                 "max_metric": report.max_metric, "diagnosis": diagnosis, "move": None,
                 "execution_report_id": execution.id,
                 "rationale": None, "resources": execution.resources}
        trajectory.append(entry)
        if report.passed:
            design = (register_design(registry, store, report.id, register_as)
                      if register_as is not None else None)
            return DesignLoopResult("passed", trajectory, design)
        if round_number == max_rounds:
            break
        move, redesigned, rationale = select_move(candidate, target_node, diagnosis, available)
        if move is None:
            break
        entry.update(move=move, rationale=rationale)
        available.remove(move)
        candidate = redesigned
    return DesignLoopResult("exhausted", trajectory, None)


def register_design(registry, store, report_id, name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise ValueError("Invalid design name")
    report = store.get(report_id)
    if not report.passed:
        raise ValueError("Cannot register a failing verification report")
    identifier = f"design:{name}"
    if identifier in registry.designs:
        raise ValueError("Duplicate design name")
    entry = {"id": identifier, "status": "experimental", "relation": report.inferred_relation,
             "accuracy": report.model_dump().get("accuracy"),
             "evidence": report.id, "template": report.template,
             "target_node": report.target_node, "family": report.family, "target": report.target}
    try:
        with store._connection:
            store._connection.execute("INSERT INTO designs VALUES (?,?)",
                                      (name, canonical_json(entry)))
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate design name") from exc
    registry.designs[identifier] = deepcopy(entry)
    return deepcopy(entry)


def instantiate_design(store, name, *, model_params):
    entry = store._design(name)
    program = Program.model_validate(entry["template"])
    _family_node(program).params = deepcopy(model_params)
    node = next(node for node in program.nodes if node.id == entry["target_node"])
    node.metadata = {"design": entry["id"], "design_status": "experimental",
                     "design_evidence": entry["evidence"]}
    return program, node.id


def design_diagnostics(registry, program, node):
    if not node.metadata:
        return {}
    identifier = node.metadata.get("design")
    if identifier is None:
        return {}
    entry = registry.get(identifier)
    expected = {"design": identifier, "design_status": "experimental",
                "design_evidence": entry["evidence"]}
    if node.metadata != expected:
        raise ValueError("Invalid design evidence metadata")

    def signature(plan, target_node):
        candidate = candidate_program(plan, target_node)
        _family_node(candidate).params = {}
        for item in candidate.nodes:
            item.metadata = {}
        return content_hash([item.model_dump(mode="json") for item in candidate.nodes])

    if signature(program, node.id) != signature(Program.model_validate(entry["template"]),
                                                entry["target_node"]):
        raise ValueError("Registered design template was altered")
    return {**expected, "design_accuracy": deepcopy(entry.get("accuracy"))}


def check_design_accuracy(registry, program):
    for node in program.nodes:
        if not node.metadata.get("design"):
            continue
        recorded = design_diagnostics(registry, program, node)["design_accuracy"]
        for gate in program.gates:
            if gate.node != node.id:
                continue
            for kind, threshold in (("l1_error_vs_exact", gate.l1_error_le),
                                    ("abs_error_vs_exact", gate.abs_error_le)):
                if threshold is None:
                    continue
                compatible = recorded is not None and recorded.get("kind") == kind
                if compatible:
                    try:
                        required = threshold.to(recorded["unit"]).value
                        compatible = required >= recorded["threshold"]
                    except (KeyError, TypeError, ValueError):
                        compatible = False
                if not compatible:
                    raise AdapterError("CF-E120", node.id,
                        f"Design accuracy {recorded}; required bar {kind}: "
                        f"{threshold.value} {threshold.unit}",
                        "Verify at the stricter bar and register the design before use")
