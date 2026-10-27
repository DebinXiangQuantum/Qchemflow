"""Budgeted derived instances of a signed task; no new task signatures."""

import time
from copy import deepcopy
from itertools import pairwise
from math import comb

import numpy as np

from chemflow.agent.task import ValidityAxis, task_axis_point
from chemflow.check import check
from chemflow.check.static import infer_context
from chemflow.design import VerificationReport, _metric, candidate_program
from chemflow.diagnostics import AdapterError, CheckError, EffectDenied
from chemflow.domains import pack_for_target
from chemflow.ids import canonical_json
from chemflow.impls.fci import matrix, sector_states, spin_square
from chemflow.runtime import Executor
from chemflow.runtime.budget import BudgetExceeded, resource_defaults
from chemflow.runtime.task import freeze_task
from chemflow.types import CostRecord, Sector, Spectrum
from chemflow.units import Q

ASSUMPTION = (
    "Regions interpolate bounding samples, assuming no unsampled status reversal. "
    "They are not interval proofs. Weinstein sigma bounds distance to some eigenvalue; "
    "a ground-energy bound additionally assumes ground-state identification."
)


def instantiate(program, axis, value):
    result = program.model_copy(deep=True)

    def scaled(molecule):
        from pyscf.gto import format_atom

        atoms = format_atom(molecule["atoms"], unit="Bohr")
        origin = np.asarray(atoms[0][1])
        return {**molecule, "atoms": "; ".join(
            symbol + " " + " ".join(str(float(coordinate)) for coordinate in
                origin + value * (np.asarray(position) - origin))
            for symbol, position in atoms)}

    if axis == "geometry_scale":
        from chemflow.domains.reactivity import components

        nodes = {node.id: node for node in result.nodes}
        for node in result.nodes:
            if node.op not in ("bond_dissociation", "ranking"):
                continue
            target = deepcopy(node.params["target"])
            node.params["target"] = target
            molecules = ([target["molecule"]] if node.op == "bond_dissociation" else
                         [item["molecule"] for item in target["items"]])
            for molecule in molecules:
                molecule.update(scaled(molecule))
            for index, molecule in enumerate(components(target)):
                nodes[node.inputs[f"component{index}"]].params["molecule"] = molecule
    offset = 0.
    if axis in ("internal_coordinate", "separation"):
        operator = "pes_scan" if axis == "internal_coordinate" else "interaction_scan"
        curve = next((node for node in result.nodes if node.op == operator), None)
        if curve is None:
            raise ValueError("Coordinate family requires a scan target")
        target = curve.params["target"]
        center = (target["coordinate"]["values"][0] if axis == "internal_coordinate" else
                  target["separations"][0])
        offset = value - center
    for node in result.nodes:
        if axis in ("internal_coordinate", "separation"):
            if node.op == "geometry_transform" and (
                    (axis == "separation") ==
                    (node.params.get("operation") == "translate_fragment")):
                node.params["value"] += offset
            if node.op == "constrained_optimize" and axis == "internal_coordinate":
                for constraint in node.params["constraints"]:
                    constraint["value"] += offset
            if node.op == "pes_scan" and axis == "internal_coordinate":
                target = node.params["target"]
                target["coordinate"]["values"] = [point + offset
                                                   for point in target["coordinate"]["values"]]
            if node.op in ("interaction_scan", "scan_interaction") and axis == "separation":
                target = node.params["target"]
                target["separations"] = [point + offset for point in target["separations"]]
        if axis == "binding_ph" and node.op == "protein_structure":
            node.params["ph"] = value
        elif axis == "geometry_scale" and node.op == "molecule":
            node.params = scaled(node.params)
        elif axis == "geometry_scale" and node.op in (
                "solvation_free_energy", "frontier_orbitals", "eom_charge_energy"):
            node.params["molecule"] = scaled(node.params["molecule"])
        elif axis == "geometry_scale" and node.op == "charge_energy_difference":
            node.params["target"]["molecule"] = scaled(node.params["target"]["molecule"])
        elif (axis, node.op) in (("hubbard_u", "lattice_model"),
                                 ("impurity_u", "impurity_model")):
            node.params["U"] = value
    return result


def default_axes(target):
    axis = {"molecular_ground_energy": "geometry_scale",
            "solvation_free_energy": "geometry_scale",
            "frontier_orbitals": "geometry_scale",
            "ionization_energy": "geometry_scale",
            "electron_affinity": "geometry_scale",
            "binding_free_energy": "binding_ph",
            "reaction_energy": "geometry_scale",
            "bond_dissociation": "geometry_scale",
            "ranking": "geometry_scale",
            "lattice_ground_energy": "hubbard_u",
            "pairing_correlation": "hubbard_u",
            "spectral_function": "impurity_u", "pes_scan": "internal_coordinate",
            "interaction_scan": "separation"}.get(target.kind)
    if axis is None:
        raise ValueError("No falsification family for this target")
    point = task_axis_point(target, axis)
    bounds = (0.5, 3.0) if axis == "geometry_scale" else (min(0., point), max(8., 2 * point))
    if axis in ("internal_coordinate", "separation"):
        bounds = (0.8 * point, 1.2 * point)
        if axis == "internal_coordinate" and target.coordinate.type == "dihedral":
            bounds = (point - 30, point + 30)
        elif axis == "internal_coordinate" and target.coordinate.type == "angle":
            bounds = (max(1., point - 20), min(179., point + 20))
    if axis == "binding_ph":
        bounds = (max(0., point - 1.), min(14., point + 1.))
    return (ValidityAxis(axis=axis, range=bounds, points=12,
                         resolution=(bounds[1] - bounds[0]) / 100),)


def state_checks(program, node_id, computed, store):
    node = next(item for item in program.nodes if item.id == node_id)
    result = computed[node_id]
    data = computed.get(node.inputs.get("hamiltonian"))
    if data is None:
        return None, []
    sector = computed.get(node.inputs.get("sector"))
    if isinstance(result, Spectrum):
        sector = result.sector
    if isinstance(sector, dict):
        sector = Sector.model_validate(sector)
    if sector is None:
        return None, []
    diagnostics = result.diagnostics
    if node.params.get("measurement", "exact") != "exact":
        return None, []
    basis_ref = diagnostics.get("determinants_ref")
    state_ref = diagnostics.get("state_ref")
    if not basis_ref and not state_ref and not isinstance(result, Spectrum):
        return None, []
    if comb(sector.n_orb, sector.n_alpha) * comb(sector.n_orb, sector.n_beta) > 4096:
        raise ValueError("Exact sector variance exceeds the 4096 determinant reference limit")
    states = sector_states(sector)
    operator = matrix(data.hamiltonian, store, states)
    energy = (result.energies[0] if isinstance(result, Spectrum) else
              result.value.to("hartree").value)
    failures = []
    basis = None
    if isinstance(result, Spectrum):
        basis = list(result.subspace.determinants)
    elif basis_ref:
        raw = store.get(basis_ref, verify=True)
        basis = ([int(alpha) | (int(beta) << sector.n_orb) for alpha, beta in raw]
                 if raw.ndim == 2 else raw.tolist())
    state = None
    if basis is not None:
        indices = [states.index(mask) for mask in basis]
        projected = operator[indices][:, indices].toarray()
        energies, vectors = np.linalg.eigh(projected)
        target_spin = node.params.get("s", node.params.get("total_spin", sector.total_spin))
        if target_spin is not None:
            compatible = []
            for index, eigenvalue in enumerate(energies):
                embedded = np.zeros(len(states))
                embedded[indices] = vectors[:, index]
                if abs(spin_square(embedded, sector) - target_spin * (target_spin + 1)) < 1e-8:
                    compatible.append(eigenvalue)
                    break
            if not compatible:
                raise ValueError("Independent projected spin resolution is inconclusive")
            expected = float(min(compatible))
        else:
            expected = float(energies[0])
        if abs(energy - expected) > 1e-8:
            failures.append({"property": "projected_lowest_eigenvalue", "observed": energy,
                             "expected": expected})
        if isinstance(result, Spectrum):
            state = np.zeros(len(states), dtype=complex)
            state[indices] = store.get(result.vectors_ref, verify=True)[:, 0]
    if state_ref:
        state = store.get(state_ref, verify=True)
    if state is None:
        return None, failures
    if state.shape != (len(states),) or not np.isfinite(state).all() or not np.isclose(
            np.vdot(state, state).real, 1., atol=1e-10, rtol=0):
        failures.append({"property": "normalized_sector_state", "observed": "invalid",
                         "expected": "finite normalized state in sector ordering"})
        return None, failures
    applied = operator @ state
    mean = float(np.vdot(state, applied).real)
    if abs(energy - mean) > 1e-8:
        failures.append({"property": "rayleigh_quotient", "observed": energy, "expected": mean})
    sigma = float(np.linalg.norm(applied - mean * state))
    return sigma, failures


class FamilySearch:
    def __init__(self, registry, executor, record, store, ledger, designs):
        self.registry, self.executor = registry, executor
        self.reference_executor = Executor(registry, executor.store)
        self.record, self.store, self.ledger, self.designs = record, store, ledger, designs
        self.maps = {}
        self.refuted = {report["program_hash"] for report in designs.falsifications(
            record.content_hash) if report["verdict"] == "falsified"}

    def prepare(self, plan, node_id):
        self.store.verify(self.record)
        contexts, issues = infer_context(plan, self.registry)
        issues += check(plan, self.registry)
        pack = pack_for_target(self.record.spec.target.kind)
        identity_plan = plan.model_copy(deep=True)
        ritz_nodes = {node.id for node in plan.nodes if node.op == "rayleigh_ritz"}
        identity_plan.targets = [node for node in plan.targets if node not in ritz_nodes]
        issues += pack.identity(identity_plan, self.record, contexts)
        if plan.gates or plan.scans:
            raise ValueError("Family screening requires a gate-free, unscanned plan")
        if issues:
            raise CheckError(issues)
        candidate = candidate_program(plan, node_id)
        active = next((node.params.get("active") for node in candidate.nodes
                       if node.op == "hamiltonian"), None)
        if self.record.spec.target.kind == "reaction_energy":
            active = None
        base, _ = pack.base_plan(self.record, active)
        reference = candidate_program(base, base.references[0])
        node = next(item for item in candidate.nodes if item.id == node_id)
        if node.op == "rayleigh_ritz":
            reference_contexts, _ = infer_context(reference, self.registry)
            expected = reference_contexts[reference.targets[0]]
            if self.record.spec.target.operator != "ground_energy" or any(
                    contexts[node_id].get(key) != expected.get(key) for key in
                    ("hamiltonian_id", "sector_id", "target_spin")):
                raise ValueError("Ritz target identity differs from signed task")
        else:
            issues = pack.identity(candidate, self.record, contexts)
            if issues:
                raise CheckError(issues)
        for program in (candidate, reference):
            program.budget = dict(self.record.spec.budget)
            for node in program.nodes:
                impl = self.registry.implementations[node.binding or node.op]
                effect = impl.effects if impl.effects != "pure" else node.effects
                permission = "simulator" if effect == "random" else effect
                if permission not in self.record.spec.allowed_effects and (
                        effect not in self.record.spec.allowed_effects):
                    raise EffectDenied(node.id, effect)
        return candidate, reference

    def evaluate(self, candidate, reference, axis, value, required_relation=None):
        try:
            return self._evaluate(candidate, reference, axis, value, required_relation)
        except BudgetExceeded:
            raise
        except (CheckError, ValueError, RuntimeError) as exc:
            if isinstance(exc, CheckError) and any(
                    issue.code == "CF-E070" for issue in exc.diagnostics):
                raise
            return {"value": value, "status": "unknown", "reason": str(exc), "failures": []}

    def _evaluate(self, candidate, reference, axis, value, required_relation=None):
        from chemflow.verify.integration import certify_results

        trial = instantiate(candidate, axis, value)
        exact_plan = instantiate(reference, axis, value)
        node_id = trial.targets[0]
        runs = []
        for executor, program in ((self.executor, trial), (self.reference_executor, exact_plan)):
            run = executor.run(program, freeze_task(program), ledger=self.ledger)
            refusals = [str(reason) for reason in run.get("refused", {}).values()]
            if run.get("budget_overruns") or any("CF-E070" in reason for reason in refusals):
                raise AdapterError("CF-E070", node_id, "; ".join(refusals) or
                                   "Family instance exceeded the task budget",
                                   "Increase the task budget")
            if program.targets[0] not in run:
                raise ValueError("Family instance refused: " + ("; ".join(refusals) or
                                 "candidate or exact reference unavailable"))
            runs.append(run)
        computed, exact_run = runs
        certify_results(trial, self.registry, computed)
        result, exact = computed[node_id], exact_run[exact_plan.targets[0]]
        metric, _, difference = _metric(result, exact)
        checked_at = time.monotonic()
        cpu_started = time.process_time()
        self.ledger.reserve(CostRecord(**resource_defaults({}, "cpu_numeric")))
        failures, bound = [], None
        try:
            for node in trial.nodes:
                if not hasattr(computed.get(node.id), "diagnostics"):
                    continue
                sigma, checks = state_checks(trial, node.id, computed, self.executor.store)
                failures.extend({**failure, "node": node.id} for failure in checks)
                if node.id == node_id:
                    bound = sigma
        finally:
            overruns = self.ledger.record(CostRecord(**resource_defaults({
                "wall_s": time.monotonic() - checked_at,
                "cpu_core_s": time.process_time() - cpu_started}, "cpu_numeric")))
        if overruns:
            raise AdapterError("CF-E070", node_id, "Independent checks exceeded the task budget",
                               "Increase the task budget")
        relation = result.relation.value
        if relation not in ("equivalent", "variational_upper_bound") or (
                result.diagnostics.get("relation_status") != "proved"):
            bound = None
        checked_relation = required_relation or relation
        if checked_relation == "equivalent" and metric > 1e-8:
            failures.insert(0, {"property": checked_relation, "observed": metric,
                                "expected": "<= 1e-8"})
        if checked_relation == "variational_upper_bound" and difference is not None and difference < -1e-9:
            failures.insert(0, {"property": checked_relation, "observed": difference,
                                "expected": ">= -1e-9"})
        scalar = lambda item: (item.energies[0] if isinstance(item, Spectrum) else
                               item.value.to("hartree" if item.value.dimension == "energy"
                                             else item.value.unit).value
                               if hasattr(item, "value") else
                               item.model_dump(mode="json"))
        return {"value": value, "result": scalar(result), "exact": scalar(exact),
                "error": difference if difference is not None else metric,
                "certified_bound": bound, "failures": failures,
                "relation": relation, "certificate": computed["relation_certificates"][node_id],
                "derived_from": self.record.content_hash, "instance_hash": trial.hash}

    def validity(self, plan, node_id):
        started = time.monotonic()
        spec = self.record.spec
        if not spec.validity:
            raise ValueError("algorithm.validity requires a signed validity domain")
        candidate, reference = self.prepare(plan, node_id)
        gate = next((item for item in spec.gates if item.kind in (
            "abs_error_vs_exact", "l1_error_vs_exact")), None)
        if gate is None:
            raise ValueError("Validity requires a signed accuracy bar")
        bar = Q(gate.threshold, gate.unit).to(
            "dimensionless" if gate.kind == "l1_error_vs_exact" or
            spec.target.kind == "pairing_correlation" else "hartree").value
        axes, unevaluable = [], []
        task_point = None
        for axis in spec.validity:
            samples = {}

            def sample(value, samples=samples, axis=axis):
                value = float(value)
                if value not in samples:
                    point = self.evaluate(candidate, reference, axis.axis, value)
                    if point.get("status") == "unknown":
                        unevaluable.append({"instance": {axis.axis: value},
                                            "reason": point["reason"]})
                        samples[value] = point
                        return point
                    bound = point["certified_bound"] if node_id not in plan.references else None
                    point["status"] = ("invalid" if abs(point["error"]) > bar else
                                       "certified" if bound is not None and bound <= bar and
                                       not point["failures"] else "sampled_valid")
                    samples[value] = {key: item for key, item in point.items()
                                      if key not in ("failures", "certificate")}
                return samples[value]

            initial = np.linspace(*axis.range, axis.points)
            for value in initial:
                sample(value)
            task_point = sample(task_axis_point(spec.target, axis.axis))
            pending = list(pairwise(sorted(samples)))
            while pending:
                lo, hi = pending.pop()
                statuses = (sample(lo)["status"], sample(hi)["status"])
                if statuses[0] != statuses[1] and hi - lo > axis.resolution:
                    middle = (lo + hi) / 2
                    if middle in (lo, hi):
                        raise ValueError("Validity resolution is below floating-point spacing")
                    sample(middle)
                    pending.extend(((lo, middle), (middle, hi)))
            points = [samples[value] for value in sorted(samples)]
            regions = []
            for left, right in pairwise(points):
                status = left["status"] if left["status"] == right["status"] else "unknown"
                if regions and regions[-1]["status"] == status:
                    regions[-1]["hi"] = right["value"]
                else:
                    regions.append({"lo": left["value"], "hi": right["value"], "status": status})
            axes.append({**axis.model_dump(mode="json"), "points": points, "regions": regions,
                         "counterexamples": [{"value": point["value"], "error": point["error"]}
                                             for point in points if point["status"] == "invalid"]})
        node = next(item for item in plan.nodes if item.id == node_id)
        if all(point["status"] == "unknown" for axis in axes for point in axis["points"]):
            raise AdapterError("CF-E080", node_id, "No family instance could be evaluated: " +
                               "; ".join(sorted({item["reason"] for item in unevaluable})),
                               "Choose an evaluable family domain")
        result = {"node": node_id, "method": node.refinement or node.op, "bar": bar,
                  "unevaluable": unevaluable,
                  "axes": axes, "task_point": task_point, "assumption": ASSUMPTION,
                  "wall_s": time.monotonic() - started, "task_hash": self.record.content_hash}
        self.maps[(plan.hash, node_id)] = result
        return result

    def falsify(self, plan, node_id):
        started = time.monotonic()
        candidate, reference = self.prepare(plan, node_id)
        axes = self.record.spec.validity or default_axes(self.record.spec.target)
        seed = 1729
        rng = np.random.default_rng(seed)
        instances, counterexamples, axioms, unevaluable = [], [], set(), []
        for axis in axes:
            center = task_axis_point(self.record.spec.target, axis.axis)
            baseline = self.evaluate(candidate, reference, axis.axis, center)
            samples = {center: baseline}

            def record_instance(point, axis=axis):
                if point.get("status") == "unknown":
                    unevaluable.append({"instance": {axis.axis: point["value"]},
                                        "reason": point["reason"]})
                    return
                instances.append({axis.axis: point["value"], "error": point["error"],
                                  "failures": point["failures"],
                                  "derived_from": self.record.content_hash,
                                  "instance_hash": point["instance_hash"]})

            record_instance(baseline)

            relation = baseline.get("relation")

            def sample(value, samples=samples, axis=axis,
                       record_instance=record_instance):
                nonlocal relation
                value = float(value)
                if value not in samples:
                    point = self.evaluate(candidate, reference, axis.axis, value,
                                          required_relation=relation)
                    samples[value] = point
                    record_instance(point)
                    if relation is None:
                        relation = point.get("relation")
                return samples[value]

            values = [*axis.range, center, sum(axis.range) / 2]
            values.extend(rng.uniform(*axis.range, 12).tolist())
            for value in values:
                sample(value)
            properties = {(failure["property"], failure.get("node", node_id))
                          for point in samples.values() for failure in point["failures"]}
            for property_name, failure_node in sorted(properties, key=lambda item: (
                    item[0] not in ("equivalent", "variational_upper_bound"), item)):
                def fails(value, property_name=property_name, failure_node=failure_node,
                          sample=sample):
                    return [failure for failure in sample(value)["failures"]
                            if failure["property"] == property_name and
                            failure.get("node", node_id) == failure_node]

                for side in (-1, 1):
                    candidates = [value for value in samples if (value - center) * side >= 0
                                  and fails(value)]
                    if not candidates:
                        continue
                    bad = min(candidates, key=lambda value: abs(value - center))
                    good = center
                    bracket_known = all(point.get("status") != "unknown"
                                        for value, point in samples.items()
                                        if min(good, bad) <= value <= max(good, bad))
                    if fails(center):
                        bad = center
                    elif bracket_known:
                        while abs(bad - good) > axis.resolution:
                            middle = (good + bad) / 2
                            if middle in (good, bad):
                                raise ValueError("Shrinking resolution is below floating-point spacing")
                            if sample(middle).get("status") == "unknown":
                                bracket_known = False
                                break
                            if fails(middle):
                                bad = middle
                            else:
                                good = middle
                    for failure in fails(bad):
                        example = {"instance": {axis.axis: bad}, **failure}
                        if bracket_known:
                            example["shrink_bracket"] = sorted((good, bad))
                        if example not in counterexamples:
                            counterexamples.append(example)
                    axioms.update(step["lemma"] for step in sample(bad)["certificate"].get(
                        "steps", []) if step["id"] == f"{failure_node}:axiom")
        if not instances:
            raise AdapterError("CF-E080", node_id, "No family instance could be evaluated: " +
                               "; ".join(sorted({item["reason"] for item in unevaluable})),
                               "Choose an evaluable family domain")
        if counterexamples:
            self.refuted.add(candidate_program(plan, node_id).hash)
        payload = {"verdict": "falsified" if counterexamples else "clean", "seed": seed,
                   "instances": instances, "generators": ["edge", "random"],
                   "unevaluable": unevaluable,
                   "counterexamples": counterexamples, "refuted_axioms": sorted(axioms),
                   "program_hash": candidate_program(plan, node_id).hash,
                   "task_hash": self.record.content_hash,
                   "domain": [axis.model_dump(mode="json") for axis in axes],
                   "wall_s": time.monotonic() - started,
                   "assumption": "Clean means no counterexample found, not a universal proof."}
        report = VerificationReport(canonical_json(payload))
        self.designs._record(report)
        return {**payload, "report_id": report.id}
