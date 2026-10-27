from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import norm

from chemflow.agent.task import TaskRecord as AgentTaskRecord
from chemflow.agent.task import TaskStore
from chemflow.check import check
from chemflow.context import Context
from chemflow.diagnostics import AdapterError, CheckError, Diagnostic, EffectDenied
from chemflow.ids import content_hash
from chemflow.ir.schema import Program
from chemflow.registry import Registry
from chemflow.relations import (
    TargetRelation,
    combine_difference,
    combine_linear,
    compose,
    response_relation,
    ritz_relation,
    satisfies,
)
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.runtime.budget import (
    RESOURCE_DIMENSIONS,
    BudgetExceeded,
    BudgetLedger,
    resource_defaults,
)
from chemflow.runtime.result import ExecutionResult
from chemflow.runtime.task import TaskRecord, freeze_task
from chemflow.types import (
    CostRecord,
    EnergyCurve,
    Estimate,
    Evidence,
    EvidenceStatus,
    GateVerdict,
    Provenance,
    QuantityKind,
    Samples,
    SpectralEstimate,
    Spectrum,
    Subspace,
    Uncertainty,
    UpstreamRef,
)
from chemflow.units import Q


def _input_hash(value: Any) -> str:
    if isinstance(value, Estimate):
        return content_hash({"value": value.value.model_dump(), "kind": value.kind,
                             "relation": value.relation, "context": dict(value.context.keys),
                             "uncertainty": value.uncertainty.model_dump()})
    if hasattr(value, "hamiltonian") and hasattr(value, "model_id"):
        return content_hash({"hamiltonian": value.hamiltonian.id,
                             "model": value.model_protocol_id, "geometry": value.geometry_id})
    if hasattr(value, "model_dump"):
        return content_hash(value)
    if isinstance(value, (tuple, list)):
        return content_hash([_input_hash(item) for item in value])
    return content_hash(value)


def _check_relation_consistency(results: dict[str, Any], nodes_by_id: dict) -> None:
    groups: dict[tuple, list[tuple[str, Estimate]]] = {}
    for node_id, estimate in results.items():
        if not isinstance(estimate, Estimate) or estimate.relation not in (
                TargetRelation.equivalent, TargetRelation.variational_upper_bound):
            continue
        node = nodes_by_id[node_id]
        keys = estimate.context.keys
        operator = (node.inputs.get("operator", node.op) if node.op not in
                    ("energy_difference", "compare") else tuple(sorted(node.inputs.items())))
        identity = (operator, *(keys.get(key) for key in (
            "hamiltonian_id", "model_protocol_id", "sector_id", "target_spin",
            "quantity", "site_pair")),
            node.inputs.get("prep") if node.op in ("expectation", "sample") else None)
        if estimate.value.dimension != "energy":
            identity = (*identity, estimate.kind, estimate.target, estimate.value.dimension)
        groups.setdefault(identity, []).append((node_id, estimate))
    issues = []
    for estimates in groups.values():
        for index, (left_id, left) in enumerate(estimates):
            for right_id, right in estimates[index + 1:]:
                if (left.relation == right.relation ==
                        TargetRelation.variational_upper_bound):
                    continue
                unit = "hartree" if left.value.dimension == "energy" else left.value.unit
                left_value = left.value.to(unit).value
                right_value = right.value.to(unit).value
                if (left.relation == right.relation == TargetRelation.equivalent and
                        abs(left_value - right_value) <= 1e-7):
                    continue
                if (left.relation == TargetRelation.variational_upper_bound and
                        left_value >= right_value - 1e-7):
                    continue
                if (right.relation == TargetRelation.variational_upper_bound and
                        right_value >= left_value - 1e-7):
                    continue
                display_unit = "Ha" if unit == "hartree" else unit
                issues.append(Diagnostic(
                    code="CF-E130", node=left_id,
                    message=(f"Inconsistent relations: {left_id}={left_value:.12g} {display_unit}, "
                             f"{right_id}={right_value:.12g} {display_unit}"),
                    fix_hints=["Check the solver roots, target identity, and relation claims"]))
    if issues:
        raise CheckError(issues)


class Executor:
    def __init__(self, registry: Registry, store: ArtifactStore | None = None):
        self.registry = registry
        self.store = store or ArtifactStore()
        self.cache: dict[str, Any] = {}
        self.cache_lineage: dict[str, UpstreamRef] = {}
        self.cache_hits = 0

    def run_plan(self, plan: Program, record: AgentTaskRecord,
                 store: TaskStore, *,
                 ledger: BudgetLedger | None = None,
                 forecast_only: bool = False) -> dict[str, Any]:
        from chemflow.check.static import infer_context, infer_relations

        if not hasattr(store, "verify"):
            raise AdapterError("CF-E120", None, "Task store required", "Supply the signing store")
        store.verify(record)
        if plan.gates or not plan.targets or len(set(plan.targets)) != len(plan.targets):
            raise AdapterError("CF-E120", None, "Plan gates forbidden or targets missing",
                               "Declare unique target nodes without gates")
        for node in plan.nodes:
            implementation = self.registry.implementations.get(node.binding or node.op)
            if implementation is None or implementation.effects not in (
                    "pure", "random", "network") or node.effects not in ("pure", "random", "network"):
                raise AdapterError("CF-E080", node.id, "Unknown implementation or effect",
                                   "Bind an implementation with pure, random or network effects")
            effect = (implementation.effects if implementation.effects != "pure"
                      else node.effects)
            permission = "simulator" if effect == "random" else effect
            if permission not in record.spec.allowed_effects and effect not in record.spec.allowed_effects:
                raise EffectDenied(node.id, effect)
        contexts, issues = infer_context(plan, self.registry)
        if issues:
            raise CheckError(issues)
        from chemflow.domains import pack_for_target

        issues = pack_for_target(record.spec.target.kind).identity(plan, record, contexts)
        if issues:
            issue = issues[0]
            error = AdapterError(issue.code, issue.node, issue.message,
                                 issue.fix_hints[0] if issue.fix_hints else "Match the frozen task")
            error.diagnostics = [issue]
            raise error
        relations = infer_relations(plan, self.registry)
        reference = next((node.id for node in plan.nodes if node.id in plan.targets and
                          node.op == record.spec.target.operator and
                          relations.get(node.id) == TargetRelation.equivalent), None)
        from chemflow.ir.schema import AcceptanceGate
        from chemflow.units import Q

        execution = plan.model_copy(deep=True)
        execution.budget = dict(record.spec.budget)
        for node_id in plan.targets:
            context = contexts.get(node_id, {})
            for gate in record.spec.gates:
                attached = AcceptanceGate(node=node_id)
                if gate.kind == "relation_at_least":
                    attached.relation = gate.relation
                else:
                    if gate.kind == "l1_error_vs_exact":
                        attached.l1_error_le = Q(gate.threshold, gate.unit)
                    else:
                        attached.abs_error_le = Q(gate.threshold, gate.unit)
                    attached.relation = gate.relation
                    attached.reference = reference
                    if reference is not None and (
                        contexts[reference].get("hamiltonian_id") !=
                        context.get("hamiltonian_id") or
                        contexts[reference].get("sector_id") != context.get("sector_id") or
                        contexts[reference].get("target_spin") != context.get("target_spin")):
                        attached.reference = None
                attached.freeze()
                execution.gates.append(attached)
        if forecast_only:
            return self._run(execution, freeze_task(execution), ledger=ledger,
                             forecast_only=True)
        result = self.run(execution, freeze_task(execution), ledger=ledger)
        from chemflow.verify.integration import certify_results

        certify_results(plan, self.registry, result)
        for index, gate in enumerate(execution.gates):
            if ((gate.abs_error_le is not None or gate.l1_error_le is not None)
                    and gate.reference is None and gate.node not in result["refused"]):
                verdict = result["gates"][index]
                result["gates"][index] = verdict.model_copy(update={
                    "passed": False,
                    "reason": "Missing exact reference of the same target identity"})
        return result

    def run(self, program: Program, task: TaskRecord | None = None, *,
            ledger: BudgetLedger | None = None) -> dict[str, Any]:
        return self._run(program, task, ledger=ledger)

    def _run(self, program: Program, task: TaskRecord | None = None, *,
             ledger: BudgetLedger | None = None,
             preflight: bool = True, started_execution: bool = False,
             forecast_only: bool = False) -> dict[str, Any]:
        from chemflow.design import check_design_accuracy

        check_design_accuracy(self.registry, program)
        issues = check(program, self.registry, task=task)
        if issues:
            raise CheckError(issues)
        ledger = ledger or BudgetLedger(program.budget)
        from chemflow.check.static import infer_context

        static_contexts, _ = infer_context(program, self.registry)
        if program.scans:
            if len(program.scans) != 1:
                raise AdapterError("CF-E080", None, "Only one-dimensional scans supported",
                                   "Use one scan axis")
            scan = program.scans[0]
            instances = []
            for coordinate in scan.values.value:
                instance = program.model_copy(deep=True)
                instance.scans = []
                for node in instance.nodes:
                    if node.op == "molecule":
                        node.params["atoms"] = node.params["atoms"].replace(
                            "{" + scan.name + "}", str(coordinate))
                        node.params.pop("params", None)
                instances.append(instance)
            forecast = BudgetLedger(ledger.limits)
            forecast.spent = ledger.spent.copy()
            for instance in instances:
                for node in instance.nodes:
                    implementation = self.registry.implementations[node.binding or node.op]
                    try:
                        predicted = CostRecord(**resource_defaults(
                            implementation.resource_model(**node.params),
                            implementation.backend_kind))
                        forecast.reserve(predicted)
                        forecast.record(predicted)
                    except (BudgetExceeded, ValueError, TypeError) as exc:
                        raise AdapterError("CF-E070", node.id, str(exc),
                                           "Fit all scan points within the task budget",
                                           nothing_consumed=True) from exc
            if forecast_only:
                return {}
            points = []
            for instance in instances:
                point = self._run(instance, task=freeze_task(instance), ledger=ledger,
                                  preflight=False,
                                  started_execution=started_execution)
                points.append(point)
                started_execution = started_execution or bool(point["costs"])
            return {"scan": points, "refused": {
                f"{index}:{node}": reason for index, point in enumerate(points)
                for node, reason in point["refused"].items()},
                "budget_overruns": [item for point in points for item in point["budget_overruns"]]}
        refused: dict[str, str] = {}
        overruns: list[dict[str, Any]] = []
        results: dict[str, Any] = {"refused": refused, "budget_overruns": overruns}
        costs: dict[str, CostRecord] = {}
        lineage: dict[str, UpstreamRef] = {}
        nodes_by_id = {node.id: node for node in program.nodes}
        pending = {node.id: node for node in program.nodes}
        order = []
        resolved: set[str] = set()
        while pending:
            ready = [node for node in pending.values()
                     if all(source in resolved for source in node.inputs.values())]
            if not ready:
                raise AdapterError("CF-E080", None, "Cyclic graph or missing input node",
                                   "Connect an acyclic graph")
            order.extend(ready)
            for node in ready:
                resolved.add(node.id)
                del pending[node.id]
        forecast = BudgetLedger(ledger.limits)
        forecast.spent = ledger.spent.copy()
        cached_results: dict[str, Any] = {}
        for node in order if preflight else []:
            impl = self.registry.implementations[node.binding or node.op]
            inputs = {port: cached_results[source] for port, source in node.inputs.items()
                      if source in cached_results}
            effect = impl.effects if impl.effects != "pure" else node.effects
            if len(inputs) == len(node.inputs) and effect == "pure" and impl.cacheable:
                key = content_hash({"node": node.model_dump(), "impl_version": impl.version,
                                    "inputs": [_input_hash(item) for item in inputs.values()]})
                if key in self.cache:
                    cached_results[node.id] = self.cache[key]
                    continue
            try:
                predicted = CostRecord(**resource_defaults(
                    impl.resource_model(**inputs, **node.params), impl.backend_kind))
                forecast.reserve(predicted)
                forecast.record(predicted)
            except (BudgetExceeded, ValueError, TypeError) as exc:
                raise AdapterError("CF-E070", node.id, str(exc),
                                   "Reduce predicted cost to the remaining task budget",
                                   nothing_consumed=True) from exc
        if forecast_only:
            return {}
        for node in order:
            dependencies = [source for source in node.inputs.values() if source in refused]
            if dependencies:
                refused[node.id] = f"CF-E070: dependency {dependencies[0]} refused: {refused[dependencies[0]]}"
                continue
            impl = self.registry.implementations[node.binding or node.op]
            inputs = {port: results[source] for port, source in node.inputs.items()}
            if "sector" in inputs and hasattr(inputs.get("hamiltonian"), "for_sector"):
                try:
                    inputs["hamiltonian"] = inputs["hamiltonian"].for_sector(
                        inputs["sector"], self.store)
                except (ValueError, RuntimeError) as exc:
                    raise AdapterError("CF-E080", node.id, str(exc),
                                       "Check the requested model sector") from exc
            if not impl.capability_predicate(**inputs):
                raise AdapterError("CF-E080", node.id, "Implementation cannot run inputs",
                                   "Choose a capable binding")
            if not impl.validity_predicate(**inputs):
                raise AdapterError("CF-E020", node.id, "Invalid scientific inputs",
                                   "Check model and sector")
            effect = impl.effects if impl.effects != "pure" else node.effects
            key = content_hash({"node": node.model_dump(), "impl_version": impl.version,
                                "inputs": [_input_hash(item) for item in inputs.values()]})
            if effect == "pure" and impl.cacheable and key in self.cache:
                self.cache_hits += 1
                results[node.id] = self.cache[key]
                cached_lineage = self.cache_lineage.get(key)
                if cached_lineage is not None:
                    cached_value = results[node.id]
                    lineage[node.id] = cached_lineage.model_copy(update={
                        "device": (cached_value.device if isinstance(cached_value, Samples) else
                                   cached_value.provenance.device)})
                costs[node.id] = CostRecord(wall_s=0, cpu_core_s=0, gpu_s=0, sim_shots=0,
                                           qpu_shots=0, qpu_s=0)
                continue
            predicted = CostRecord(**resource_defaults(
                impl.resource_model(**inputs, **node.params), impl.backend_kind))
            try:
                ledger.reserve(predicted)
            except (BudgetExceeded, ValueError) as exc:
                if started_execution or costs:
                    refused[node.id] = (str(exc) if isinstance(exc, BudgetExceeded)
                                        else f"CF-E070: {exc}")
                    continue
                raise AdapterError("CF-E070", node.id, str(exc),
                                   "Supply bounded cost or increase budget",
                                   nothing_consumed=True) from exc
            started = time.monotonic()
            cpu_started = time.process_time()
            runtime_params = dict(node.params)
            if node.op == "qmmm_energy":
                runtime_params["remaining_budget"] = {
                    dimension: (max(0, limit - ledger.spent[dimension])
                                if ledger.spent[dimension] is not None else 0)
                    for dimension, limit in ledger.limits.items()}
            try:
                value = impl.adapter(store=self.store, **inputs, **runtime_params)
            except Exception as exc:
                if (effect == "pure" and node.op in ("ground_energy", "correlation_function",
                                                     "eom_charge_energy") and
                        node.metadata.get("optional_control") and
                        node.id not in program.references):
                    incurred = CostRecord(wall_s=time.monotonic() - started,
                        cpu_core_s=time.process_time() - cpu_started,
                        **{key: item for key, item in predicted.model_dump().items()
                           if key not in ("wall_s", "cpu_core_s", "diagnostics")},
                        diagnostics={"backend_kind": impl.backend_kind.value})
                    dimensions = ledger.record(incurred)
                    overruns.extend({"node": node.id, "dimension": dimension,
                                     "limit": ledger.limits[dimension],
                                     "spent": ledger.spent[dimension]} for dimension in dimensions)
                    costs[node.id] = incurred
                    refused[node.id] = (str(exc) if isinstance(exc, AdapterError)
                                        else f"CF-E080: {exc}")
                    continue
                if effect != "pure" and not getattr(exc, "nothing_consumed", False):
                    incurred = predicted.model_copy(update={
                        key: amount for key, amount in getattr(exc, "incurred_cost", {}).items()
                        if amount is not None and key in RESOURCE_DIMENSIONS})
                    ledger.record(incurred)
                if isinstance(exc, AdapterError):
                    exc.diagnostics = [issue.model_copy(update={"node": node.id})
                                       for issue in exc.diagnostics]
                    raise
                raise AdapterError("CF-E080", node.id, str(exc),
                                   "Check implementation inputs and supported scope") from exc
            execution_result = value if isinstance(value, ExecutionResult) else None
            if execution_result is not None:
                value = execution_result.value
            measured_cost = CostRecord(wall_s=time.monotonic() - started,
                                       cpu_core_s=time.process_time() - cpu_started,
                                       **{key: item for key, item in predicted.model_dump().items()
                                          if key not in ("wall_s", "cpu_core_s")})
            metadata = (value[2] if node.op == "ground_energy" and isinstance(value, tuple)
                        and len(value) == 3 else {})
            if metadata.get("sim_shots"):
                measured_cost = measured_cost.model_copy(update={"sim_shots": metadata["sim_shots"]})
            if isinstance(value, Samples):
                if value.source != impl.backend_kind:
                    raise AdapterError("CF-E060", node.id, "Samples source conflicts with binding",
                                       "Report actual backend")
                measured_cost = measured_cost.model_copy(update={
                    "qpu_shots" if impl.backend_kind == "qpu" else "sim_shots": value.shots})
            if execution_result is not None:
                measured_cost = measured_cost.model_copy(update={
                    key: item for key, item in execution_result.cost.items()
                    if key in RESOURCE_DIMENSIONS})
                measured_cost = measured_cost.model_copy(update={
                    "diagnostics": {**execution_result.diagnostics,
                                    "measured_resources": dict(execution_result.cost)}})
            measured_cost = measured_cost.model_copy(update={"diagnostics": {
                **measured_cost.diagnostics, "backend_kind": impl.backend_kind.value}})
            dimensions = ledger.record(measured_cost.model_copy(update={
                dimension: getattr(predicted, dimension)
                for dimension in RESOURCE_DIMENSIONS
                if getattr(measured_cost, dimension) is None}))
            if dimensions:
                measured_cost = measured_cost.model_copy(update={"diagnostics": {
                    **measured_cost.diagnostics, "budget_overrun": dimensions}})
                overruns.extend({"node": node.id, "dimension": dimension,
                                 "limit": ledger.limits[dimension],
                                 "spent": ledger.spent[dimension]} for dimension in dimensions)
            costs[node.id] = measured_cost
            if isinstance(value, Estimate) and value.provenance.backend_kind != impl.backend_kind:
                raise AdapterError("CF-E060", node.id, "Provenance backend conflicts with binding",
                                   "Report actual backend")
            if isinstance(value, Estimate):
                upstream = tuple(lineage[source].model_copy(update={"port": port})
                                 for port, source in node.inputs.items() if source in lineage)
                value = value.model_copy(update={"cost": measured_cost,
                    "provenance": value.provenance.model_copy(update={
                        "impl_id": impl.id, "impl_version": impl.version,
                        "input_hashes": [_input_hash(item) for item in inputs.values()],
                        "upstream": upstream})})
            if isinstance(value, (Subspace, Spectrum, SpectralEstimate, EnergyCurve)):
                upstream = []
                for port, source_id in node.inputs.items():
                    if source_id in lineage:
                        upstream.append(lineage[source_id].model_copy(update={"port": port}))
                    source_provenance = getattr(inputs[port], "provenance", None)
                    if source_provenance is not None:
                        upstream.extend(source_provenance.upstream)
                upstream = list({(item.node, item.impl_id): item for item in upstream}.values())
                provenance = Provenance(impl_id=impl.id, impl_version=impl.version,
                                        backend_kind=impl.backend_kind, device="local",
                                        seed=node.params.get("seed"), upstream=tuple(upstream),
                                        input_hashes=[_input_hash(item) for item in inputs.values()])
                updates = {"provenance": provenance, "cost": measured_cost}
                if isinstance(value, Spectrum):
                    updates["relation"] = ritz_relation(value.coverage == 1)
                if isinstance(value, SpectralEstimate):
                    updates["relation"] = response_relation(
                        all(fraction == 1 for fraction in value.diagnostics["coverage"].values()))
                value = value.model_copy(update=updates)
            if node.op in ("ground_energy", "expectation", "sample", "energy_difference",
                           "compare", "reaction_energy", "periodic_scf", "potential_energy") or (
                    node.op == "correlation_function" and node.params.get("site_pair") is not None):
                refinement = self.registry.refinements.get(node.refinement or
                                                          ("fci" if node.op == "ground_energy"
                                                           else ""))
                relation = refinement.relation if refinement else TargetRelation.equivalent
                if node.op == "sample":
                    relation = compose(relation, TargetRelation.stochastic_estimate)
                if node.op == "ground_energy" and node.params.get("measurement") == "shots":
                    relation = compose(relation, TargetRelation.stochastic_estimate)
                if node.op in ("energy_difference", "compare"):
                    relation = combine_difference(inputs["left"].relation,
                                                  inputs["right"].relation)
                    if node.params.get("model_change"):
                        relation = compose(relation, TargetRelation.model_change)
                if node.op == "reaction_energy":
                    relation = combine_linear([(node.params["coefficients"][port], item.relation)
                                               for port, item in inputs.items()
                                               if port in node.params["coefficients"]])
                if node.op == "periodic_scf":
                    relation = TargetRelation.approximate
                if node.op == "correlation_function":
                    relation = (TargetRelation.equivalent if inputs["state"].relation ==
                                TargetRelation.equivalent else TargetRelation.approximate)
                if node.op == "potential_energy":
                    relation = TargetRelation.model_change
                diagnostics = {}
                if node.op == "ground_energy":
                    numerical, diagnostics = value[:2]
                    diagnostics = {**diagnostics, "sector": inputs["hamiltonian"].sector.model_dump()}
                    ham_node = nodes_by_id[node.inputs["hamiltonian"]]
                    if ham_node.op == "hamiltonian" and not ham_node.params.get("active"):
                        molecule = results[ham_node.inputs["molecule"]]
                        model = nodes_by_id[ham_node.inputs["model"]].params
                        diagnostics.update(molecule_witness=molecule,
                                           electronic_model_witness=model)
                    stat_err = metadata.get("stat_err")
                elif isinstance(value, tuple):
                    numerical, stat_err = value
                else:
                    numerical, stat_err = value, None
                weakening = metadata.get("weakening")
                reason = None
                if weakening is not None:
                    requested = TargetRelation(weakening["relation"])
                    if not satisfies(relation, requested):
                        raise AdapterError("CF-E050", node.id, "Implementation attempted to strengthen relation",
                                           "Only weaken the registered relation")
                    if requested != relation:
                        relation = requested
                        reason = weakening["reason"]
                        diagnostics["relation_weakening_reason"] = reason
                upstream = []
                for port, source_id in node.inputs.items():
                    source_value = inputs[port]
                    if isinstance(source_value, (Samples, Estimate)):
                        upstream.append(lineage[source_id].model_copy(update={"port": port}))
                        if isinstance(source_value, Estimate):
                            upstream.extend(source_value.provenance.upstream)
                upstream = list({(item.node, item.impl_id, item.backend_kind,
                                  item.device, item.job_ids): item for item in reversed(upstream)}.values())
                provenance = Provenance(impl_id=impl.id, impl_version=impl.version,
                                        backend_kind=impl.backend_kind,
                                        device=execution_result.device if execution_result else "local",
                                        software_versions=(execution_result.software_versions
                                                           if execution_result else {}),
                                        seed=node.params.get("seed"),
                                        relation_weakening_reason=reason,
                                        upstream=tuple(upstream),
                                        input_hashes=[_input_hash(item)
                                                      for item in inputs.values()])
                source = inputs.get("hamiltonian")
                context = (Context(keys={"model_id": source.model_id,
                                         "model_protocol_id": source.model_protocol_id,
                                         "geometry_id": source.geometry_id,
                                         "hamiltonian_id": source.hamiltonian.id})
                           if source is not None and hasattr(source, "model_id") else
                           inputs["left"].context if node.op in ("energy_difference", "compare") else
                           Context())
                if node.op == "ground_energy":
                    sector_node = nodes_by_id[node.inputs["sector"]]
                    spin = node.params.get("s", node.params.get("total_spin",
                                           sector_node.params.get("s", sector_node.params.get(
                                               "total_spin"))))
                    context = Context(keys={**context.keys,
                                            "sector_id": content_hash(sector_node.params),
                                            **({"target_spin": str(spin)} if spin is not None else {})})
                    if static_contexts[node.id].get("reaction_protocol"):
                        context = Context(keys={**context.keys, "model_protocol_id":
                                                static_contexts[node.id]["model_protocol_id"]})
                if node.op in ("expectation", "sample"):
                    operator = inputs["operator"]
                    encoding = operator[0] if isinstance(operator, tuple) else operator
                    context = Context(keys={"hamiltonian_id": encoding.hamiltonian_id,
                                            "encoding_id": encoding.id,
                                            "prepared_state_id": _input_hash(inputs["prep"])})
                kind = (QuantityKind.energy_difference if node.op in
                        ("energy_difference", "compare") else
                        QuantityKind.total_electronic_energy)
                if node.op in ("reaction_energy", "periodic_scf", "potential_energy",
                               "correlation_function") or (
                        source is not None and hasattr(source, "lattice_id")):
                    context = Context(keys={key: str(item) for key, item in
                                            static_contexts[node.id].items()
                                            if item is not None and key not in
                                            ("type", "kind", "unit")})
                    kind = QuantityKind(static_contexts[node.id]["kind"])
                value = Estimate(value=Q(float(numerical), "dimensionless" if node.op ==
                                         "correlation_function" else "hartree"),
                                 kind=kind, target=node.id,
                                 relation=relation, uncertainty=Uncertainty(stat_err=stat_err),
                                 evidence=[Evidence._runtime(EvidenceStatus.diagnostic_only)],
                                 context=context,
                                 provenance=provenance,
                                 cost=measured_cost,
                                 diagnostics={**diagnostics, **(execution_result.diagnostics
                                                               if execution_result else {})})
            if node.metadata.get("design"):
                from chemflow.design import design_diagnostics

                diagnostics = design_diagnostics(self.registry, program, node)
                if isinstance(value, (Estimate, Spectrum, SpectralEstimate, EnergyCurve)):
                    value = value.model_copy(update={"diagnostics": {
                        **value.diagnostics, **diagnostics}})
            results[node.id] = value
            if isinstance(value, (Samples, Estimate, Subspace, Spectrum, SpectralEstimate, EnergyCurve)):
                lineage[node.id] = UpstreamRef(port="", node=node.id, impl_id=impl.id,
                    backend_kind=impl.backend_kind,
                    device=value.device if isinstance(value, Samples) else value.provenance.device,
                    job_ids=value.job_ids if isinstance(value, Samples) else
                    execution_result.job_ids if execution_result else ())
            if effect == "pure" and impl.cacheable:
                self.cache[key] = value
                if node.id in lineage:
                    self.cache_lineage[key] = lineage[node.id]
        results["_artifacts"] = self.store
        from chemflow.verify.integration import certify_results

        original_results = {id(value): node_id for node_id, value in results.items()
                            if isinstance(value, (Estimate, SpectralEstimate, Spectrum, EnergyCurve))}
        certify_results(program, self.registry, results)
        for cache_key, cached in self.cache.items():
            if id(cached) in original_results:
                self.cache[cache_key] = results[original_results[id(cached)]]
        for plugin in self.registry.runtime_checks:
            plugin(program, results)
        _check_relation_consistency(results, nodes_by_id)
        if program.gates:
            verdicts = []
            for gate in program.gates:
                candidate = results.get(gate.node)
                relation = (candidate.relation if isinstance(candidate, (Estimate, SpectralEstimate, EnergyCurve)) else
                            TargetRelation.approximate)
                passed = isinstance(candidate, (Estimate, SpectralEstimate, EnergyCurve))
                reason = "Gate satisfied" if passed else "Node has no estimate"
                measured = None
                stat_err = None
                z = None
                if gate.l1_error_le is not None:
                    reference = results.get(gate.reference)
                    if not isinstance(candidate, SpectralEstimate) or not isinstance(
                            reference, SpectralEstimate):
                        passed, reason = False, "Reference or candidate has no spectral estimate"
                    elif (reference.relation != TargetRelation.equivalent or
                          reference.diagnostics.get("relation_status") != "proved"):
                        passed, reason = False, "L1 gate requires an exact reference"
                    elif (candidate.omega != reference.omega or candidate.unit != reference.unit or
                          static_contexts[gate.node].get("spectral_identity") !=
                          static_contexts[gate.reference].get("spectral_identity")):
                        passed, reason = False, "Spectral target identity or grid mismatch"
                    else:
                        error = float(np.trapezoid(np.abs(np.asarray(candidate.spectral) -
                                                        np.asarray(reference.spectral)),
                                                   candidate.omega))
                        measured = Q(error, "dimensionless")
                        if error > gate.l1_error_le.value:
                            passed, reason = False, "L1 spectral error exceeds threshold"
                if gate.relation is not None:
                    passed = (passed and candidate.diagnostics.get("relation_status") == "proved"
                              and satisfies(relation, TargetRelation(gate.relation)))
                    if not passed:
                        reason = "Relation does not match required relation"
                if gate.abs_error_le is not None:
                    reference = results.get(gate.reference)
                    if isinstance(candidate, EnergyCurve):
                        from chemflow.types.curves import curve_error

                        try:
                            error = curve_error(candidate, reference)
                            measured = Q(error, "hartree").to(gate.abs_error_le.unit)
                            passed = (passed and reference.relation == TargetRelation.equivalent and
                                reference.diagnostics.get("relation_status") == "proved" and
                                measured.value <= gate.abs_error_le.value)
                            if not passed:
                                reason = "Pointwise error exceeds threshold or reference is unverified"
                        except (ValueError, AttributeError):
                            passed, reason = False, "Curve identity or reference mismatch"
                    elif not isinstance(candidate, Estimate) or not isinstance(reference, Estimate):
                        passed = False
                        reason = "Reference or candidate has no estimate"
                    elif (reference.relation != TargetRelation.equivalent or
                          reference.diagnostics.get("relation_status") != "proved"):
                        passed = False
                        reason = "Absolute-error gate requires an exact reference"
                    else:
                        try:
                            value = candidate.value.to(gate.abs_error_le.unit).value
                            target = reference.value.to(gate.abs_error_le.unit).value
                            measured = Q(abs(value - target), gate.abs_error_le.unit)
                            if "values_hartree" in candidate.diagnostics:
                                values = candidate.diagnostics["values_hartree"]
                                expected = reference.diagnostics.get("values_hartree", [])
                                if not values or len(values) != len(expected):
                                    raise ValueError("Energy collection dimensions differ")
                                measured = Q(max(abs(left - right) for left, right in
                                                  zip(values, expected)), "hartree").to(
                                                      gate.abs_error_le.unit)
                            if (candidate.relation == TargetRelation.stochastic_estimate or
                                candidate.uncertainty.stat_err is not None):
                                if candidate.uncertainty.stat_err is None:
                                    passed = False
                                    reason = "Stochastic estimate has no statistical error"
                                else:
                                    stat_err = Q(candidate.uncertainty.stat_err,
                                                 candidate.value.unit).to(
                                        gate.abs_error_le.unit).value
                                    z = float(norm.ppf((1 + gate.confidence) / 2))
                            if measured.value + (z * stat_err if z is not None else 0) > gate.abs_error_le.value:
                                passed = False
                                reason = "Absolute error including statistical margin exceeds threshold"
                        except ValueError:
                            passed = False
                            reason = "Incompatible gate units"
                refusal = next((identifier for identifier in (gate.node, gate.reference)
                                if identifier in refused), None)
                if refusal is not None:
                    passed, reason = False, f"Node {refusal} refused: {refused[refusal]}"
                verdicts.append(GateVerdict(node=gate.node, passed=passed, measured=measured,
                                            threshold=gate.abs_error_le or gate.l1_error_le,
                                            reference=gate.reference,
                                            relation=relation, reason=reason, stat_err=stat_err, z=z))
            results["gates"] = verdicts
        results["costs"] = costs
        return results
