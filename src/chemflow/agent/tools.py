from __future__ import annotations

import difflib
import fcntl
import json
import os
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from chemflow.agent.costs import estimate_cost as predict_cost
from chemflow.agent.memory import ResultsDB
from chemflow.agent.report import result_headline
from chemflow.agent.task import TaskRecord, TaskStore
from chemflow.cards import load_cards
from chemflow.check import check
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.ids import canonical_json
from chemflow.ir.render import render
from chemflow.ir.schema import Edge, Node, Program
from chemflow.registry import Registry
from chemflow.runtime import Executor
from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetLedger

ROLES = frozenset({"planner", "modeler", "designer", "critic", "analyst", "orchestrator"})
DESCRIPTIONS = {
    "app.list": "List the thirteen applications, inputs, defaults, limitations and examples.",
    "app.draft": "Draft unsigned application tasks for human review; never sign or execute.",
    "registry.search": "Find registered operators, refinements, or implementations before choosing a plan node.",
    "registry.get": "Inspect a registered item by identifier to verify its ports, effects, and capabilities.",
    "algorithm.verify": "Verify impurity_model ground-energy or spectral-function targets only, against the frozen task's error gate and independent exact reference; agents cannot supply an accuracy bar.",
    "algorithm.prove": "Execute a node's dependency closure at budgeted screen scale and return its relation certificate and independent kernel verdict.",
    "algorithm.validity": "Map the signed validity domain at the signed accuracy bar.",
    "algorithm.falsify": "Search seeded family instances for relation and implementation counterexamples.",
    "algorithm.register": "Register a passing server-recorded verification report as an experimental design.",
    "algorithm.list": "List server-recorded experimental designs and their verification evidence.",
    "cards.search": "Find method cards matching a query before comparing candidate recipes and costs.",
    "cards.get": "Read a method card by identifier for its validity limits and cost model.",
    "plan.new": "Create an empty server-side plan and retain its returned plan_id for subsequent edits.",
    "plan.base": "Build frozen-task classical controls and reference from the matching active space; returns a new plan_id.",
    "plan.candidates": "Add candidate recipes to a stored base plan; preserve controls and references and use the returned plan_id.",
    "plan.add_node": "Add one typed node to a stored plan; use registered operators and JSON-encoded parameter values.",
    "plan.refine": "Apply a refinement template and JSON-encoded parameters to a node in a stored plan.",
    "plan.bind": "Bind a node to an implementation after checking its capabilities and effects.",
    "plan.remove_node": "Remove an unreferenced node from a stored plan; referenced nodes cannot be removed.",
    "design.submit": "Statically check a stored candidate plan and end the designer's turn only on success; errors allow repair.",
    "design.decline": "Decline without a candidate, or verify that plan_id exceeds the remaining budget; only success ends the turn.",
    "model.choose": "Choose an automatic active space with a reason and end the modeler's turn immediately.",
    "plan.render": "Render a stored plan for inspection; plans cannot carry acceptance gates.",
    "plan.diff": "Compare two stored plan handles to review proposed changes before screening.",
    "check": "Statically check a stored plan for port and context issues before attempting execution.",
    "estimate_cost": "Predict candidate costs from cards and recorded runs; unknown estimates need a resource model.",
    "run": "Execute a validated stored plan; all devices consume the frozen task resource budget, never plan gates.",
    "results.query": "Search recorded task results using task or operator filters without changing evidence.",
    "results.get": "Retrieve a recorded result by identifier to inspect provenance and evidence.",
    "diagnose.model": "Diagnose a molecular model and suggest an active space; spin target must match the frozen task.",
    "report.write": "Write a report using result_id headlines from run.results; reject invented or altered provenance.",
}


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryInput(ToolInput):
    query: str = Field("", description="Search text")


class AppDraftInput(ToolInput):
    app_id: str
    inputs: str | dict | None = Field(None, description="JSON object of application inputs, or null",
        json_schema_extra={"anyOf": [{"type": "string"}, {"type": "null"}]})
    parameters: str | dict | None = Field(None,
        description="JSON object of parameter overrides, or null",
        json_schema_extra={"anyOf": [{"type": "string"}, {"type": "null"}]})
    example: bool = False

    @model_validator(mode="before")
    @classmethod
    def encode_objects(cls, values):
        if isinstance(values, dict):
            return {key: json.dumps(value) if key in ("inputs", "parameters") and
                    isinstance(value, dict) else value for key, value in values.items()}
        return values


class RegistrySearchInput(QueryInput):
    kind: str = Field("", description="Optional registry kind")


class IdentifierInput(ToolInput):
    identifier: str = Field("", description="Item identifier")


class ActiveSpace(ToolInput):
    n_elec: int
    n_orb: int


class PortSource(ToolInput):
    port: str
    source: str


class Parameter(ToolInput):
    key: str
    value_json: str


class NodeSpec(ToolInput):
    id: str
    op: str
    inputs: list[PortSource]
    params: list[Parameter]
    effects: Literal["pure", "random", "network"]
    refinement: str | None
    binding: str | None


class Candidate(ToolInput):
    template: Literal["sci", "sqd", "vqe", "vqe_sampled"]
    select_cutoff: float | None = None
    shots: int | None = None


class PlanInput(ToolInput):
    plan_id: str = Field(description="Server-side plan handle")


class DeclineInput(ToolInput):
    reason: str = Field(min_length=1, pattern=r"\S", description="Reason for declining to propose")
    plan_id: str | None = Field(default=None, description="Plan to verify as unaffordable")
    force: bool = Field(default=False, description="Keep controls despite unscreened candidates")

    @field_validator("plan_id", mode="before")
    @classmethod
    def nullable_plan(cls, value):
        return None if value in ("null", "") else value


class ModelChoiceInput(ToolInput):
    n_elec: int = Field(strict=True, description="Number of active electrons")
    n_orb: int = Field(strict=True, description="Number of active spatial orbitals")
    reason: str = Field(min_length=1, pattern=r"\S", description="Reason for this active space")


class VerifyDesignInput(PlanInput):
    target_node: str
    instances: int = Field(gt=0, strict=True)
    seed: int = Field(ge=0, strict=True)


class RegisterDesignInput(ToolInput):
    report_id: str
    name: str


class ActiveInput(PlanInput):
    active: ActiveSpace | None = Field(default=None, description="Optional chosen active space")


class CandidatesInput(PlanInput):
    candidates: list[Candidate] = Field(default_factory=list, description="Candidate recipes")


class AddNodeInput(PlanInput):
    node: NodeSpec = Field(description="Node to add")
    target: bool = Field(False, description="Mark node as target")


class NodeIdInput(PlanInput):
    node_id: str = Field("", description="Node identifier")


class RefineInput(NodeIdInput):
    template: str = Field("", description="Refinement template")
    params: list[Parameter] = Field(default_factory=list, description="Node parameters")


class BindInput(NodeIdInput):
    binding: str = Field("", description="Implementation binding")


class DiffInput(PlanInput):
    other_plan_id: str = Field(description="Comparison plan handle")


class CostInput(PlanInput):
    target_size: int | None = Field(None, description="Prediction size")


class RunInput(PlanInput):
    scale: Literal["screen", "production"] = Field("screen", description="Execution purpose")


class ResultsQueryInput(ToolInput):
    task_id: str | None = None
    operator: str | None = None
    record_hash: str | None = None
    plan_hash: str | None = None
    target_kind: str | None = None


class DiagnoseInput(ToolInput):
    system_json: str = Field(description="JSON molecular system")
    model_json: str = Field(description="JSON electronic model")


class ReportInput(ToolInput):
    markdown: str = Field(description="Markdown report")
    headlines_json: str = Field("[]", description='JSON list of {"result_id": "id from run.results"}; server supplies provenance')


def _strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)

    def close(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                close(item)
        elif isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
                for name, prop in value.get("properties", {}).items():
                    prop.setdefault("description", name.replace("_", " ").capitalize())
            value.pop("default", None)
            for item in value.values():
                close(item)

    close(schema)
    return schema


class PlanStore:
    def __init__(self, directory: Path | None = None, record: TaskRecord | None = None):
        self._plans: dict[str, Program] = {}
        self.directory = directory
        self.record = record

    def put(self, plan: Program) -> str:
        if plan.gates:
            raise ValueError("Plans cannot contain gates")
        identifier = uuid4().hex
        copy = plan.model_copy(deep=True)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".plan-")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(copy.model_dump_json())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.directory / f"{identifier}.json")
                if self.record is not None:
                    context = {"budget": self.record.spec.budget,
                               "task_hash": self.record.content_hash}
                    with open(temporary, "w", encoding="utf-8") as stream:
                        stream.write(canonical_json(context))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.directory / f"{identifier}.resources.json")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        self._plans[identifier] = copy
        return identifier

    def get(self, identifier: str) -> Program:
        if self.directory is not None:
            if len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):
                raise KeyError(identifier)
            plan = Program.model_validate_json((self.directory / f"{identifier}.json").read_text())
        else:
            plan = self._plans[identifier].model_copy(deep=True)
        if plan.gates:
            raise ValueError("Plans cannot contain gates")
        return plan


class ToolOutput(BaseModel):
    data: Any


class ToolRegistry:
    def __init__(self):
        self.design_base_plan_id: str | None = None
        self._tools: dict[str, tuple[Callable, frozenset[str], type[BaseModel],
                                     type[BaseModel]]] = {}

    def register(self, name: str, roles: set[str] | frozenset[str], function: Callable,
                 input_model: type[BaseModel] = ToolInput,
                 output_model: type[BaseModel] = ToolOutput) -> None:
        if not roles <= ROLES or "freeze" in name or name in self._tools:
            raise ValueError("Invalid tool registration")
        self._tools[name] = (function, frozenset(roles), input_model, output_model)

    def schemas(self, role: str) -> list[dict]:
        from chemflow.design import supports_verification

        record = getattr(self, "task_record", None)
        can_verify = record is not None and supports_verification(
            record.spec.target.model_dump(mode="json"))
        can_map_validity = record is not None and bool(record.spec.validity)
        return [{"name": name, "description": DESCRIPTIONS[name],
                 "parameters": self._schema_for(name, input_model),
                 "output": output_model.model_json_schema()}
                for name, (_, roles, input_model, output_model) in self._tools.items()
                if role in roles and (can_verify or name not in
                                      {"algorithm.verify", "algorithm.register"})
                and (can_map_validity or name != "algorithm.validity")]

    def _schema_for(self, name: str, model: type[BaseModel]) -> dict:
        schema = _strict_schema(model)
        definitions = schema.get("$defs", {})

        def inline(value, ancestors=()):
            if isinstance(value, list):
                return [inline(item, ancestors) for item in value]
            if not isinstance(value, dict):
                return value
            if "$ref" in value:
                reference = value["$ref"]
                if not reference.startswith("#/$defs/") or reference in ancestors:
                    raise ValueError(f"Cannot inline schema reference: {reference}")
                resolved = inline(definitions[reference.removeprefix("#/$defs/")],
                                  (*ancestors, reference))
                return {**resolved, **inline({key: item for key, item in value.items()
                                             if key != "$ref"}, ancestors)}
            return {key: inline(item, ancestors) for key, item in value.items()
                    if key != "$defs"}

        return inline(schema)

    def call(self, role: str, name: str, data: dict | None = None) -> dict:
        if name not in self._tools:
            raise PermissionError(f"Tool unavailable: {name}")
        function, roles, input_model, output_model = self._tools[name]
        if role not in roles:
            raise PermissionError(f"Role {role} cannot call {name}")
        inputs = input_model.model_validate(data or {})
        try:
            result = output_model.model_validate({"data": function(inputs)}).model_dump(mode="json")
            return json.loads(canonical_json(result))
        except (CheckError, ValueError, PermissionError, KeyError):
            raise
        except Exception as exc:
            raise AdapterError("CF-E080", None, f"Tool {name} failed: {exc}",
                               "Check tool inputs and implementation") from exc


def make_tools(registry: Registry, executor: Executor, record: TaskRecord,
               store: TaskStore, results: ResultsDB | None = None,
               plan_directory: Path | None = None) -> ToolRegistry:
    from chemflow.agent.orchestrator import Orchestrator
    from chemflow.text import FrozenTextReferences

    store.verify(record)
    frozen_record = record.model_copy(deep=True)
    text_refs = FrozenTextReferences()
    text_refs.collect(frozen_record.model_dump(mode="json"), "task")
    if getattr(frozen_record.spec.target, "active", None) != "auto":
        frozen_base, _ = Orchestrator._base_plan(frozen_record, None)
        text_refs.collect(frozen_base.model_dump(mode="json"), "base_plan")
    tools = ToolRegistry()
    tools.task_record = record
    tools.plans = PlanStore(plan_directory, record)
    operator_enum = Enum("RegisteredOperator", {f"OP_{index}": operator.id
                         for index, operator in enumerate(registry.operators.values())},
                         type=str)
    typed_node = create_model("RegisteredNode", __base__=NodeSpec, op=(operator_enum, ...))
    add_input = create_model("RegisteredAddInput", __base__=AddNodeInput,
                             node=(typed_node, ...))
    if plan_directory is not None:
        plan_directory.mkdir(parents=True, exist_ok=True)
        executor.store.persist(plan_directory / "artifacts")
    memory = results or ResultsDB(plan_directory / "results.sqlite" if plan_directory else ":memory:")
    ledger = BudgetLedger(record.spec.budget,
                          plan_directory / f"{record.content_hash}.ledger.json"
                          if plan_directory else None)
    run_lock = RLock()
    proof_runs = {}
    created_plans = set()
    submitted_candidates = set()
    all_roles = set(ROLES)
    from chemflow.design import DesignStore, candidate_program, register_design, verify_design

    tools.designs = DesignStore(plan_directory / "designs.sqlite" if plan_directory else None)
    from chemflow.agent.families import FamilySearch

    family_search = FamilySearch(registry, executor, record, store, ledger, tools.designs)
    tools.family_search = family_search
    for entry in tools.designs.list():
        registry.designs[entry["id"]] = entry

    def verify_algorithm(data: VerifyDesignInput) -> dict:
        store.verify(record)
        plan = candidate_program(tools.plans.get(data.plan_id), data.target_node)
        for node in plan.nodes:
            implementation = registry.implementations.get(node.binding or node.op)
            if implementation is None:
                raise ValueError("Unknown design implementation")
            effect = implementation.effects if implementation.effects != "pure" else node.effects
            permission = "simulator" if effect == "random" else effect
            if (effect not in ("pure", "random", "network") or
                    permission not in record.spec.allowed_effects and
                    effect not in record.spec.allowed_effects):
                raise AdapterError("CF-E120", node.id,
                                   f"Effect {effect} not allowed by frozen task",
                                   "Freeze a new task version")

        def execute():
            report = verify_design(plan, data.target_node,
                target=record.spec.target.model_dump(mode="json"), registry=registry,
                store=tools.designs, instances=data.instances, seed=data.seed,
                task=record, ledger=ledger)
            if report.passed:
                falsification = family_search.falsify(tools.plans.get(data.plan_id), data.target_node)
                apply_refutation(falsification)
            return {**report.model_dump(), "report_id": report.id}

        with run_lock:
            if ledger.path is None:
                return execute()
            with ledger.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger.reload()
                return execute()

    tools.register("algorithm.verify", {"designer", "orchestrator"}, verify_algorithm,
                   VerifyDesignInput)

    def register_algorithm(data: RegisterDesignInput) -> dict:
        try:
            verified = tools.designs.get(data.report_id)
        except KeyError as exc:
            raise ValueError(f"Unknown report id {data.report_id!r}; "
                             "must come from algorithm.verify") from exc
        matching = [item for item in tools.designs.falsifications(record.content_hash)
                    if item["program_hash"] == verified.program_hash]
        if not matching or any(item["verdict"] != "clean" for item in matching):
            raise ValueError("Registration requires clean falsification for the same program hash")
        return register_design(registry, tools.designs, data.report_id, data.name)

    tools.register("algorithm.register", {"designer", "orchestrator"},
                   register_algorithm, RegisterDesignInput)
    tools.register("algorithm.list", all_roles, lambda data: tools.designs.list())

    def plan_input(inputs: ToolInput) -> Program:
        return tools.plans.get(inputs.plan_id)

    def parameters(entries: list[Parameter]) -> dict:
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("Duplicate parameter key")
        return {entry.key: text_refs.resolve(json.loads(entry.value_json)) for entry in entries}

    def edit(inputs: ToolInput, action: str) -> dict:
        plan = plan_input(inputs)
        nodes = {node.id: node for node in plan.nodes}
        if action == "add_node":
            spec = inputs.node
            node = Node(id=spec.id, op=spec.op.value.removeprefix("chemflow."), effects=spec.effects,
                        inputs={item.port: item.source for item in spec.inputs},
                        params=parameters(spec.params), refinement=spec.refinement,
                        binding=spec.binding)
            if len(node.inputs) != len(spec.inputs):
                raise ValueError("Duplicate input port")
            if node.id in nodes:
                raise ValueError("Duplicate node id")
            plan.nodes.append(node)
            if inputs.target:
                plan.targets.append(node.id)
            plan.edges.extend(Edge(source=source, target=node.id, port=port)
                              for port, source in node.inputs.items())
        elif action == "remove_node":
            if any(inputs.node_id in node.inputs.values() for node in plan.nodes):
                raise ValueError("Node still referenced")
            plan.nodes = [node for node in plan.nodes if node.id != inputs.node_id]
            plan.edges = [edge for edge in plan.edges if edge.source != inputs.node_id]
            plan.targets = [node for node in plan.targets if node != inputs.node_id]
        else:
            node = nodes[inputs.node_id]
            if action == "refine":
                node.refinement = inputs.template
                node.params.update(parameters(inputs.params))
            else:
                node.binding = inputs.binding
        new_id = tools.plans.put(plan)
        created_plans.add(new_id)
        return {"plan_id": new_id,
                "diagnostics": [item.model_dump(mode="json") for item in check(plan, registry)]}

    from chemflow.apps import catalog, draft

    tools.register("app.list", all_roles, lambda data: catalog())
    tools.register("app.draft", {"planner"}, lambda data: draft(data.app_id,
        json.loads(data.inputs) if data.inputs else None,
        json.loads(data.parameters) if data.parameters else None, data.example), AppDraftInput)
    tools.register("registry.search", all_roles, lambda data: [
        {"id": item["id"], "kind": "design", "status": item["status"],
         "relation": item["relation"], "evidence": item["evidence"]}
        if isinstance(item, dict) else
        {"id": item.id if hasattr(item, "id") else item.template_id,
         "kind": type(item).__name__,
         **({"semantics": item.semantics, "parameters": item.parameters}
            if hasattr(item, "semantics") else {})}
        for item in registry.search(data.query, data.kind or None)],
                   RegistrySearchInput)
    tools.register("registry.get", all_roles, lambda data: _registry_data(registry, data.identifier),
                   IdentifierInput)
    cards = {card.id: card for card in load_cards(registry)}
    from chemflow.cards import card_data

    tools.register("cards.search", all_roles, lambda data: [
        card_data(card, record.spec) for card in cards.values()
        if data.query.lower() in (card.id + card.target + card.template).lower()], QueryInput)
    tools.register("cards.get", all_roles, lambda data: card_data(cards[data.identifier], record.spec),
                   IdentifierInput)
    tools.register("plan.new", {"modeler", "designer"}, lambda data: {
        "plan_id": tools.plans.put(Program(name=data.identifier or "agent-plan"))}, IdentifierInput)
    def base(data: ToolInput) -> dict:
        plan_input(data)
        plan, controls = Orchestrator._base_plan(
            frozen_record, data.active.model_dump() if data.active is not None else None)
        text_refs.collect(plan.model_dump(mode="json"), "base_plan")
        return {"plan_id": tools.plans.put(plan), "controls": controls}

    def candidates(data: ToolInput) -> dict:
        candidates = [candidate.model_dump(exclude_none=True) for candidate in data.candidates]
        plan, identifiers = Orchestrator._candidate_plan(plan_input(data), candidates)
        errors = check(plan, registry, purpose="deploy")
        if errors:
            raise CheckError(errors)
        plan_id = tools.plans.put(plan)
        created_plans.add(plan_id)
        return {"plan_id": plan_id, "ids": identifiers,
                "candidates": {identifier: candidate_status(plan, identifier)
                               for identifier in identifiers.values()}}

    def candidate_status(plan, identifier):
        from chemflow.check.static import infer_context
        from chemflow.domains import pack_for_target
        from chemflow.verify.semantics import dependency_closure

        trial = plan.model_copy(update={"targets": [identifier]})
        contexts, _ = infer_context(trial, registry)
        issues = pack_for_target(record.spec.target.kind).identity(trial, record, contexts)
        costs = [_cost_for(node, registry, cards) for node in dependency_closure(plan, identifier)]
        summary = {label: None if any(cost[key] is None for cost in costs) else
                   sum(cost[key] for cost in costs)
                   for label, key in (("sim_shots", "sim_shots"), ("qpu_shots", "qpu_shots"),
                                      ("cpu", "cpu_core_s"))}
        return {"identity_ok": not issues, "check": len(check(trial, registry, purpose="deploy")),
                "estimate_cost": summary}

    def candidate_key(plan, identifier):
        from chemflow.ids import content_hash
        from chemflow.verify.semantics import dependency_closure

        return content_hash([node.model_dump() for node in dependency_closure(plan, identifier)])

    def unscreened_candidates():
        from chemflow.verify.semantics import dependency_closure

        if tools.design_base_plan_id is None:
            return []
        base = tools.plans.get(tools.design_base_plan_id)
        pending = set()
        for plan_id in sorted(created_plans):
            plan = tools.plans.get(plan_id)
            if (plan.gates or plan.nodes[:len(base.nodes)] != base.nodes or
                    plan.references != base.references or not set(base.targets) <= set(plan.targets)):
                continue
            for identifier in set(plan.targets) - set(base.targets):
                try:
                    if candidate_key(plan, identifier) in submitted_candidates:
                        continue
                    needed = {node.id for node in dependency_closure(plan, identifier)}
                    needed.update(node.id for node in base.nodes)
                    trial = plan.model_copy(update={
                        "nodes": [node for node in plan.nodes if node.id in needed],
                        "edges": [edge for edge in plan.edges if edge.target in needed],
                        "targets": [*base.targets, identifier]})
                    if any(issue.severity == "error" for issue in
                           check(trial, registry, purpose="deploy")):
                        continue
                    executor.run_plan(trial, record, store, ledger=ledger, forecast_only=True)
                except (CheckError, KeyError, ValueError):
                    continue
                pending.add(identifier)
        return sorted(pending)

    tools.register("plan.base", {"modeler"}, base, ActiveInput)
    tools.register("model.choose", {"modeler"}, lambda data: data.model_dump(), ModelChoiceInput)
    tools.register("plan.candidates", {"designer"}, candidates, CandidatesInput)
    def submit(data: PlanInput) -> dict:
        try:
            trial = tools.plans.get(data.plan_id)
        except (KeyError, OSError) as exc:
            raise ValueError(f"Unknown plan_id {data.plan_id!r}; submit a plan_id returned "
                             "by extending observation.base_plan_id") from exc
        except ValueError as exc:
            raise ValueError("Extend observation.base_plan_id without changing gates: "
                             f"{exc}") from exc
        if tools.design_base_plan_id is None:
            raise ValueError("design.submit requires the run's observation.base_plan_id")
        base_plan = tools.plans.get(tools.design_base_plan_id)
        if (trial.gates or trial.nodes[:len(base_plan.nodes)] != base_plan.nodes or
                trial.references != base_plan.references or
                not set(base_plan.targets) <= set(trial.targets)):
            raise ValueError("Extend observation.base_plan_id: preserve its nodes as a prefix, "
                             "references and targets; do not add gates")
        if not set(trial.targets) - set(base_plan.targets):
            raise ValueError("Extend observation.base_plan_id with at least one new target")
        errors = [issue for issue in check(trial, registry, purpose="deploy")
                  if issue.severity == "error"]
        if errors:
            raise CheckError(errors)
        submitted_candidates.update(candidate_key(trial, identifier) for identifier in
                                    set(trial.targets) - set(base_plan.targets))
        return {"plan_id": data.plan_id}

    tools.register("design.submit", {"designer"}, submit, PlanInput)

    def decline(data: DeclineInput) -> dict:
        if data.plan_id is None:
            with run_lock:
                if ledger.path is not None:
                    ledger.reload()
                pending = unscreened_candidates()
            if pending and not data.force:
                raise ValueError(f"Unscreened identity-valid candidates: {', '.join(pending)}. "
                    "Submit a plan containing these candidates; screening, not card evidence "
                    "from other systems, decides. Decline again with `force: true` and a "
                    "task-specific reason to keep only the controls.")
            return {"reason": data.reason, **({"unscreened_candidate_ids": pending,
                                               "force": True} if data.force else {})}
        plan = tools.plans.get(data.plan_id)

        def forecast() -> dict:
            try:
                executor.run_plan(plan, record, store, ledger=ledger, forecast_only=True)
            except AdapterError as exc:
                if exc.diagnostics[0].code != "CF-E070" or not exc.nothing_consumed:
                    raise
                return {"reason": data.reason, "plan_id": data.plan_id,
                        "plan_hash": plan.hash, "refusal": {
                            "code": "CF-E070", "node": exc.diagnostics[0].node,
                            "message": str(exc), "nothing_consumed": True,
                            "spent_at_refusal": ledger.spent.copy()}}
            raise ValueError("Plan fits the remaining task budget; cannot decline it as unaffordable")

        with run_lock:
            if ledger.path is None:
                return forecast()
            with ledger.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger.reload()
                return forecast()

    tools.register("design.decline", {"designer"}, decline, DeclineInput)
    for action, input_model in (("add_node", AddNodeInput), ("refine", RefineInput),
                                ("bind", BindInput), ("remove_node", NodeIdInput)):
        tools.register(f"plan.{action}", {"modeler", "designer"},
                       lambda data, action=action: edit(data, action),
                       add_input if action == "add_node" else input_model)
    tools.register("plan.render", {"modeler", "designer"},
                   lambda data: render(plan_input(data)), PlanInput)
    tools.register("plan.diff", {"modeler", "designer"}, lambda data: "\n".join(
        difflib.unified_diff(render(tools.plans.get(data.other_plan_id)).splitlines(),
                             render(plan_input(data)).splitlines())), DiffInput)
    tools.register("check", all_roles, lambda data: [item.model_dump(mode="json")
                   for item in check(plan_input(data), registry)], PlanInput)
    def cost_estimates(data: ToolInput) -> dict:
        target_size = data.target_size
        return {node.id: {"card": _cost_for(node, registry, cards),
                "target": predict_cost(memory, node.refinement, target_size)
                if target_size and node.refinement else None}
                for node in plan_input(data).nodes}

    tools.register("estimate_cost", {"designer", "critic"}, cost_estimates, CostInput)

    def run(data: ToolInput) -> dict:
        plan = plan_input(data)
        if data.scale not in ("screen", "production"):
            raise ValueError("Unknown execution scale")
        if data.scale == "production":
            errors = check(plan, registry, purpose="production")
            if errors:
                raise CheckError(errors)
        with run_lock:
            if ledger.path is None:
                computed = executor.run_plan(plan, record, store, ledger=ledger)
            else:
                with ledger.path.with_suffix(".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    ledger.reload()
                    computed = executor.run_plan(plan, record, store, ledger=ledger)
        from chemflow.verify.integration import certify_results

        certify_results(plan, registry, computed, plan_directory)
        for node in plan.nodes:
            value = computed.get(node.id)
            if hasattr(value, "diagnostics") and candidate_program(
                    plan, node.id).hash in family_search.refuted:
                value.diagnostics["relation_status"] = "refuted"
        if record.spec.validity:
            for node in plan.nodes:
                if node.op == record.spec.target.operator and node.id in computed:
                    mapped = family_tool(NodeIdInput(plan_id=data.plan_id, node_id=node.id),
                                         "validity")
                    value = computed[node.id]
                    computed[node.id] = value.model_copy(update={"diagnostics": {
                        **value.diagnostics, "validity": mapped}})
        proof_runs[plan.hash] = computed
        identifiers = {}
        headlines = {}
        for node in plan.nodes:
            from chemflow.types import EnergyCurve, Estimate, SpectralEstimate

            if isinstance(computed.get(node.id), (Estimate, SpectralEstimate, EnergyCurve)):
                headline = result_headline(node.id, computed,
                    {gate.node: gate for gate in computed.get("gates", [])},
                    plan.references[0] if plan.references else "", record.spec, plan, registry)
                headline["scale"] = data.scale
                identifiers[node.id] = memory.add(task_id=record.content_hash,
                    record_hash=record.hash, plan_hash=plan.hash, node_id=node.id,
                    operator=node.op, estimate=computed[node.id], headline=headline)
                headlines[node.id] = {**headline, "result_id": identifiers[node.id]}
        return {"plan_hash": plan.hash, "scale": data.scale, "results": identifiers,
                "refused": computed["refused"], "budget_overruns": computed["budget_overruns"],
                "headlines": headlines,
                "estimates": {node_id: value.model_dump(
                    mode="json", fallback=lambda value: dict(value), warnings=False)
                              for node_id, value in computed.items()
                              if node_id in identifiers or isinstance(value, SpectralEstimate)},
                "gates": [gate.model_dump(mode="json") for gate in computed.get("gates", [])],
                "costs": {key: value.model_dump(mode="json") for key, value in
                          computed["costs"].items()}}

    tools.register("run", {"designer", "orchestrator"}, run, RunInput)

    def prove_algorithm(data: NodeIdInput) -> dict:
        from chemflow.check.static import infer_context
        from chemflow.diagnostics import EffectDenied
        from chemflow.domains import pack_for_target
        from chemflow.ids import content_hash
        from chemflow.runtime.task import freeze_task
        from chemflow.verify import check_certificate, prove
        from chemflow.verify.integration import certify_results

        plan = plan_input(data)
        store.verify(record)
        if plan.gates or plan.scans:
            raise ValueError("Proof screening requires a gate-free, unscanned stored plan")
        contexts, issues = infer_context(plan, registry)
        issues += check(plan, registry)
        issues += pack_for_target(record.spec.target.kind).identity(plan, record, contexts)
        if issues:
            raise CheckError(issues)
        closure = candidate_program(plan, data.node_id)
        closure.budget = dict(record.spec.budget)
        for node in closure.nodes:
            implementation = registry.implementations[node.binding or node.op]
            effect = implementation.effects if implementation.effects != "pure" else node.effects
            permission = "simulator" if effect == "random" else effect
            if permission not in record.spec.allowed_effects and effect not in record.spec.allowed_effects:
                raise EffectDenied(node.id, effect)

        def execute():
            computed = proof_runs.get(plan.hash)
            if computed is None or data.node_id not in computed:
                computed = executor.run(closure, freeze_task(closure), ledger=ledger)
                proof_runs[plan.hash] = {**proof_runs.get(plan.hash, {}), **computed}
                computed = proof_runs[plan.hash]
            certify_results(plan, registry, computed, plan_directory)
            certificate = prove(plan, data.node_id, registry=registry, computed=computed)
            verdict = check_certificate(certificate, plan, registry=registry, computed=computed)
            relation_status = "proved" if verdict.valid else "declared"
            if candidate_program(plan, data.node_id).hash in family_search.refuted:
                relation_status = "refuted"
                computed[data.node_id].diagnostics["relation_status"] = relation_status
            identifier = content_hash(certificate)
            if plan_directory is not None:
                path = plan_directory / "certificates"
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{identifier}.json").write_text(canonical_json(certificate) + "\n")
            return {"certificate": certificate, "certificate_id": identifier,
                    "relation_status": relation_status,
                    "kernel_verdict": {"valid": verdict.valid, "reasons": list(verdict.reasons)},
                    "scale": "screen"}

        with run_lock:
            if ledger.path is None:
                return execute()
            with ledger.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger.reload()
                return execute()

    tools.register("algorithm.prove", {"designer", "critic"}, prove_algorithm, NodeIdInput)

    def apply_refutation(result):
        if result["verdict"] != "falsified":
            return
        for stored in tools.plans._plans.values():
            computed = proof_runs.get(stored.hash, {})
            for node in stored.nodes:
                if candidate_program(stored, node.id).hash == result["program_hash"]:
                    memory.refute(stored.hash, node.id, result["report_id"])
                    value = computed.get(node.id)
                    if hasattr(value, "diagnostics"):
                        value.diagnostics["relation_status"] = "refuted"

    def family_tool(data, action):
        def execute():
            result = getattr(family_search, action)(plan_input(data), data.node_id)
            if action == "falsify":
                apply_refutation(result)
            return result

        with run_lock:
            if ledger.path is None:
                return execute()
            with ledger.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger.reload()
                return execute()

    for action in ("validity", "falsify"):
        tools.register(f"algorithm.{action}", {"designer", "critic"},
                       lambda data, action=action: family_tool(data, action), NodeIdInput)
    tools.register("results.query", all_roles, lambda data: memory.query(**data.model_dump(
        exclude_none=True)),
                   ResultsQueryInput)
    tools.register("results.get", all_roles, lambda data: memory.get(data.identifier),
                   IdentifierInput)
    tools.register("diagnose.model", {"modeler"}, lambda data: _diagnose(
        json.loads(data.system_json), json.loads(data.model_json)),
                   DiagnoseInput)
    from chemflow.agent.report import write_recorded_report

    def report(data: ToolInput) -> str:
        return write_recorded_report(data.markdown, json.loads(data.headlines_json), memory,
                                     record.hash)

    tools.register("report.write", {"analyst"}, report, ReportInput)
    return tools


def _registry_data(registry: Registry, identifier: str) -> dict:
    item = registry.get(identifier)
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude={"adapter", "capability_predicate",
                                                      "validity_predicate", "resource_model",
                                                      "subgraph_builder"})
    return {}


def _cost_for(node: Node, registry: Registry, cards: dict) -> dict:
    card = next((card for card in cards.values() if card.template == node.refinement), None)
    model = (card.cost_model if card else None) or {}
    prediction = {dimension: model[dimension] for dimension in RESOURCE_DIMENSIONS
                  if dimension in model}
    implementation = registry.implementations.get(node.binding or node.op)
    if implementation is not None:
        prediction.update(implementation.resource_model(**node.params))
        from chemflow.runtime.budget import resource_defaults

        prediction = resource_defaults(prediction, implementation.backend_kind)
    return {**dict.fromkeys(RESOURCE_DIMENSIONS), **prediction}


def _diagnose(system: dict, model: dict) -> dict:
    from pyscf import gto, mp, scf

    molecule = gto.M(atom=system["atoms"], basis=model["basis"], verbose=0,
                     charge=system.get("charge", 0), spin=system.get("spin", 0),
                     unit=system.get("unit", "angstrom"))
    reference = scf.RHF(molecule).run()
    occupied = molecule.nelectron // 2
    gap = float(reference.mo_energy[occupied] - reference.mo_energy[occupied - 1])
    perturbation = mp.MP2(reference).run()
    occupations = sorted((float(value) for value in
                          __import__("numpy").linalg.eigvalsh(perturbation.make_rdm1())),
                         reverse=True)
    fractional = [index for index, value in enumerate(occupations) if 0.02 <= value <= 1.98]
    selected = sorted(fractional, key=lambda index: abs(occupations[index] - 1))[:10]
    if len(selected) < 2:
        raise ValueError("Auto active space needs at least two fractional MP2 orbitals")
    orbital_count = len(selected)
    electron_count = round(sum(occupations[index] for index in selected))
    if electron_count % 2 != molecule.nelectron % 2:
        electron_count += 1 if electron_count < sum(occupations[index] for index in selected) else -1
    if not 0 < electron_count <= 2 * orbital_count or (molecule.nelectron - electron_count) < 0:
        raise ValueError("Auto active space has infeasible electron count")
    return {"rhf_energy": float(reference.e_tot), "homo_lumo_gap": gap,
            "mp2_noons": occupations, "selected_orbitals": selected,
            "orbital_provenance": ("size from MP2 natural occupations; "
                                   "orbitals: canonical HF frontier window"),
            "total_electrons": molecule.nelectron, "suggested_active": {
                "n_elec": electron_count, "n_orb": orbital_count}}
