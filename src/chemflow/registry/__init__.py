import inspect
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemflow.relations import TargetRelation
from chemflow.types import BackendKind


class OperatorSpec(BaseModel, frozen=True):
    id: str
    version: int
    inputs: dict[str, str]
    output: str
    semantics: str
    parameters: dict[str, str] = Field(default_factory=dict)
    parameter_defaults: dict[str, Any] = Field(default_factory=dict)
    kind: str = "deterministic"
    context_rules: list[str] = Field(default_factory=list)
    effects: str = "pure"
    differentiable_wrt: list[str] = Field(default_factory=list)


class RefinementRule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target: str
    template_id: str
    relation: TargetRelation
    inputs: dict[str, str] = Field(default_factory=dict)
    context_rules: list[str] = Field(default_factory=list)
    subgraph_builder: Callable[..., Any] | None = None


class ImplementationSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    version: str
    target: str
    implements: str | None = None
    backend_kind: BackendKind
    execution_role: Literal["classical", "qpu", "emulator", "verification_only"] | None = None
    emulates: str | None = None
    adapter: Callable[..., Any]
    capability_predicate: Callable[..., bool] = lambda **kwargs: True
    validity_predicate: Callable[..., bool] = lambda **kwargs: True
    error_sources: list[str] = Field(default_factory=list)
    resource_model: Callable[..., dict[str, float | None]] = lambda **kwargs: {}
    effects: str = "pure"
    cacheable: bool = True
    params: frozenset[str] | None = None

    @model_validator(mode="after")
    def default_execution_role(self):
        if self.execution_role is None:
            self.execution_role = {BackendKind.cpu_numeric: "classical",
                                   BackendKind.gpu_numeric: "classical",
                                   BackendKind.qpu: "qpu"}.get(self.backend_kind)
        return self


class Registry:
    def __init__(self) -> None:
        self.loaded_packs: set[str] = set()
        self.operators: dict[str, OperatorSpec] = {}
        self.refinements: dict[str, RefinementRule] = {}
        self.implementations: dict[str, ImplementationSpec] = {}
        self.checks: list[Callable] = []
        self.context_checks: list[Callable] = []
        self.runtime_checks: list[Callable] = []
        self.designs: dict[str, dict] = {}

    def register(self, spec: OperatorSpec | RefinementRule | ImplementationSpec) -> None:
        if isinstance(spec, OperatorSpec):
            self.operators[f"{spec.id}@{spec.version}"] = spec
        elif isinstance(spec, RefinementRule):
            self.refinements[spec.template_id] = spec
        else:
            from chemflow.backends import QPU_CATALOG

            defaults = {BackendKind.cpu_numeric: "classical",
                        BackendKind.gpu_numeric: "classical", BackendKind.qpu: "qpu"}
            if spec.backend_kind in defaults:
                expected = defaults[spec.backend_kind]
                if spec.execution_role not in (None, expected):
                    raise ValueError(f"{spec.id}: backend requires execution_role={expected}")
                spec.execution_role = expected
            elif spec.execution_role not in ("emulator", "verification_only"):
                raise ValueError(f"{spec.id}: simulator requires an explicit execution_role")
            if spec.execution_role == "emulator" and spec.emulates not in QPU_CATALOG:
                raise ValueError(f"{spec.id}: emulator requires an op in QPU_CATALOG")
            if spec.execution_role == "emulator" and spec.emulates != spec.target:
                raise ValueError(f"{spec.id}: emulator must implement its emulated op")
            if spec.execution_role != "emulator" and spec.emulates is not None:
                raise ValueError(f"{spec.id}: only emulators may declare emulates")
            if spec.params is None:
                inherited = self.implementations.get(spec.target)
                spec.params = inherited.params if inherited else frozenset()
            self.implementations[spec.id] = spec
            operator_key = f"chemflow.{spec.target}@1"
            operator = self.operators.get(operator_key)
            if operator is not None and spec.id == spec.target:
                defaults = {key: value.default for key, value in
                            inspect.signature(spec.adapter).parameters.items()
                            if key in operator.parameters and
                            value.default is not inspect.Parameter.empty}
                self.operators[operator_key] = operator.model_copy(update={
                    "parameter_defaults": {**defaults, **operator.parameter_defaults}})

    def get(self, identifier: str) -> OperatorSpec | RefinementRule | ImplementationSpec | dict:
        for collection in (self.operators, self.refinements, self.implementations, self.designs):
            if identifier in collection:
                if collection is self.designs:
                    return deepcopy(collection[identifier])
                return collection[identifier]
        if f"{identifier}@1" in self.operators:
            return self.operators[f"{identifier}@1"]
        raise KeyError(f"Unsupported registration: {identifier}")

    def search(self, query: str, kind: str | None = None) -> list[Any]:
        collections = {"operator": self.operators, "refinement": self.refinements,
                       "implementation": self.implementations, "design": self.designs}
        return [deepcopy(item) if label == "design" else item
                for label, collection in collections.items() if kind in (None, label)
                for key, item in collection.items()
                if query.lower() in (key + " " + getattr(item, "semantics", "")).lower()]
