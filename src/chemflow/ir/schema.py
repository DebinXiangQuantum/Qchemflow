from __future__ import annotations

from pydantic import BaseModel, Field

from chemflow.ids import content_hash
from chemflow.units import Quantity


class Node(BaseModel):
    id: str
    op: str
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    binding: str | None = None
    refinement: str | None = None
    effects: str = "pure"
    metadata: dict = Field(default_factory=dict)


class Edge(BaseModel, frozen=True):
    source: str
    target: str
    port: str


class Scan(BaseModel):
    name: str
    values: Quantity


class BoundedLoop(BaseModel):
    body: list[str]
    max_iterations: int
    stop_condition: str


class AcceptanceGate(BaseModel):
    node: str
    relation: str | None = None
    abs_error_le: Quantity | None = None
    l1_error_le: Quantity | None = None
    reference: str | None = None
    relative_to: str | None = None
    confidence: float = Field(default=0.95, gt=0, lt=1)
    frozen_hash: str | None = None

    def payload(self) -> dict:
        return self.model_dump(exclude={"frozen_hash"}, mode="json")

    def freeze(self) -> None:
        self.frozen_hash = content_hash(self.payload())

    def intact(self) -> bool:
        return self.frozen_hash == content_hash(self.payload())


class Program(BaseModel):
    name: str
    version: int = 1
    targets: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    scans: list[Scan] = Field(default_factory=list)
    loops: list[BoundedLoop] = Field(default_factory=list)
    gates: list[AcceptanceGate] = Field(default_factory=list)
    budget: dict[str, float] = Field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self)

    def to_json(self) -> str:
        from chemflow.ids import canonical_json

        return canonical_json(self)

    @classmethod
    def from_json(cls, source: str) -> Program:
        return cls.model_validate_json(source)
