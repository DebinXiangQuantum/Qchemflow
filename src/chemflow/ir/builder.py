from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from chemflow.ir.schema import AcceptanceGate, Edge, Node, Program, Scan
from chemflow.units import Q, Quantity


class Builder:
    def __init__(self, name: str):
        self.ir = Program(name=name)

    def node(self, op: str, **arguments: Any) -> Node:
        inputs = {key: value.id for key, value in arguments.items() if isinstance(value, Node)}
        params = {key: value for key, value in arguments.items() if not isinstance(value, Node)}
        node = Node(id=f"n{len(self.ir.nodes)}", op=op, inputs=inputs, params=params)
        self.ir.nodes.append(node)
        self.ir.edges.extend(Edge(source=source, target=node.id, port=port)
                             for port, source in inputs.items())
        return node

    def molecule(self, atoms: str, **kwargs: Any) -> Node:
        return self.node("molecule", atoms=atoms, **kwargs)

    def geometry(self, **kwargs: Any) -> Node:
        return self.node("geometry", **kwargs)

    def electronic_model(self, **kwargs: Any) -> Node:
        return self.node("electronic_model", **kwargs)

    def hamiltonian(self, molecule: Node, model: Node, **kwargs: Any) -> Node:
        return self.node("hamiltonian", molecule=molecule, model=model, **kwargs)

    def sector(self, **kwargs: Any) -> Node:
        return self.node("sector", **kwargs)

    def ground_energy(self, hamiltonian: Node, sector: Node, **kwargs: Any) -> Node:
        return self.node("ground_energy", hamiltonian=hamiltonian, sector=sector, **kwargs)

    def scan(self, name: str, values: Quantity) -> str:
        self.ir.scans.append(Scan(name=name, values=values))
        return name

    def accept(self, node: Node, **kwargs: Any) -> AcceptanceGate:
        gate = AcceptanceGate(node=node.id, **kwargs)
        gate.freeze()
        self.ir.gates.append(gate)
        return gate

    def budget(self, **limits: float) -> None:
        self.ir.budget.update(limits)

    def refine(self, node: Node, template: str, **params: Any) -> None:
        node.refinement = template
        node.params.update(params)

    def bind(self, node: Node, implementation: str) -> None:
        node.binding = implementation

    def to_ir(self) -> Program:
        return self.ir.model_copy(deep=True)


@contextmanager
def program(name: str) -> Iterator[Builder]:
    yield Builder(name)


__all__ = ["Builder", "Q", "program"]
