"""Registered worker implementations of QCL operations.

Each implementation module registers handlers with ``@implements("op", ...)``. A handler
receives the execution context, the IR node, its id and operation name, and its resolved
inputs, and stores its result in ``ctx.values[identifier]``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

IMPLEMENTATIONS: dict[str, Callable[..., None]] = {}


@dataclass
class Context:
    values: dict[int, Any]
    ci_vectors: dict[int, Any]
    by_id: dict[int, dict[str, Any]]
    density_sources: set[int]
    profile: list[dict[str, Any]]
    store: Any
    request: dict[str, Any]
    qpu_backend_factory: Callable[[dict[str, Any]], Any] | None


def implements(*operations: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def register(function: Callable[..., None]) -> Callable[..., None]:
        for operation in operations:
            if operation in IMPLEMENTATIONS:
                raise ValueError(f"Duplicate QCL implementation for {operation}")
            IMPLEMENTATIONS[operation] = function
        return function
    return register


from qcl._impls import (  # noqa: F401  (registration side effects)
    ci,
    domains,
    engines,
    estimates,
    geometry,
    host,
    integrals,
    linalg,
    post,
    quantum,
    scf,
    transfer,
)
