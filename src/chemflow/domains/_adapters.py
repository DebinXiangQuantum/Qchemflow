"""Registration and result construction shared by external domain adapters."""

from chemflow.context import Context
from chemflow.registry import ImplementationSpec, OperatorSpec
from chemflow.types import BackendKind, CostRecord, Estimate, Provenance
from chemflow.units import Q


def register_adapter(registry, name, adapter, inputs, output, parameters, semantics,
                     effects="pure", cacheable=True):
    registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs=inputs,
        output=output, parameters={key: "JSON" for key in parameters},
        semantics=semantics, effects=effects))
    registry.register(ImplementationSpec(id=name, version="1", target=name,
        adapter=adapter, backend_kind=BackendKind.cpu_numeric,
        params=frozenset(parameters), effects=effects, cacheable=cacheable))


def estimate(value, kind, relation, keys, device, diagnostics=None, seed=None):
    return Estimate(value=Q(value, "hartree" if kind != "docking_score" else "kcal/mol"),
        kind=kind, target=keys["hamiltonian_id"], relation=relation, context=Context(keys=keys),
        provenance=Provenance(impl_id=device, impl_version="1", backend_kind="cpu_numeric",
                              device=device, seed=seed), cost=CostRecord(),
        diagnostics=diagnostics or {})
