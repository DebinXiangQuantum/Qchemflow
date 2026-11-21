"""Result construction shared by the domain kernels."""

from chemflow.context import Context
from chemflow.types import CostRecord, Estimate, Provenance
from chemflow.units import Q


def estimate(value, kind, relation, keys, device, diagnostics=None, seed=None):
    return Estimate(value=Q(value, "hartree" if kind != "docking_score" else "kcal/mol"),
        kind=kind, target=keys["hamiltonian_id"], relation=relation, context=Context(keys=keys),
        provenance=Provenance(impl_id=device, impl_version="1", backend_kind="cpu_numeric",
                              device=device, seed=seed), cost=CostRecord(),
        diagnostics=diagnostics or {})
