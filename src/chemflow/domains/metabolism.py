"""Thin model-defined C-H reactivity target, with no clinical or enzyme prediction."""

from chemflow import build
from chemflow.context import Context
from chemflow.diagnostics import Diagnostic
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.types import CostRecord, Estimate, Provenance
from chemflow.units import Q

from .reactivity import resources, site_reactivity


def metabolic_sites(molecule, electronic_model, level, *, store=None):
    from chemflow.agent.task import MetabolicSitesTarget

    target = MetabolicSitesTarget(kind="metabolic_sites", molecule=molecule,
                                  electronic_model=electronic_model, level=level)
    parameters = target.model_dump(exclude={"kind", "operator"})
    result = site_reactivity(store, molecule, electronic_model, level)
    score = max((site["score"] for site in result["sites"]), default=0.0)
    return Estimate(value=Q(score, "dimensionless"), kind="metabolic_sites",
        target=content_hash(parameters), relation="equivalent", context=Context(keys={}),
        provenance=Provenance(impl_id="metabolic_sites", impl_version="1",
                              backend_kind="cpu_numeric", device="PySCF/tblite/FreeSASA"),
        cost=CostRecord(), diagnostics={**result, "declared_model": parameters,
            "scoring_formula": "SASA_fraction * sigmoid((90 - BDE_kcal/mol) / 3)",
            "experimental_relation": "model_change", "validity_status": "uncalibrated",
            "limitations": "C-H abstraction heuristic; no enzyme or metabolic-rate validation"})


def register(registry):
    parameters = {"molecule": "dict", "electronic_model": "dict", "level": "dict"}
    registry.register(OperatorSpec(id="chemflow.metabolic_sites", version=1, inputs={},
        output="Estimate", parameters=parameters,
        semantics="Model-defined empirical metabolic C-H site scores from BDE and FreeSASA; "
                  "headline is maximum score, not measured metabolism or an accuracy claim."))
    registry.register(ImplementationSpec(id="metabolic_sites", version="1", target="metabolic_sites",
        adapter=metabolic_sites, implements="metabolic_sites", backend_kind="cpu_numeric",
        execution_role="classical", params=frozenset(parameters), resource_model=resources))
    registry.register(RefinementRule(target="metabolic_sites", template_id="metabolic_sites",
                                     relation="equivalent"))


def base_plan(record, active=None):
    parameters = record.spec.target.model_dump(exclude={"kind", "operator"})
    with build.program("metabolic-sites") as builder:
        control = builder.node("metabolic_sites", **parameters)
        reference = builder.node("metabolic_sites", **parameters)
        builder.ir.targets = [control.id]
        builder.ir.references = [reference.id]
        return builder.to_ir(), [control.id]


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    expected = record.spec.target.model_dump(exclude={"kind", "operator"})
    nodes = {node.id: node for node in plan.nodes}
    return [Diagnostic(code="CF-E120", node=identifier,
                       message="Metabolic site model differs from frozen task",
                       fix_hints=plan_hints(plan, base_plan(record)[0], identifier))
            for identifier in set(plan.targets + plan.references)
            if identifier not in nodes or nodes[identifier].op != "metabolic_sites" or
            nodes[identifier].params != expected or nodes[identifier].inputs]
