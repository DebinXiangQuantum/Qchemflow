"""Model-defined aqueous solubility, not measured experimental solubility."""

from chemflow import build
from chemflow.context import Context
from chemflow.diagnostics import Diagnostic
from chemflow.empirical import evaluate, model_card
from chemflow.ids import content_hash
from chemflow.registry import ImplementationSpec, OperatorSpec, RefinementRule
from chemflow.types import CostRecord, Estimate, Provenance
from chemflow.units import Q


def logs_esol(molecule_smiles, *, store=None):
    return evaluate("esol", molecule_smiles)


def logs_gse(molecule_smiles, melting_point_C, *, store=None):
    return evaluate("gse", molecule_smiles, melting_point_C)


def empirical_estimate(molecule_smiles, model, melting_point_C=None, *, store=None):
    result = evaluate(model, molecule_smiles, melting_point_C)
    card = model_card(model)
    identity = content_hash((model, molecule_smiles, melting_point_C))
    return Estimate(value=Q(result["value"], result["unit"]), kind="aqueous_solubility",
                    target=identity, relation="equivalent", context=Context(keys={}),
                    provenance=Provenance(impl_id="empirical_estimate", impl_version="1",
                                          backend_kind="cpu_numeric", device="RDKit/NumPy",
                                          software_versions=card["software_versions"]),
                    cost=CostRecord(), diagnostics={**result, "model_card": card,
                                                  "experimental_relation": "model_change"})


def resources(**kwargs):
    return {"wall_s": None, "cpu_core_s": None, "gpu_s": 0, "sim_shots": 0,
            "qpu_s": 0, "qpu_shots": 0}


def register(registry):
    for name, adapter, parameters, output in (
        ("logs_esol", logs_esol, {"molecule_smiles": "str"}, "EmpiricalResult"),
        ("logs_gse", logs_gse, {"molecule_smiles": "str", "melting_point_C": "float"},
         "EmpiricalResult"),
        ("empirical_estimate", empirical_estimate,
         {"molecule_smiles": "str", "model": "str", "melting_point_C": "float | None"},
         "Estimate"),
    ):
        registry.register(OperatorSpec(id=f"chemflow.{name}", version=1, inputs={},
            output=output, parameters=parameters,
            semantics="Model-defined aqueous solubility logS (ESOL/GSE); empirical estimate "
                      "of experimental solubility, with applicability evidence."))
        registry.register(ImplementationSpec(id=name, version="1", target=name, adapter=adapter,
            implements=name,
            backend_kind="cpu_numeric", execution_role="classical", params=frozenset(parameters),
            resource_model=resources))
        registry.register(RefinementRule(target=name, template_id=name, relation="equivalent"))


def base_plan(record, active=None):
    parameters = record.spec.target.model_dump(exclude={"kind", "operator"})
    with build.program("aqueous-solubility") as builder:
        control = builder.node("empirical_estimate", **parameters)
        reference = builder.node("empirical_estimate", **parameters)
        builder.ir.targets = [control.id]
        builder.ir.references = [reference.id]
        return builder.to_ir(), [control.id]


def identity(plan, record, contexts):
    from chemflow.verify.identity_diff import plan_hints

    expected = record.spec.target.model_dump(exclude={"kind", "operator"})
    nodes = {node.id: node for node in plan.nodes}
    return [Diagnostic(code="CF-E120", node=identifier,
                       message="Aqueous solubility model or molecule differs from frozen task",
                       fix_hints=plan_hints(plan, base_plan(record)[0], identifier))
            for identifier in set(plan.targets + plan.references)
            if identifier not in nodes or nodes[identifier].op != "empirical_estimate" or
            nodes[identifier].params != expected or nodes[identifier].inputs]
