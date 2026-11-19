from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation
from chemflow.runtime import Executor
from chemflow.runtime.executor import _check_relation_consistency
from chemflow.units import Q


def test_exact_expectations_of_distinct_preparations_do_not_conflict():
    with build.program("r42") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        reference = builder.ground_energy(hamiltonian, sector)
        builder.refine(reference, "fci")
        builder.bind(reference, "fci_pyscf")
    result = Executor(molecular_registry()).run(builder.to_ir())[reference.id]
    left = result.model_copy(update={"target": "left", "relation": TargetRelation.equivalent})
    right = result.model_copy(update={"target": "right", "relation": TargetRelation.equivalent,
                                      "value": Q(result.value.value + 1, "hartree")})
    from chemflow.ir.schema import Node

    nodes = {"left": Node(id="left", op="expectation", inputs={
                 "operator": "encoded", "prep": "prepared_a"}),
             "right": Node(id="right", op="expectation", inputs={
                 "operator": "encoded", "prep": "prepared_b"})}
    _check_relation_consistency({"left": left, "right": right}, nodes)
