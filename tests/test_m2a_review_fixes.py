from chemflow.check.static import check
from chemflow.ir.builder import Builder
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime.task import freeze_task


def test_static_gate_rejects_non_equivalent_relation_requirement():
    builder = Builder("r27")
    molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
    model = builder.electronic_model(basis="sto-3g")
    hamiltonian = builder.hamiltonian(molecule, model)
    sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
    energy = builder.ground_energy(hamiltonian, sector)
    builder.refine(energy, "ccsd")
    builder.bind(energy, "ccsd")
    builder.accept(energy, relation="variational_upper_bound")
    program = builder.to_ir()
    assert "CF-E050" in [issue.code for issue in check(program, molecular_registry(),
                                                      freeze_task(program))]


def test_statistical_error_uses_candidate_unit():
    from chemflow.context import Context
    from chemflow.relations import TargetRelation
    from chemflow.runtime.executor import Executor
    from chemflow.types import (
        BackendKind,
        CostRecord,
        Estimate,
        Provenance,
        QuantityKind,
        Uncertainty,
    )
    from chemflow.units import Q

    registry = molecular_registry()
    builder = Builder("r28")
    molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
    model = builder.electronic_model(basis="sto-3g")
    hamiltonian = builder.hamiltonian(molecule, model)
    sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
    candidate = builder.ground_energy(hamiltonian, sector)
    reference = builder.ground_energy(hamiltonian, sector)
    builder.accept(candidate, abs_error_le=Q(0.05, "eV"), reference=reference.id)
    program = builder.to_ir()
    executor = Executor(registry)
    provenance = Provenance(impl_id="test", impl_version="1", backend_kind=BackendKind.cpu_numeric,
                            device="local")
    original = executor.run(program, freeze_task(program))[candidate.id]
    stochastic = Estimate(value=Q(original.value.to("eV").value, "eV"),
                          kind=QuantityKind.total_electronic_energy, target=candidate.id,
                          relation=TargetRelation.stochastic_estimate,
                          uncertainty=Uncertainty(stat_err=0.03), context=Context(),
                          provenance=provenance, cost=CostRecord())
    key = next(key for key, value in executor.cache.items() if value is original)
    executor.cache[key] = stochastic
    verdict = executor.run(program, freeze_task(program))["gates"][0]
    assert not verdict.passed
    assert abs(verdict.stat_err - 0.03) < 1e-12
