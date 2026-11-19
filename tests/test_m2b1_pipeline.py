"""Focused M2b.1 pipeline invariants."""

from chemflow import build
from chemflow.check import check
from chemflow.check.static import infer_context
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task


def test_sector_prep_samples_inherit_hamiltonian_context():
    with build.program("typed-sampling") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"), model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        prep = builder.node("prepare_state", hamiltonian=hamiltonian, sector=sector)
        samples = builder.node("sample_configurations", prep=prep, shots=32, seed=4)
        candidate = builder.ground_energy(hamiltonian, sector, samples=samples,
                                          n_batches=1, max_iter=1)
        builder.refine(candidate, "sqd")
        builder.bind(candidate, "sqd_sector")
    program = builder.to_ir()
    registry = molecular_registry()
    contexts, issues = infer_context(program, registry)
    assert not issues
    assert contexts[samples.id]["hamiltonian_id"] == contexts[prep.id]["hamiltonian_id"]
    assert check(program, registry, task=freeze_task(program)) == []
    results = Executor(registry).run(program)
    assert set(results["costs"]) == {node.id for node in program.nodes}
    assert results["costs"][samples.id].sim_shots == 32
    assert results["costs"][candidate.id].sim_shots in (None, 0)
