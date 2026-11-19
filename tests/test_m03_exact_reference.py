from chemflow import build
from chemflow.check import check
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.units import Q


def test_abs_error_rejects_approximate_reference(monkeypatch) -> None:
    with build.program("reference") as builder:
        hamiltonian = builder.hamiltonian(
            builder.molecule("H 0 0 0; H 0 0 0.74"),
            builder.electronic_model(basis="sto-3g"),
        )
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        reference = builder.ground_energy(hamiltonian, sector)
        builder.refine(reference, "hf")
        builder.bind(reference, "hf")
        candidate = builder.ground_energy(hamiltonian, sector)
        builder.accept(candidate, abs_error_le=Q(1, "hartree"), reference=reference.id)
    program = builder.to_ir()
    assert "CF-E050" in {issue.code for issue in check(
        program, molecular_registry(), task=freeze_task(program))}
    monkeypatch.setattr("chemflow.runtime.executor.check", lambda *args, **kwargs: [])
    verdict = Executor(molecular_registry()).run(program, task=freeze_task(program))["gates"][0]
    assert not verdict.passed
    assert "exact reference" in verdict.reason
