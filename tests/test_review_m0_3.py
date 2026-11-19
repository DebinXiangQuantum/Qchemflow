"""Reviewer regression tests, round 3 (written by Claude; implementers must not edit)."""
import pytest

from chemflow import build
from chemflow.check import check
from chemflow.diagnostics import CheckError
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.units import Q


def _codes(program) -> list[str]:
    return [d.code for d in check(program, molecular_registry(), task=freeze_task(program))]


def _run(program):
    return Executor(molecular_registry()).run(program, task=freeze_task(program))


def _h2(builder, r: float = 0.74):
    model = builder.electronic_model(basis="sto-3g")
    sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
    return builder.hamiltonian(builder.molecule(f"H 0 0 0; H 0 0 {r}"), model), sector


def test_abs_error_gate_requires_exact_reference() -> None:
    with build.program("approx_reference") as b:
        ham, sector = _h2(b)
        reference = b.ground_energy(ham, sector)
        b.refine(reference, "hf")
        b.bind(reference, "hf")
        candidate = b.ground_energy(ham, sector)
        b.refine(candidate, "hf")
        b.bind(candidate, "hf")
        b.accept(candidate, abs_error_le=Q(1.6e-3, "hartree"), reference=reference.id)
    assert "CF-E050" in _codes(b.to_ir())


def _sampled_gate_program(epsilon: float):
    with build.program("stochastic_gate") as b:
        ham, _ = _h2(b)
        encoding = b.node("encode", hamiltonian=ham)
        prep = b.node("prepare", hamiltonian=ham, encoding=encoding)
        sampled = b.node("sample", operator=encoding, prep=prep, shots=20, seed=1)
        sampled.effects = "random"
        exact = b.node("expectation", operator=encoding, prep=prep)
        b.accept(sampled, abs_error_le=Q(epsilon, "hartree"), reference=exact.id)
    return b.to_ir(), sampled.id, exact.id


def test_stochastic_gate_accounts_for_statistical_error() -> None:
    program, sampled, exact = _sampled_gate_program(1.0)
    results = _run(program)
    estimate = results[sampled]
    sigma = estimate.uncertainty.stat_err
    delta = abs(estimate.value.value - results[exact].value.value)
    assert sigma and sigma > 0
    # Point error within epsilon, but not within the default 95% two-sided margin.
    program, sampled, _ = _sampled_gate_program(delta + 1.0 * sigma)
    tight = _run(program)["gates"][0]
    assert tight.passed is False
    program, sampled, _ = _sampled_gate_program(delta + 2.5 * sigma)
    assert _run(program)["gates"][0].passed is True


def test_relation_gate_is_a_minimum_strength() -> None:
    with build.program("strength") as b:
        ham, sector = _h2(b)
        exact = b.ground_energy(ham, sector)
        b.accept(exact, relation="variational_upper_bound")
        upper = b.ground_energy(ham, sector)
        b.refine(upper, "hf")
        b.bind(upper, "hf")
        b.accept(upper, relation="variational_upper_bound")
    verdicts = {v.node: v.passed for v in _run(b.to_ir())["gates"]}
    assert verdicts[exact.id] is True
    assert verdicts[upper.id] is True
    # Revised in M2a review (R31): a statically certain gate failure is a CF-E050 error for
    # every node kind, so a stochastic sample cannot reach an upper-bound gate at runtime.
    with build.program("strength_static") as b:
        ham, _ = _h2(b)
        encoding = b.node("encode", hamiltonian=ham)
        prep = b.node("prepare", hamiltonian=ham, encoding=encoding)
        sampled = b.node("sample", operator=encoding, prep=prep, shots=100, seed=3)
        sampled.effects = "random"
        b.accept(sampled, relation="variational_upper_bound")
    assert "CF-E050" in _codes(b.to_ir())


def test_invalid_active_space_is_a_diagnostic_not_a_crash() -> None:
    with build.program("bad_active") as b:
        b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"),
                      active={"n_orb": 2})
    found = _codes(b.to_ir())
    assert {"CF-E020", "CF-E080"} & set(found)


def test_cpu_budget_is_enforced_on_measured_spend() -> None:
    with build.program("cpu") as b:
        ham, sector = _h2(b)
        b.ground_energy(ham, sector)
        b.budget(cpu_core_s=0)
    with pytest.raises(CheckError) as info:
        _run(b.to_ir())
    assert "CF-E070" in {d.code for d in info.value.diagnostics}
