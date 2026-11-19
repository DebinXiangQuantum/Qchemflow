"""Reviewer regression tests for A1a (written by Claude; implementers must not edit).

A1a opens the method layer into primitives with defined meaning, so an agent can assemble an
algorithm no template provides — first case: the real-axis spectral function of the
single-impurity Anderson model (SQD in N, N±1 sectors + Lehmann sum; user experiment
experiments/green_function_demo.py). Oracles are a full Fock-space exact diagonalisation
written in this file.

Contract (A1a brief): pack `dynamics` with operator `impurity_model` and task kind
`spectral_function`; pack `algorithms` with operators `full_subspace`, `subspace_from_samples`,
`extend_subspace`, `rayleigh_ritz`; operator `spectral_function` with refinements
`exact_lehmann` (binding `spectral_exact`) and `lehmann` (binding `lehmann_from_spectra`);
gate kind `l1_error_vs_exact`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.runtime import Executor

AIM = {"U": 2.0, "eps_bath": [-1.2, 0.0, 1.2], "t_bath": [0.5, 0.6, 0.5], "unit": "hartree"}
OMEGA = {"start": -4.0, "stop": 4.0, "n": 801, "unit": "hartree"}
ETA = 0.1
PACKS = ["molecular", "algorithms", "dynamics"]
# The accuracy bar is part of the problem specification (A1c, user decision 2026-10-12):
# it is declared once here, used by the task gate and program gates, and read by the
# assertions below. The L1 integral of A(omega) over omega is dimensionless.
ACCURACY = {"kind": "l1_error_vs_exact", "threshold": 0.1, "unit": "dimensionless"}


def _accept(program, node: str, accuracy: dict = ACCURACY):
    """State the bar in the program (the language's `accept`). It names no reference: the
    verifier brings the independent exact one."""
    from chemflow.ir import AcceptanceGate
    from chemflow.units import Q

    measure = {"l1_error_vs_exact": "l1_error_le", "abs_error_vs_exact": "abs_error_le"}
    program.gates.append(AcceptanceGate(node=node, **{
        measure[accuracy["kind"]]: Q(accuracy["threshold"], accuracy["unit"])}))
    return program


# --- independent oracle: full Fock space, Jordan–Wigner ordering alpha 0..M-1, beta M..2M-1 --

def _fock(model: dict):
    m = 1 + len(model["eps_bath"])
    h1 = np.zeros((m, m))
    h1[0, 0] = -model["U"] / 2
    for index, (energy, hop) in enumerate(zip(model["eps_bath"], model["t_bath"])):
        h1[index + 1, index + 1] = energy
        h1[0, index + 1] = h1[index + 1, 0] = hop
    n = 2 * m
    dim = 1 << n
    annihilate = []
    for mode in range(n):
        op = np.zeros((dim, dim))
        for state in range(dim):
            if state >> mode & 1:
                sign = (-1) ** (state & ((1 << mode) - 1)).bit_count()
                op[state ^ (1 << mode), state] = sign
        annihilate.append(op)
    create = [op.T for op in annihilate]
    hamiltonian = np.zeros((dim, dim))
    for p, q in itertools.product(range(m), repeat=2):
        if h1[p, q]:
            for spin in (0, m):
                hamiltonian += h1[p, q] * create[p + spin] @ annihilate[q + spin]
    hamiltonian += model["U"] * (create[0] @ annihilate[0]) @ (create[m] @ annihilate[m])
    return m, hamiltonian, create, annihilate


def _sector(m: int, n_alpha: int, n_beta: int) -> list[int]:
    alpha, beta = (1 << m) - 1, ((1 << m) - 1) << m
    return [s for s in range(1 << (2 * m))
            if (s & alpha).bit_count() == n_alpha and (s & beta).bit_count() == n_beta]


def _oracle(model: dict, n_alpha: int, n_beta: int):
    m, hamiltonian, create, annihilate = _fock(model)
    ground_sector = _sector(m, n_alpha, n_beta)
    energies, vectors = np.linalg.eigh(hamiltonian[np.ix_(ground_sector, ground_sector)])
    e0 = energies[0]
    psi = np.zeros(hamiltonian.shape[0])
    psi[ground_sector] = vectors[:, 0]
    omega = np.linspace(OMEGA["start"], OMEGA["stop"], OMEGA["n"])
    green = np.zeros_like(omega, dtype=complex)
    for operator, sector, sign in ((create[0], _sector(m, n_alpha + 1, n_beta), 1),
                                   (annihilate[0], _sector(m, n_alpha - 1, n_beta), -1)):
        excited = (operator @ psi)[sector]
        values, states = np.linalg.eigh(hamiltonian[np.ix_(sector, sector)])
        weights = np.abs(states.T @ excited) ** 2
        poles = sign * (values - e0)
        green += (weights[None, :] / (omega[:, None] - poles[None, :] + 1j * ETA)).sum(axis=1)
    return e0, omega, -green.imag / np.pi


def _l1(omega, left, right) -> float:
    return float(np.trapezoid(np.abs(np.asarray(left) - np.asarray(right)), omega))


# --- the model and the exact reference -------------------------------------------------------

def test_impurity_model_ground_state_matches_the_independent_oracle() -> None:
    from chemflow.domains import load_registry

    with build.program("aim") as builder:
        model = builder.node("impurity_model", **AIM)
        sector = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
        energy = builder.ground_energy(model, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "fci_pyscf")
    estimate = Executor(load_registry(PACKS)).run(builder.to_ir())[energy.id]
    assert estimate.value.to("hartree").value == pytest.approx(_oracle(AIM, 2, 2)[0], abs=1e-9)


def test_exact_spectral_reference_matches_the_independent_lehmann_sum() -> None:
    from chemflow.domains import load_registry

    with build.program("aim-exact") as builder:
        model = builder.node("impurity_model", **AIM)
        sector = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
        spectrum = builder.node("spectral_function", hamiltonian=model, sector=sector,
                                orbital=0, spin="alpha", omega=OMEGA, eta=ETA)
        builder.refine(spectrum, "exact_lehmann")
        builder.bind(spectrum, "spectral_exact")
    result = Executor(load_registry(PACKS)).run(builder.to_ir())[spectrum.id]
    _, omega, oracle = _oracle(AIM, 2, 2)
    np.testing.assert_allclose(result.omega, omega, atol=1e-12)
    assert _l1(omega, result.spectral, oracle) < 1e-8
    assert result.relation == "equivalent"
    assert result.diagnostics["sum_rule"] == pytest.approx(
        float(np.trapezoid(oracle, omega)), abs=1e-8)


# --- primitives: existing methods are compositions; relations are inferred -------------------

def _designed_plan(shots: int = 20000, closure: str = "singles", full: bool = False):
    """What an agent would assemble from primitives: SQD in N, then N±1 by ladder operators."""
    with build.program("aim-designed") as builder:
        model = builder.node("impurity_model", **AIM)
        n = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
        plus = builder.sector(n_alpha=3, n_beta=2, n_orb=4)
        minus = builder.sector(n_alpha=1, n_beta=2, n_orb=4)
        if full:
            ground_space = builder.node("full_subspace", sector=n)
        else:
            prep = builder.node("prepare_state", hamiltonian=model, sector=n, ansatz="lucj",
                                init="ccsd")
            samples = builder.node("sample_configurations", prep=prep, shots=shots, seed=7)
            ground_space = builder.node("subspace_from_samples", samples=samples, sector=n)
        ground = builder.node("rayleigh_ritz", hamiltonian=model, subspace=ground_space,
                              n_roots=1)
        spectra = {}
        for name, sector, operator in (("particle", plus, "create"),
                                       ("hole", minus, "annihilate")):
            space = (builder.node("full_subspace", sector=sector) if full else
                     builder.node("extend_subspace", subspace=ground_space, sector=sector,
                                  operator=operator, orbital=0, spin="alpha", closure=closure))
            spectra[name] = builder.node("rayleigh_ritz", hamiltonian=model, subspace=space,
                                         n_roots=8)
        response = builder.node("spectral_function", ground=ground,
                                particle=spectra["particle"], hole=spectra["hole"],
                                orbital=0, spin="alpha", omega=OMEGA, eta=ETA)
        builder.refine(response, "lehmann")
        builder.bind(response, "lehmann_from_spectra")
    return builder.to_ir(), ground.id, response.id


def test_rayleigh_ritz_is_labelled_a_variational_upper_bound_from_any_subspace() -> None:
    from chemflow.domains import load_registry

    program, ground, _ = _designed_plan()
    result = Executor(load_registry(PACKS)).run(program)
    exact = _oracle(AIM, 2, 2)[0]
    spectrum = result[ground]
    assert spectrum.relation == "variational_upper_bound"
    assert spectrum.energies[0] >= exact - 1e-9


def test_full_subspaces_reproduce_the_exact_spectral_function_and_are_labelled_equivalent():
    from chemflow.domains import load_registry

    program, _, response = _designed_plan(full=True)
    result = Executor(load_registry(PACKS)).run(program)[response]
    _, omega, oracle = _oracle(AIM, 2, 2)
    assert _l1(omega, result.spectral, oracle) < 1e-8
    assert result.relation == "equivalent"


def test_sampled_design_is_close_but_only_approximate() -> None:
    """Revised after A1a: this one-layer graph (N-sector samples, then one ladder step with
    singles closure) measures L1 = 0.081; the implementer showed that no non-full subspace of
    this form gets below 0.052. The earlier 0.05 bound wrongly borrowed the user's L1 = 0.004,
    which came from a different algorithm (all three sectors sampled from VQE-optimised LUCJ
    mixtures, eta = 0.08 on [-6, 6])."""
    from chemflow.domains import load_registry

    program, _, response = _designed_plan()
    result = Executor(load_registry(PACKS)).run(program)[response]
    _, omega, oracle = _oracle(AIM, 2, 2)
    assert _l1(omega, result.spectral, oracle) < ACCURACY["threshold"]
    assert result.relation == "approximate"
    assert {"sum_rule", "coverage"} <= set(result.diagnostics)


def _enriched_plan():
    """A new composition of the same primitives: enrich the N-sector space by an
    N -> N+1 -> N ladder walk before building the response spaces."""
    with build.program("aim-enriched") as builder:
        model = builder.node("impurity_model", **AIM)
        n = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
        plus = builder.sector(n_alpha=3, n_beta=2, n_orb=4)
        minus = builder.sector(n_alpha=1, n_beta=2, n_orb=4)
        prep = builder.node("prepare_state", hamiltonian=model, sector=n, ansatz="lucj",
                            init="ccsd")
        samples = builder.node("sample_configurations", prep=prep, shots=20000, seed=7)
        sampled = builder.node("subspace_from_samples", samples=samples, sector=n)
        ground = builder.node("rayleigh_ritz", hamiltonian=model, subspace=sampled, n_roots=1)
        up = builder.node("extend_subspace", subspace=sampled, sector=plus, operator="create",
                          orbital=0, spin="alpha", closure="singles")
        enriched = builder.node("extend_subspace", subspace=up, sector=n,
                                operator="annihilate", orbital=0, spin="alpha",
                                closure="singles")
        spectra = {}
        for name, sector, operator in (("particle", plus, "create"),
                                       ("hole", minus, "annihilate")):
            space = builder.node("extend_subspace", subspace=enriched, sector=sector,
                                 operator=operator, orbital=0, spin="alpha", closure="singles")
            spectra[name] = builder.node("rayleigh_ritz", hamiltonian=model, subspace=space,
                                         n_roots=8)
        response = builder.node("spectral_function", ground=ground,
                                particle=spectra["particle"], hole=spectra["hole"],
                                orbital=0, spin="alpha", omega=OMEGA, eta=ETA)
        builder.refine(response, "lehmann")
        builder.bind(response, "lehmann_from_spectra")
    return builder.to_ir(), response.id


def test_a_new_composition_of_primitives_is_measured_against_the_old_one() -> None:
    from chemflow.domains import load_registry

    registry = load_registry(PACKS)
    _, omega, oracle = _oracle(AIM, 2, 2)
    one_layer, _, first = _designed_plan()
    enriched, second = _enriched_plan()
    baseline = Executor(registry).run(one_layer)[first]
    improved = Executor(registry).run(enriched)[second]
    assert _l1(omega, improved.spectral, oracle) < _l1(omega, baseline.spectral, oracle)
    assert improved.relation == "approximate"  # the ground subspace is still sampled


def test_spectral_function_task_gates_a_designed_algorithm_against_the_exact_reference() -> None:
    from chemflow.domains import get_pack, load_registry

    spec = TaskSpec.model_validate({
        "question": "AIM impurity spectral function", "target": {
            "kind": "spectral_function", "model": {"impurity_model": AIM},
            "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "orbital": 0, "spin": "alpha",
            "omega": OMEGA, "eta": ETA},
        "gates": [ACCURACY],
        "budget": {"wall_s": 600}, "allowed_effects": ["pure", "simulator"], "decision": "d"})
    store = TaskStore("reviewer")
    record = store.freeze(spec, "reviewer")
    base, _ = get_pack("dynamics").base_plan(record)  # exact Lehmann reference
    designed, _, response = _designed_plan()
    plan = base.model_copy(deep=True)
    known = {node.id for node in plan.nodes}
    for node in designed.nodes:  # graft the designed algorithm onto the frozen task's plan
        if node.id not in known:
            plan.nodes.append(node)
    plan.targets.append(response)
    result = Executor(load_registry(PACKS)).run_plan(plan, record, store)
    verdicts = {gate.node: gate for gate in result["gates"]}
    assert verdicts[response].passed
    assert verdicts[plan.references[0]].passed
