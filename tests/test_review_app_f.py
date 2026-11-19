"""Reviewer tests for APP batch F: deployable VQE through QPU Pauli measurement (written by
Claude; implementers must not edit).

The platform the user described runs VQE on quantum hardware: UCCSD, 8–30 Pauli terms,
COBYLA, 1024 shots, and ideal or noise models. Under R130, ChemFlow's statevector VQE is
verification-only. VQE becomes deployable once its energy is measured by sampling rotated
circuits on the QPU. Plan: docs/research/09-application-operators.md, batch F.

Conventions:
- Pauli word letter i acts on qubit i, little-endian as in qiskit and ChemFlow's
  `QubitOperator`.
- Count keys are qiskit bitstrings: qubit 0 is the rightmost character.

Contract:
- **`chemflow.quantum.pauli`.**
  - `qwc_groups(terms)`: groups the non-identity words; within a group every pair is
    qubit-wise commuting; every word is in exactly one group. Each group has a measurement
    basis string over {X, Y, Z, I}.
  - `rotation_circuit(basis)`: a qiskit circuit mapping that basis to Z. X → H; Y → S†
    then H.
  - `estimate(terms, groups, counts, readout=None)`: `counts[k]` holds the counts measured
    after group k's rotation. Returns `{energy, stat_err, per_term}`. The identity
    coefficient is added exactly.
  - `readout` = {qubit: (p01, p10)} applies tensor-product readout mitigation: the
    measured distribution is multiplied by the inverse of each qubit's confusion matrix
    [[1−p01, p10], [p01, 1−p10]] before the expectations are taken. Here p01 is the
    probability that 0 is read as 1. This is exact for that noise model, including
    asymmetric flips; a simple 1/(1−p01−p10) rescaling is not.
- **VQE binding.** `ground_energy` refinement "vqe_sampled" with params ansatz (uccsd |
  lucj), optimizer (cobyla | spsa), max_evaluations, shots_per_group, seed and sampler.
  - Bindings, with their R130 roles:
    - `vqe_lqcloud` (role qpu; QPU_CATALOG lists it for ground_energy, with MQ02 limits);
    - `vqe_aer_mq02` (role emulator; Aer MQ02 noise model);
    - `vqe_shots_ideal` (role emulator; ideal shot sampling).
  - The relation is stochastic_estimate, proved. Diagnostics carry stat_err, evaluations
    and groups.
  - The resource model forecasts evaluations × groups × shots_per_group as qpu_shots (QPU
    binding) or sim_shots (emulators).
  - check(purpose="deploy") accepts all three bindings; "production" accepts only
    vqe_lqcloud.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")


def _h2_terms():
    from chemflow import build
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    with build.program("h2") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                          builder.electronic_model(basis="sto-3g"))
        encoded = builder.node("encode", hamiltonian=hamiltonian)
    computed = Executor(molecular_registry()).run(builder.to_ir())
    value = computed[encoded.id]
    operator = value[1] if isinstance(value, tuple) else value
    return dict(operator.terms)


def _qwc(a, b):
    return all(x == y or "I" in (x, y) for x, y in zip(a, b, strict=True))


def test_f1_grouping_is_a_qubit_wise_commuting_partition() -> None:
    from chemflow.quantum.pauli import qwc_groups

    terms = _h2_terms()
    words = {word for word in terms if set(word) != {"I"}}
    groups = qwc_groups(terms)
    seen = [word for group in groups for word in group["terms"]]
    assert sorted(seen) == sorted(words)
    for group in groups:
        for a, b in itertools.combinations(group["terms"], 2):
            assert _qwc(a, b)
        for word in group["terms"]:
            assert all(letter in ("I", basis) for letter, basis in zip(word, group["basis"],
                                                                         strict=True))


def _exact_counts(state, basis, total=10 ** 9):
    from qiskit.quantum_info import Statevector

    from chemflow.quantum.pauli import rotation_circuit

    rotated = Statevector(state).evolve(rotation_circuit(basis))
    probabilities = rotated.probabilities_dict()
    return {key: round(value * total) for key, value in probabilities.items()}


def _exact_energy(terms, state):
    from qiskit.quantum_info import SparsePauliOp

    operator = SparsePauliOp.from_list([(word[::-1], coefficient)
                                        for word, coefficient in terms.items()])
    return float(np.real(np.vdot(state, operator.to_matrix() @ state)))


def test_f1_estimates_from_rotated_counts_reproduce_the_expectation() -> None:
    from chemflow.quantum.pauli import estimate, qwc_groups

    terms = _h2_terms()
    rng = np.random.default_rng(3)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    groups = qwc_groups(terms)
    counts = [_exact_counts(state, group["basis"]) for group in groups]
    result = estimate(terms, groups, counts)
    assert result["energy"] == pytest.approx(_exact_energy(terms, state), abs=1e-6)


def test_f1_tensor_readout_mitigation_recovers_the_expectation() -> None:
    from chemflow.quantum.pauli import estimate, qwc_groups

    terms = _h2_terms()
    rng = np.random.default_rng(5)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    flips = {0: (0.02, 0.05), 1: (0.03, 0.01), 2: (0.04, 0.02), 3: (0.01, 0.03)}
    groups = qwc_groups(terms)
    noisy = []
    for group in groups:
        probabilities = _exact_counts(state, group["basis"], total=10 ** 12)
        distribution = np.zeros(16)
        for key, value in probabilities.items():
            distribution[int(key, 2)] = value / 10 ** 12
        for qubit, (p01, p10) in flips.items():  # p01: 0 read as 1; p10: 1 read as 0
            matrix = np.array([[1 - p01, p10], [p01, 1 - p10]])
            reshaped = distribution.reshape([2] * 4)
            axis = 3 - qubit  # bitstring index order: qubit 3 is the first character
            reshaped = np.moveaxis(np.tensordot(matrix, np.moveaxis(reshaped, axis, 0), axes=1),
                                   0, axis)
            distribution = reshaped.reshape(16)
        noisy.append({format(index, "04b"): round(value * 10 ** 12)
                      for index, value in enumerate(distribution) if value > 0})
    raw = estimate(terms, groups, noisy)["energy"]
    mitigated = estimate(terms, groups, noisy, readout=flips)["energy"]
    exact = _exact_energy(terms, state)
    assert abs(raw - exact) > 1e-3
    assert mitigated == pytest.approx(exact, abs=1e-6)


def _vqe_plan(binding, shots=4096, evaluations=60):
    from chemflow import build

    with build.program("vqe") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                          builder.electronic_model(basis="sto-3g"))
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        energy = builder.ground_energy(hamiltonian, sector, s=0)
        builder.refine(energy, "vqe_sampled", ansatz="uccsd", optimizer="cobyla",
                       max_evaluations=evaluations, shots_per_group=shots, seed=7)
        builder.bind(energy, binding)
    program = builder.to_ir()
    program.targets = [energy.id]
    return program, energy.id


def test_f2_sampled_vqe_on_the_ideal_emulator_reaches_chemical_accuracy_statistically() -> None:
    from pyscf import fci, gto, scf

    from chemflow.domains import load_registry, pack_names
    from chemflow.runtime import Executor

    registry = load_registry(pack_names())
    program, node = _vqe_plan("vqe_shots_ideal", shots=20000, evaluations=80)
    result = Executor(registry).run(program)[node]
    exact = fci.FCI(scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                                  verbose=0)).run()).kernel()[0]
    stat_err = result.diagnostics["stat_err"]
    assert stat_err > 0
    assert abs(result.value.to("hartree").value - exact) <= max(0.005, 4 * stat_err)
    assert result.relation.value == "stochastic_estimate"
    assert result.diagnostics.get("relation_status") == "proved"


def test_f2_deployment_roles_and_resource_forecast() -> None:
    from chemflow.backends import QPU_CATALOG
    from chemflow.check.static import check
    from chemflow.domains import load_registry, pack_names

    registry = load_registry(pack_names())
    assert registry.implementations["vqe_lqcloud"].execution_role == "qpu"
    assert "vqe_lqcloud" in QPU_CATALOG.get("ground_energy", {})
    for binding, production_ok in (("vqe_lqcloud", True), ("vqe_aer_mq02", False),
                                   ("vqe_shots_ideal", False)):
        program, _node = _vqe_plan(binding)
        assert not [i for i in check(program, registry, purpose="deploy") if i.code == "CF-E140"]
        production = [i for i in check(program, registry, purpose="production")
                      if i.code == "CF-E140"]
        assert (not production) == production_ok, binding
    implementation = registry.implementations["vqe_lqcloud"]
    forecast = implementation.resource_model(max_evaluations=60, shots_per_group=1024,
                                             n_groups=5)
    assert forecast["qpu_shots"] == 60 * 5 * 1024
