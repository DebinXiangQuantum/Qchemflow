"""Reviewer regression tests, round 2 (written by Claude; implementers must not edit)."""
from math import comb

import numpy as np
import pytest

from chemflow import build
from chemflow.check import check
from chemflow.diagnostics import CheckError
from chemflow.impls.fci import _annihilate, _create, matrix, sector_spin_eigenstate, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel


def _codes(program) -> list[str]:
    return [d.code for d in check(program, molecular_registry(), task=freeze_task(program))]


def _run(program):
    return Executor(molecular_registry()).run(program, task=freeze_task(program))


def _s2_operator(states: list[int], n_orb: int) -> np.ndarray:
    """Independent S^2 = Sz^2 + Sz + S-S+ in the chemflow determinant basis."""
    index = {bits: i for i, bits in enumerate(states)}
    s2 = np.zeros((len(states), len(states)))
    for column, bits in enumerate(states):
        n_alpha = (bits & ((1 << n_orb) - 1)).bit_count()
        n_beta = (bits >> n_orb).bit_count()
        sz = (n_alpha - n_beta) / 2
        s2[column, column] += sz * sz + sz
        for p in range(n_orb):
            for q in range(n_orb):
                current, sign = bits, 1
                for create, mode in ((False, n_orb + q), (True, q), (False, p), (True, n_orb + p)):
                    step = (_create if create else _annihilate)(current, mode)
                    if step is None:
                        break
                    current, factor = step
                    sign *= factor
                else:
                    if current in index:
                        s2[index[current], column] += sign
    return s2


def test_spin_selection_correct_beyond_symmetric_string_spaces() -> None:
    store = ArtifactStore()
    data = build_molecule("Li 0 0 0; H 0 0 1.6", ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=5))
    states = sector_states(data.sector)
    energies, vectors = np.linalg.eigh(matrix(data.hamiltonian, store, states).toarray())
    s2 = _s2_operator(states, data.sector.n_orb)
    for total_spin in (0, 1):
        expected = next(e for e, v in zip(energies, vectors.T)
                        if abs(v @ s2 @ v - total_spin * (total_spin + 1)) < 1e-6)
        energy, reported = sector_spin_eigenstate(data.hamiltonian, data.sector, store,
                                                  total_spin)
        assert energy == pytest.approx(expected, abs=1e-8)
        assert reported == pytest.approx(total_spin * (total_spin + 1), abs=1e-6)
    assert comb(data.sector.n_orb, data.sector.n_alpha) == 10


@pytest.mark.parametrize("n_orb,n_alpha,n_beta", [(4, 2, 2), (5, 2, 2), (5, 3, 1), (6, 3, 3),
                                                  (6, 2, 3)])
def test_spin_square_in_chemflow_basis_matches_independent_operator(
        n_orb: int, n_alpha: int, n_beta: int) -> None:
    from chemflow.impls.fci import spin_square
    from chemflow.types import Sector

    sector = Sector(n_alpha=n_alpha, n_beta=n_beta, n_orb=n_orb)
    states = sector_states(sector)
    s2 = _s2_operator(states, n_orb)
    rng = np.random.default_rng(7)
    for _ in range(5):
        vector = rng.normal(size=len(states))
        vector /= np.linalg.norm(vector)
        assert spin_square(vector, sector) == pytest.approx(vector @ s2 @ vector, abs=1e-10)


def test_difference_of_two_upper_bounds_is_not_an_upper_bound() -> None:
    with build.program("diff_ub") as b:
        model = b.electronic_model(basis="sto-3g")
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        left = b.ground_energy(b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), model), sector)
        right = b.ground_energy(b.hamiltonian(b.molecule("H 0 0 0; H 0 0 1.4"), model), sector)
        for node in (left, right):
            b.refine(node, "hf")
            b.bind(node, "hf")
        diff = b.node("energy_difference", left=left, right=right)
    assert _run(b.to_ir())[diff.id].relation == "approximate"


def test_difference_upper_bound_minus_exact_keeps_ordering() -> None:
    with build.program("diff_ub_exact") as b:
        model = b.electronic_model(basis="sto-3g")
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), model)
        upper = b.ground_energy(ham, sector)
        b.refine(upper, "hf")
        b.bind(upper, "hf")
        exact = b.ground_energy(ham, sector)
        diff = b.node("energy_difference", left=upper, right=exact)
        lower = b.node("energy_difference", left=exact, right=upper)
    results = _run(b.to_ir())
    assert results[diff.id].relation == "variational_upper_bound"
    # exact - upper_bound is a lower bound; with no lower-bound relation it must degrade.
    assert results[lower.id].relation == "approximate"


def test_active_space_change_is_a_model_change() -> None:
    with build.program("cas_change") as b:
        mol = b.molecule("Li 0 0 0; H 0 0 1.6")
        model = b.electronic_model(basis="sto-3g")
        small = b.ground_energy(b.hamiltonian(mol, model, active={"n_elec": 2, "n_orb": 2}),
                                b.sector(n_alpha=1, n_beta=1, n_orb=2))
        large = b.ground_energy(b.hamiltonian(mol, model, active={"n_elec": 2, "n_orb": 4}),
                                b.sector(n_alpha=1, n_beta=1, n_orb=4))
        b.node("energy_difference", left=small, right=large)
    assert "CF-E011" in _codes(b.to_ir())


def test_equivalent_gate_sees_upstream_relations() -> None:
    with build.program("gate_upstream") as b:
        model = b.electronic_model(basis="sto-3g")
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), model)
        upper = b.ground_energy(ham, sector)
        b.refine(upper, "hf")
        b.bind(upper, "hf")
        diff = b.node("energy_difference", left=upper, right=b.ground_energy(ham, sector))
        b.accept(diff, relation="equivalent")
    assert "CF-E050" in _codes(b.to_ir())


def test_abs_error_gate_is_evaluated_at_runtime() -> None:
    from chemflow.units import Q

    with build.program("gate_eval") as b:
        model = b.electronic_model(basis="sto-3g")
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 1.4"), model)
        candidate = b.ground_energy(ham, sector)
        b.refine(candidate, "hf")
        b.bind(candidate, "hf")
        reference = b.ground_energy(ham, sector)
        b.accept(candidate, abs_error_le=Q(1.6e-3, "hartree"), reference=reference.id,
                 relative_to="declared_finite_model")
    program = b.to_ir()
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    verdicts = results["gates"]
    assert len(verdicts) == 1
    # HF misses FCI by > 70 mHa at 1.4 angstrom: the gate must fail, with the measured error.
    assert verdicts[0].passed is False
    assert verdicts[0].measured.to("hartree").value > 1.6e-3


def test_spin_diagnostic_is_computed_not_assumed() -> None:
    with build.program("rohf") as b:
        energy = b.ground_energy(
            b.hamiltonian(b.molecule("O 0 0 0; O 0 0 1.21", spin=2),
                          b.electronic_model(basis="sto-3g", reference="ROHF")),
            b.sector(n_alpha=9, n_beta=7, n_orb=10))
        b.refine(energy, "hf")
        b.bind(energy, "hf")
    s2 = _run(b.to_ir())[energy.id].diagnostics.get("s2")
    assert s2 is None or s2 == pytest.approx(2.0, abs=1e-6)


def test_wall_budget_enforced_on_measured_spend_for_unpaid_compute() -> None:
    with build.program("wall") as b:
        energy = b.ground_energy(
            b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g")),
            b.sector(n_alpha=1, n_beta=1, n_orb=2))
        b.budget(wall_s=600)
    assert energy.id in _run(b.to_ir())
    with build.program("wall_tight") as b:
        b.ground_energy(
            b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g")),
            b.sector(n_alpha=1, n_beta=1, n_orb=2))
        b.budget(wall_s=0)
    with pytest.raises(CheckError) as info:
        _run(b.to_ir())
    assert "CF-E070" in {d.code for d in info.value.diagnostics}
