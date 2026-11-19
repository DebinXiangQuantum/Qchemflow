import numpy as np
import pytest

from chemflow.impls.fci import sector_spin_eigenstate, sector_states
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sqd import sqd_sector
from chemflow.runtime import ArtifactStore
from chemflow.types import ActiveSpaceSpec, ElectronicModel, Samples


def test_spin_closed_strings():
    from chemflow.impls.sqd import _spin_closed
    from chemflow.types import Sector

    sector = Sector(n_alpha=1, n_beta=1, n_orb=2)
    assert _spin_closed({1 | (2 << 2)}, sector) == [5, 6, 9, 10]


def test_recovery():
    from chemflow.impls.sqd import _fix_spin_bits

    bits = np.array([0, 0, 1, 0], dtype=np.uint8)
    _fix_spin_bits(bits, 2, np.ones(4), np.random.default_rng(0))
    assert sum(bits) == 2


@pytest.mark.parametrize("atoms,active,spin", [
    ("H 0 0 0; H 0 0 0.74", None, 0),
    ("O 0 0 0; O 0 0 1.21", ActiveSpaceSpec(n_elec=2, n_orb=2), 0),
])
def test_full_sector_matches_spin_fci(atoms, active, spin):
    store = ArtifactStore()
    data = build_molecule(atoms, ElectronicModel(basis="sto-3g"), store, active=active)
    states = sector_states(data.sector)
    samples = Samples(counts={str(state): 2 for state in states}, shots=2 * len(states),
                      encoding_id="test", basis="computational",
                      source="quantum_simulator", seed=3)
    energy, info = sqd_sector(data, data.sector, store, samples, seed=3,
                              n_batches=1, total_spin=spin)
    reference, _ = sector_spin_eigenstate(data.hamiltonian, data.sector, store, spin)
    assert energy == pytest.approx(reference, abs=1e-8)
    assert info["s2"] == pytest.approx(0, abs=1e-5)


def test_noisy_recovery_is_variational():
    from chemflow.impls.statevector import sample_configurations

    store = ArtifactStore()
    data = build_molecule("H 0 0 0; H 0 0 0.74", ElectronicModel(basis="sto-3g"), store)
    state = np.zeros(4)
    state[0] = 1
    samples = sample_configurations(state, 200, 5, 0.05, sector=data.sector)
    energy, info = sqd_sector(data, data.sector, store, samples, n_batches=2)
    reference, _ = sector_spin_eigenstate(data.hamiltonian, data.sector, store)
    assert energy >= reference - 1e-9
    assert info["subspace_dimension"] > 0


def test_sqd_cannot_claim_equivalence():
    from chemflow import build
    from chemflow.check import check
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import freeze_task

    with build.program("sqd-relation") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        candidate = builder.ground_energy(hamiltonian, sector)
        builder.refine(candidate, "sqd")
        builder.bind(candidate, "sqd_sector")
        builder.accept(candidate, relation="equivalent")
    program = builder.to_ir()
    assert "CF-E050" in {issue.code for issue in check(
        program, molecular_registry(), task=freeze_task(program))}


def test_sqd_graph_preserves_variational_relation():
    from chemflow import build
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    with build.program("sample-then-sqd") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        samples = builder.node("sample_configurations", prep=prep, shots=120, seed=9,
                               readout_flip_p=0.05)
        candidate = builder.ground_energy(hamiltonian, sector, samples=samples,
                                          n_batches=2, s=0)
        builder.refine(candidate, "sqd")
        builder.bind(candidate, "sqd_sector")
        reference = builder.ground_energy(hamiltonian, sector, s=0)
    results = Executor(molecular_registry()).run(builder.to_ir())
    assert results[candidate.id].relation == "variational_upper_bound"
    assert results[candidate.id].uncertainty.stat_err is None
    assert results[candidate.id].value.value >= results[reference.id].value.value - 1e-9
    assert results[samples.id].shots == 120
    assert results["costs"][samples.id].sim_shots == 120
    assert results["costs"][candidate.id].sim_shots in (None, 0)
    assert results[candidate.id].cost.sim_shots in (None, 0)
