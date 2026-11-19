"""Reviewer regression tests for M2b (written by Claude; implementers must not edit)."""
import signal
from pathlib import Path

import pytest
from pyscf import fci

from chemflow import build
from chemflow.cards import load_cards
from chemflow.check import check
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import SectorEngine
from chemflow.impls.vqe import vqe
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import ArtifactStore, Executor, freeze_task
from chemflow.types import ActiveSpaceSpec, ElectronicModel, Samples

H4 = "H 0 0 0; H 0 0 1.55; H 1.55 0 1.55; H 1.55 0 0"
H4_SHRUNK = "H 0 0 0; H 0 0 0.9; H 0.9 0 0.9; H 0.9 0 0"


def _codes(program) -> list[str]:
    return [d.code for d in check(program, molecular_registry(), task=freeze_task(program))]


def _pipeline(builder, geometry_for_samples: str, geometry_for_energy: str):
    model = builder.electronic_model(basis="sto-3g")
    active = {"n_elec": 4, "n_orb": 4}
    sector = builder.sector(n_alpha=2, n_beta=2, n_orb=4)
    h_samples = builder.hamiltonian(builder.molecule(geometry_for_samples), model, active=active)
    h_energy = (h_samples if geometry_for_energy == geometry_for_samples else
                builder.hamiltonian(builder.molecule(geometry_for_energy), model, active=active))
    prep = builder.node("prepare_state", hamiltonian=h_samples, sector=sector, ansatz="lucj",
                        init="ccsd")
    samples = builder.node("sample_configurations", prep=prep, shots=500, seed=3)
    samples.effects = "random"
    energy = builder.ground_energy(h_energy, sector, samples=samples, n_batches=2)
    builder.refine(energy, "sqd")
    builder.bind(energy, "sqd_sector")
    return prep, samples, energy


def test_sqd_consumes_samples_through_a_typed_port() -> None:
    with build.program("sqd_in_graph") as b:
        prep, samples, energy = _pipeline(b, H4, H4)
    program = b.to_ir()
    assert _codes(program) == []
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    assert results[energy.id].relation == "variational_upper_bound"
    # Every executed node has a measured cost record; shots are charged once, to the
    # sampling node, and state preparation carries its own measured cost.
    costs = results["costs"]
    assert costs[samples.id].sim_shots == 500
    assert costs[energy.id].sim_shots in (None, 0)
    assert costs[prep.id].wall_s is not None and costs[prep.id].wall_s > 0


def test_samples_from_another_hamiltonian_are_rejected() -> None:
    with build.program("foreign_samples") as b:
        _pipeline(b, H4, H4_SHRUNK)
    assert "CF-E010" in _codes(b.to_ir())


def test_domain_objects_cannot_hide_in_params() -> None:
    raw = Samples(counts={"51": 10}, shots=10, encoding_id="x", basis="computational",
                  source="test", seed=0)
    with build.program("raw_samples") as b:
        ham = b.hamiltonian(b.molecule(H4), b.electronic_model(basis="sto-3g"),
                            active={"n_elec": 4, "n_orb": 4})
        energy = b.ground_energy(ham, b.sector(n_alpha=2, n_beta=2, n_orb=4), samples=raw)
        b.refine(energy, "sqd")
        b.bind(energy, "sqd_sector")
    assert {"CF-E002", "CF-E080"} & set(_codes(b.to_ir()))


def test_sqd_never_builds_the_full_sector() -> None:
    # N2 6-31G CAS(8,10): 44,100 determinants. A dense full-sector build needs ~15 GB.
    store = ArtifactStore()
    data = build_molecule("N 0 0 0; N 0 0 1.10", ElectronicModel(basis="6-31g"), store,
                          active=ActiveSpaceSpec(n_elec=8, n_orb=10))
    n_orb, half = 10, 4
    hf = sum(1 << i for i in range(half)) | sum(1 << (n_orb + i) for i in range(half))
    counts = {str(hf): 200}
    for occupied in range(half):
        for virtual in range(half, n_orb):
            counts[str(hf ^ (1 << occupied) ^ (1 << virtual))] = 5
            counts[str(hf ^ (1 << (n_orb + occupied)) ^ (1 << (n_orb + virtual)))] = 5
    samples = Samples(counts=counts, shots=sum(counts.values()), encoding_id="synthetic",
                      basis="computational", source="test", seed=0)
    from chemflow.impls.sqd import sqd_sector

    def timeout(*_):
        raise TimeoutError("SQD exceeded 60 s on a 44,100-determinant sector")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        energy, _ = sqd_sector(data, data.sector, store, samples, n_batches=2, max_iter=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    exact = fci.direct_spin1.kernel(store.get(data.hamiltonian.h1_ref),
                                    store.get(data.hamiltonian.h2_ref), n_orb,
                                    (half, half))[0] + data.hamiltonian.ledger.total().value
    assert energy >= exact - 1e-9


def test_lucj_vqe_supports_multistart() -> None:
    store = ArtifactStore()
    data = build_molecule(H4, ElectronicModel(basis="sto-3g"), store,
                          active=ActiveSpaceSpec(n_elec=4, n_orb=4))
    engine = SectorEngine(data.hamiltonian, data.sector, store)
    single = vqe(engine, data.sector, ansatz="lucj", maxiter=300, seed=0)
    multi = vqe(engine, data.sector, ansatz="lucj", maxiter=300, seed=0, n_starts=4)
    assert multi.energy <= single.energy + 1e-9
    assert len(multi.start_energies) == 4
    assert multi.energy == pytest.approx(min(multi.start_energies), abs=1e-9)


def test_cards_do_not_cite_mutable_review_documents() -> None:
    for card in load_cards():
        for evidence in card.evidence:
            path = Path(evidence.path)
            assert path.name != "REVIEW.md" and "reviews" not in path.parts, card.id
