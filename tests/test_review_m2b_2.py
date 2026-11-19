"""Reviewer regression tests, M2b round 2 (written by Claude; implementers must not edit)."""
from pyscf import gto, mcscf, scf

from chemflow import build
from chemflow.check import check
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task
from chemflow.units import Q

N2 = "N 0 0 0; N 0 0 1.10"


def _run(program):
    return Executor(molecular_registry()).run(program, task=freeze_task(program))


def _codes(program) -> list[str]:
    return [d.code for d in check(program, molecular_registry(), task=freeze_task(program))]


def _n2_singlet_reference() -> float:
    mf = scf.RHF(gto.M(atom=N2, basis="6-31g", verbose=0)).run()
    casci = mcscf.CASCI(mf, 10, 8)
    casci.fix_spin_(ss=0)
    return casci.kernel()[0]


def test_spin_targeted_sqd_and_sci_are_exact_singlets_and_upper_bounds() -> None:
    # The symmetric product of sampled strings is not S2-invariant (reviewer spec error in
    # M2b); spin-targeted subspace methods must spin-complete their determinant space.
    with build.program("n2_spin_complete") as b:
        ham = b.hamiltonian(b.molecule(N2), b.electronic_model(basis="6-31g"),
                            active={"n_elec": 8, "n_orb": 10})
        sector = b.sector(n_alpha=4, n_beta=4, n_orb=10, s=0)
        prep = b.node("prepare_state", hamiltonian=ham, sector=sector, ansatz="lucj",
                      init="ccsd")
        samples = b.node("sample_configurations", prep=prep, shots=256, seed=11)
        samples.effects = "random"
        sqd = b.ground_energy(ham, sector, samples=samples, n_batches=2, max_iter=2, s=0)
        b.refine(sqd, "sqd")
        b.bind(sqd, "sqd_sector")
        sci = b.ground_energy(ham, sector, select_cutoff=1e-2, s=0)
        b.refine(sci, "sci")
        b.bind(sci, "sci_pyscf")
    results = _run(b.to_ir())
    singlet = _n2_singlet_reference()
    for node in (sqd, sci):
        estimate = results[node.id]
        assert abs(estimate.diagnostics["s2"]) < 1e-8, node.id
        assert estimate.relation == "variational_upper_bound", node.id
        assert estimate.value.value >= singlet - 1e-9, node.id


def test_gate_reference_must_share_the_spin_target() -> None:
    def program(reference_spin: int | None):
        with build.program("spin_identity") as b:
            ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"),
                                b.electronic_model(basis="sto-3g"))
            sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
            extra = {} if reference_spin is None else {"s": reference_spin}
            reference = b.ground_energy(ham, sector, **extra)
            candidate = b.ground_energy(ham, sector, s=0)
            b.refine(candidate, "hf")
            b.bind(candidate, "hf")
            b.accept(candidate, abs_error_le=Q(1.6e-3, "hartree"), reference=reference.id)
        return b.to_ir()

    assert "CF-E011" in _codes(program(None))
    assert "CF-E011" not in _codes(program(0))


def test_sample_shots_are_counted_once_in_the_budget() -> None:
    with build.program("shots_once") as b:
        ham = b.hamiltonian(b.molecule("H 0 0 0; H 0 0 0.74"), b.electronic_model(basis="sto-3g"))
        sector = b.sector(n_alpha=1, n_beta=1, n_orb=2)
        encoding = b.node("encode", hamiltonian=ham)
        prep = b.node("prepare", hamiltonian=ham, encoding=encoding)
        samples = b.node("sample_configurations", prep=prep, shots=120, seed=9)
        samples.effects = "random"
        energy = b.ground_energy(ham, sector, samples=samples, n_batches=2)
        b.refine(energy, "sqd")
        b.bind(energy, "sqd_sector")
        b.budget(sim_shots=150)
    results = _run(b.to_ir())
    assert results["costs"][samples.id].sim_shots == 120
    assert results["costs"][energy.id].sim_shots in (None, 0)
    assert results[energy.id].cost.sim_shots in (None, 0)

