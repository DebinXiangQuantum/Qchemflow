"""Reviewer regression tests, M2b round 3 (written by Claude; implementers must not edit).

Calibration (reviewer, N2 6-31G CAS(8,10), 44,100 determinants): configuration-based spin
completion of the product of the 61 most important FCI alpha strings grows 3,721 -> 10,193
determinants (x2.7, 23% of the sector); product-string closure grows it to the full sector.
"""
from chemflow import build
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor, freeze_task

N2 = "N 0 0 0; N 0 0 1.10"
SECTOR_DIMENSION = 210 * 210


def test_spin_completion_is_not_full_ci_in_disguise() -> None:
    with build.program("n2_config_complete") as b:
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
    program = b.to_ir()
    results = Executor(molecular_registry()).run(program, task=freeze_task(program))
    for node in (sqd, sci):
        diagnostics = results[node.id].diagnostics
        before = diagnostics["dimension_before_completion"]
        after = diagnostics["dimension_after_completion"]
        assert after <= 4 * before, (node.id, before, after)
        assert after <= 0.35 * SECTOR_DIMENSION, (node.id, after)
        assert abs(diagnostics["sector_fraction"] - after / SECTOR_DIMENSION) < 1e-12
        assert abs(diagnostics["s2"]) < 1e-8
        assert results[node.id].relation == "variational_upper_bound"
