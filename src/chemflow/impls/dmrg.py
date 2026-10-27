"""block2 variational MPS energies for site-basis lattice integrals."""

from tempfile import TemporaryDirectory

import numpy as np


def dmrg_block2(store, hamiltonian, sector, s=None, total_spin=None,
                bond_dimension=64, sweeps=10, seed=11):
    from block2 import Random
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    from chemflow.domains.materials import LatticeData

    if not isinstance(hamiltonian, LatticeData):
        raise TypeError("dmrg_block2 requires a lattice Hamiltonian")
    if any(type(value) is not int or value < 1 for value in (bond_dimension, sweeps)):
        raise ValueError("bond_dimension and sweeps must be positive integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    target = s if s is not None else total_spin
    if target is None:
        target = sector.get("s", sector.get("total_spin"))
    magnetization = sector["n_alpha"] - sector["n_beta"]
    if target is not None and (2 * target != int(2 * target) or
            2 * target < abs(magnetization) or
            (int(2 * target) - magnetization) % 2):
        raise ValueError("Total spin is incompatible with the particle sector")
    symmetry = SymmetryTypes.SU2 if target is not None else SymmetryTypes.SZ
    with TemporaryDirectory(prefix="chemflow-dmrg-") as scratch:
        driver = DMRGDriver(scratch=scratch, symm_type=symmetry, n_threads=1,
                            stack_mem=256 * 1024 * 1024)
        try:
            Random.rand_seed(seed)
            driver.initialize_system(n_sites=sector["n_orb"],
                                     n_elec=sector["n_alpha"] + sector["n_beta"],
                                     spin=int(2 * target) if target is not None else magnetization)
            physical = hamiltonian.hamiltonian
            mpo = driver.get_qc_mpo(h1e=store.get(physical.h1_ref),
                                    g2e=store.get(physical.h2_ref),
                                    ecore=physical.ledger.total().value, iprint=0)
            ket = driver.get_random_mps(tag="GROUND", bond_dim=bond_dimension)
            driver.dmrg(mpo, ket, n_sweeps=sweeps, bond_dims=[bond_dimension],
                        noises=[0.0], thrds=[1e-12], tol=1e-12, iprint=0)
            norm = float(driver.expectation(ket, driver.get_identity_mpo(), ket))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid final MPS norm")
            energy = float(driver.expectation(ket, mpo, ket)) / norm
            _, discarded, _ = driver.get_dmrg_results()
            diagnostics = {"discarded_weight": float(discarded[-1]), "mps_norm": norm,
                           "evaluation": "exact", "engine": "block2",
                           "normalized_mps_expectation": True, "seed": seed,
                           "bond_dimension": bond_dimension, "sweeps": len(discarded)}
            if target is not None:
                diagnostics["s2"] = target * (target + 1)
            return energy, diagnostics
        finally:
            driver.finalize()
