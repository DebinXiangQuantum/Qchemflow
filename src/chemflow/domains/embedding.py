"""Wavefunction embedding with an occupied-index fragment energy partition."""

from dataclasses import replace

import numpy as np

from chemflow.diagnostics import AdapterError
from chemflow.ids import content_hash
from chemflow.runtime.result import ExecutionResult
from chemflow.types import ConstantLedger, Hamiltonian, Samples, Sector


def _span(matrix, threshold):
    if matrix.shape[1] == 0:
        return matrix
    vectors, values, _ = np.linalg.svd(matrix, full_matrices=False)
    return vectors[:, values > threshold]


def _bno_expand(basis, density, eta):
    complement = np.eye(len(density)) - basis @ basis.T
    values, vectors = np.linalg.eigh(complement @ density @ complement)
    return _span(np.column_stack((basis, vectors[:, values > eta])), 1e-9)


def _cluster_reference(h1, eri, occupied):
    from pyscf import gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = 2 * occupied
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(len(h1))
    mf._eri = eri
    mf.mo_coeff = np.eye(len(h1))
    mf.mo_occ = np.array([2.] * occupied + [0.] * (len(h1) - occupied))
    mf.mo_energy = np.diag(mf.get_fock(dm=mf.make_rdm1()))
    mf.e_tot = mf.energy_tot(mf.make_rdm1())
    mf.converged = True
    return mf


def _amplitudes(h1, eri, occupied, solver_name, data, store, shots, seed):
    from pyscf import cc, ci, fci

    norb = len(h1)
    mf = _cluster_reference(h1, eri, occupied)
    if solver_name == "ccsd":
        solver = cc.CCSD(mf)
        solver.conv_tol = 1e-11
        solver.kernel()
        if not solver.converged:
            raise ValueError("Embedded CCSD did not converge")
        return solver.t1, solver.t2 + np.einsum("ia,jb->ijab", solver.t1, solver.t1)
    if solver_name == "fci":
        solver = fci.direct_spin1.FCI()
        solver.conv_tol = 1e-12
        _, vector = solver.kernel(h1, eri, norb, (occupied, occupied))
    else:
        from chemflow.impls.sqd import sqd_sector

        rng = np.random.default_rng(seed)
        strings = fci.cistring.make_strings(range(norb), occupied)
        counts = {}
        for _ in range(shots):
            mask = int(rng.choice(strings)) | (int(rng.choice(strings)) << norb)
            counts[mask] = counts.get(mask, 0) + 1
        sector = Sector(n_alpha=occupied, n_beta=occupied, n_orb=norb)
        ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), n_orb=norb,
                          space=content_hash((h1.tolist(), occupied)), ledger=ConstantLedger())
        cluster = replace(data, hamiltonian=ham, sector=sector, mf=mf, mol=mf.mol)
        samples = Samples(counts={str(mask): count for mask, count in counts.items()},
                          shots=shots, basis="computational", seed=seed,
                          encoding_id="sector_state", source="uniform_classical_sampler")
        _, _, vector = sqd_sector(cluster, sector, store, samples, seed=seed,
                                  n_batches=2, max_iter=2, return_wavefunction=True)
    cisd = ci.cisd.from_fcivec(vector, norb, (occupied, occupied))
    reference, singles, doubles = ci.cisd.cisdvec_to_amplitudes(cisd, norb, occupied)
    if abs(reference) < 1e-9:
        raise ValueError("Fragment wavefunction has vanishing HF overlap")
    return singles / reference, doubles / reference


def ewf_embedding(store, hamiltonian, sector, fragments="atoms", bath="dmet", bath_tol=1e-8,
                  bno_eta=1e-6, solver="fci", shots=4096, seed=7, s=None, total_spin=None):
    from pyscf import ao2mo, lo, mp

    data = hamiltonian
    requested_spin = s if s is not None else total_spin
    if requested_spin is None:
        requested_spin = sector.get("s", sector.get("total_spin"))
    if (fragments != "atoms" or bath not in ("full", "dmet", "bno") or
            solver not in ("fci", "ccsd", "sqd") or bath_tol < 0 or bno_eta < 0 or shots < 1):
        raise ValueError("Invalid EWF fragments, bath, solver, or thresholds")
    if (hasattr(data, "lattice_id") or data.space.frozen_orbitals or
            data.space.n_orb != data.mol.nao_nr() or
            sector["n_alpha"] != sector["n_beta"] or
            (sector["n_alpha"], sector["n_beta"], sector["n_orb"]) !=
            (data.sector.n_alpha, data.sector.n_beta, data.sector.n_orb) or
            requested_spin not in (None, 0)):
        raise AdapterError("CF-E020", None, "EWF requires a full-space closed-shell molecule",
                           "Use RHF, singlet, and the full orbital space")
    mf, mol = data.mf, data.mol
    occupied = data.sector.n_alpha
    orbitals = np.asarray(mf.mo_coeff)
    overlap = mf.get_ovlp()
    iaos = lo.vec_lowdin(lo.iao.iao(mol, orbitals[:, :occupied]), overlap)
    reference_mol = lo.iao.reference_mol(mol)
    ownership = reference_mol.aoslice_by_atom()[:, 2:]
    iao_mo = orbitals.T @ overlap @ iaos
    projectors = []
    for start, stop in ownership:
        fragment_occ = iao_mo[:occupied, start:stop]
        projectors.append(fragment_occ @ fragment_occ.T)
    if not np.allclose(sum(projectors), np.eye(occupied), atol=1e-8):
        raise ValueError("IAO occupied projectors do not resolve the occupied identity")
    density = None
    if bath == "bno":
        perturbation = mp.MP2(mf).run()
        density = perturbation.make_rdm1()
    full_h1 = mf.get_hcore()
    full_fock = mf.get_fock()
    correlation = 0.
    measured_shots = 0
    sizes, contributions = [], []
    for (start, stop), projector in zip(ownership, projectors):
        fragment = iao_mo[:, start:stop]
        occ_basis = _span(fragment[:occupied], bath_tol)
        vir_basis = _span(fragment[occupied:], bath_tol)
        if bath == "full":
            occ_basis, vir_basis = np.eye(occupied), np.eye(len(orbitals) - occupied)
        elif bath == "bno":
            occ_basis = _bno_expand(occ_basis, 2 * np.eye(occupied) -
                                    density[:occupied, :occupied], bno_eta)
            vir_basis = _bno_expand(vir_basis, density[occupied:, occupied:], bno_eta)
        cluster_occ, cluster_vir = occ_basis.shape[1], vir_basis.shape[1]
        sizes.append({"n_occ": cluster_occ, "n_vir": cluster_vir,
                      "n_orb": cluster_occ + cluster_vir})
        if not cluster_occ or not cluster_vir:
            contributions.append(0.)
            continue
        for basis, block in ((occ_basis, orbitals[:, :occupied]),
                             (vir_basis, orbitals[:, occupied:])):
            _, rotation = np.linalg.eigh(basis.T @ block.T @ full_fock @ block @ basis)
            basis[:] = basis @ rotation
        coefficients = np.column_stack((orbitals[:, :occupied] @ occ_basis,
                                         orbitals[:, occupied:] @ vir_basis))
        core_density = 2 * orbitals[:, :occupied] @ (
            np.eye(occupied) - occ_basis @ occ_basis.T) @ orbitals[:, :occupied].T
        potential = mf.get_veff(mol, core_density)
        h1 = coefficients.T @ (full_h1 + potential) @ coefficients
        eri = ao2mo.restore(1, ao2mo.kernel(mol, coefficients), len(h1))
        singles, doubles = _amplitudes(h1, eri, cluster_occ, solver, data, store, shots, seed)
        if solver == "sqd":
            measured_shots += shots
        projected = occ_basis.T @ projector @ occ_basis
        doubles = np.einsum("ik,kjab->ijab", projected, doubles)
        singles = projected @ singles
        integrals = eri[:cluster_occ, cluster_occ:, :cluster_occ, cluster_occ:]
        fock = coefficients.T @ full_fock @ coefficients
        increment = (2 * np.einsum("ia,ia", fock[:cluster_occ, cluster_occ:], singles) +
                     2 * np.einsum("iajb,ijab", integrals, doubles) -
                     np.einsum("ibja,ijab", integrals, doubles))
        correlation += float(increment)
        contributions.append(float(increment))
    diagnostics = {
        "n_fragments": len(projectors), "cluster_sizes": sizes, "bath": bath,
        "bath_tol": bath_tol, "bno_eta": bno_eta, "solver": solver,
        "fragment_correlation_energies": contributions,
        "partition": "one occupied index projected onto atom IAOs"}
    return ExecutionResult((float(mf.e_tot + correlation), diagnostics),
                           cost={"sim_shots": measured_shots})
