from math import comb

import numpy as np
from pyscf import fci

from chemflow.impls.spin_complete import (
    complete_spin_configurations,
    complete_spin_strings,
    diagonalize_configurations,
)


def state_artifact(vector, sector, store):
    from chemflow.impls.fci import sector_states

    if hasattr(vector, "_strs"):
        vector = fci.selected_ci.to_fci(vector, sector.n_orb,
                                        (sector.n_alpha, sector.n_beta))
    half = (1 << sector.n_orb) - 1
    amplitudes = [vector[fci.cistring.str2addr(sector.n_orb, sector.n_alpha, state & half),
                         fci.cistring.str2addr(sector.n_orb, sector.n_beta,
                                              state >> sector.n_orb)]
                  for state in sector_states(sector)]
    return store.put(np.asarray(amplitudes))


def sci_pyscf(data, store, select_cutoff: float = 5e-4,
              total_spin: int | None = None, max_product_dimension: int = 50000,
              spin_completion: str = "configurations"):
    if spin_completion not in ("configurations", "product_closure"):
        raise ValueError("Unknown spin completion method")
    if select_cutoff < 0:
        raise ValueError("select_cutoff must be nonnegative")
    hamiltonian = data.hamiltonian
    sector = data.sector
    h1 = store.get(hamiltonian.h1_ref)
    eri = store.get(hamiltonian.h2_ref)
    solver = fci.selected_ci.SelectedCI()
    solver.select_cutoff = select_cutoff
    solver.ci_coeff_cutoff = min(select_cutoff, solver.ci_coeff_cutoff)
    electrons = (sector.n_alpha, sector.n_beta)

    def export_basis(determinants):
        return store.put(np.asarray(sorted((int(alpha), int(beta))
                                           for alpha, beta in determinants), dtype=np.int64))

    if min(electrons) == 0:
        from chemflow.impls.pyscf_impls import fci_pyscf

        energy, spin_squared, roots, state = fci_pyscf(
            data, store, total_spin, return_roots=True, return_wavefunction=True)
        dimension = comb(sector.n_orb, max(electrons))
        return energy, {"s2": spin_squared, "determinant_count": dimension,
                        "state_ref": state_artifact(state, sector, store),
                        "determinants_ref": export_basis(
                            (alpha, beta) for alpha in fci.cistring.make_strings(
                                range(sector.n_orb), sector.n_alpha)
                            for beta in fci.cistring.make_strings(range(sector.n_orb), sector.n_beta)),
                        "select_cutoff": select_cutoff, "sector_fraction": 1.,
                        "dimension_before_completion": dimension,
                        "dimension_after_completion": dimension,
                        "spin_completion": spin_completion, "spin_completion_reason": None,
                        "n_roots_examined": roots,
                        "solver_path": "exact polarized sector (SelectedCI zero-spin limit)"}
    if select_cutoff <= 1e-8:
        solver.nroots = min(4, len(fci.cistring.make_strings(range(sector.n_orb),
                                                              sector.n_alpha)) *
                            len(fci.cistring.make_strings(range(sector.n_orb),
                                                           sector.n_beta)))
        strings = tuple(np.asarray(fci.cistring.make_strings(range(sector.n_orb), count))
                        for count in electrons)
        energy, vector = solver.kernel_fixed_space(h1, eri, sector.n_orb,
                                                    electrons, strings)
        if isinstance(vector, list):
            energy, vector = energy[0], vector[0]
    else:
        energy, vector = solver.kernel(h1, eri, sector.n_orb, electrons)
    before = len(vector._strs[0]) * len(vector._strs[1])
    reason = None
    if total_spin is not None and spin_completion == "configurations":
        determinants = complete_spin_configurations(
            {(int(alpha), int(beta)) for alpha in vector._strs[0]
             for beta in vector._strs[1]}, sector.n_alpha)
        energy, _, s2, roots, state = diagonalize_configurations(
            h1, eri, sector.n_orb, electrons, determinants, total_spin, return_roots=True,
            return_wavefunction=True)
        count = len(determinants)
        return float(energy) + hamiltonian.ledger.total().value, {
            "state_ref": state_artifact(state, sector, store),
            "determinants_ref": export_basis(determinants),
            "s2": s2, "determinant_count": count, "select_cutoff": select_cutoff,
            "dimension_before_completion": before, "dimension_after_completion": count,
            "sector_fraction": count / (comb(sector.n_orb, sector.n_alpha) *
                                        comb(sector.n_orb, sector.n_beta)),
            "spin_completion": spin_completion, "spin_completion_reason": None,
            "n_roots_examined": roots}
    if total_spin is not None and spin_completion == "product_closure" and (
            sector.n_alpha == sector.n_beta):
        selected = {int(string) for axis in vector._strs for string in axis}
        completed, reason = complete_spin_strings(selected, sector.n_orb,
                                                  max_product_dimension)
        strings = np.asarray(sorted(completed), dtype=np.int64)
        determinants = {(int(alpha), int(beta)) for alpha in strings for beta in strings}
        energy, _, s2, roots, state = diagonalize_configurations(
            h1, eri, sector.n_orb, electrons, determinants, total_spin, return_roots=True,
            return_wavefunction=True)
        count = len(determinants)
        return float(energy) + hamiltonian.ledger.total().value, {
            "state_ref": state_artifact(state, sector, store),
            "determinants_ref": export_basis(determinants),
            "s2": s2, "determinant_count": count, "select_cutoff": select_cutoff,
            "dimension_before_completion": before, "dimension_after_completion": count,
            "sector_fraction": count / (comb(sector.n_orb, sector.n_alpha) *
                                        comb(sector.n_orb, sector.n_beta)),
            "spin_completion": spin_completion, "spin_completion_reason": reason,
            "n_roots_examined": roots}
    count = len(vector._strs[0]) * len(vector._strs[1])
    s2 = float(solver.spin_square(vector, sector.n_orb,
                                   (sector.n_alpha, sector.n_beta))[0])
    # Evaluate the physical Hamiltonian, not the spin-penalty objective.
    if total_spin is not None:
        energy = solver.energy(h1, eri, vector, sector.n_orb, electrons)
    return float(energy) + hamiltonian.ledger.total().value, {
        "state_ref": state_artifact(vector, sector, store),
        "determinants_ref": export_basis((alpha, beta) for alpha in vector._strs[0]
                                          for beta in vector._strs[1]),
        "s2": s2, "determinant_count": count, "select_cutoff": select_cutoff,
        "dimension_before_completion": before, "dimension_after_completion": count,
        "sector_fraction": count / (comb(sector.n_orb, sector.n_alpha) *
                                    comb(sector.n_orb, sector.n_beta)),
        "spin_completion": spin_completion if total_spin is not None else None,
        "spin_completion_reason": reason}
