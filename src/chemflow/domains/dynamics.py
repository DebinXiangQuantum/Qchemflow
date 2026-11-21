"""Real-axis retarded spectral functions for site-basis impurity models.

H = -U/2 (n_0α+n_0β) + U n_0α n_0β + Σ ε_b n_bσ
    + Σ t_b (a†_0σ a_bσ + a†_bσ a_0σ), with zero constant.
Lehmann evaluates A(ω)=-Im G(ω+iη)/π using squared fermionic ladder
matrix elements. Both particle and hole denominators have +iη. Costs are
O(grid_size * (D_particle + D_hole)) beyond projected diagonalizations.
The exact reference enumerates full sectors (exponential time/storage).

Relations are executor/checker rules, not binding assertions: a response is
equivalent only for complete ground/particle/hole resolutions with full
sector coverage; otherwise approximate. The finite-grid sum rule is an
unrenormalized trapezoidal integral, not an assertion of unit spectral weight.
The L1 gate integrates absolute density differences on an identical grid;
its threshold is dimensionless (the gate unit identifies the energy axis).
"""

import numpy as np

from chemflow.domains.algorithms import as_sector, full_subspace, rayleigh_ritz, transition_sector
from chemflow.domains.algorithms import ladder as determinant_ladder
from chemflow.domains.materials import LatticeData
from chemflow.ids import content_hash
from chemflow.types import ConstantLedger, Hamiltonian, OrbitalSpace, Sector, SpectralEstimate
from chemflow.units import Q


def impurity_arrays(U, eps_bath, t_bath, unit="hartree"):
    if (not isinstance(eps_bath, (list, tuple)) or
            not isinstance(t_bath, (list, tuple)) or len(eps_bath) != len(t_bath)):
        raise ValueError("Bath energies and hybridisations must have equal lengths")
    scale = Q(1, unit).to("hartree").value
    values = np.asarray([U, *eps_bath, *t_bath], dtype=float) * scale
    if not np.isfinite(values).all():
        raise ValueError("Impurity parameters must be finite")
    size = 1 + len(eps_bath)
    h1 = np.diag([-values[0] / 2, *values[1:size]])
    h1[0, 1:] = h1[1:, 0] = values[size:]
    eri = np.zeros((size,) * 4)
    eri[0, 0, 0, 0] = values[0]
    return h1, eri


def impurity_model(store, **params):
    h1, eri = impurity_arrays(**params)
    size = len(h1)
    sector = Sector(n_alpha=(size + 1) // 2, n_beta=size // 2, n_orb=size)
    space = OrbitalSpace(coefficients_ref=store.put(np.eye(size)), n_orb=size,
                         n_elec=size, ordering="sites")
    ham = Hamiltonian(h1_ref=store.put(h1), h2_ref=store.put(eri), n_orb=size,
                      space=content_hash(space), ledger=ConstantLedger())
    identity = content_hash(params)
    return LatticeData(mol=None, mf=None, hamiltonian=ham, space=space, sector=sector,
                       model_id=identity, geometry_id=identity, lattice_id=identity)


def grid(omega, eta):
    if (set(omega) != {"start", "stop", "n", "unit"} or type(omega["n"]) is not int or
            omega["n"] < 2 or not np.isfinite([omega["start"], omega["stop"], eta]).all() or
            omega["stop"] <= omega["start"] or eta <= 0):
        raise ValueError("Require a finite increasing omega grid and eta > 0")
    scale = Q(1, omega["unit"]).to("hartree").value
    return np.linspace(omega["start"], omega["stop"], omega["n"]), scale


def ladder(state, mode, create):
    """Fermionic transition amplitude; the destination flips the selected bit."""
    transition = determinant_ladder(state, mode, create)
    return transition[1] if transition is not None else None


def lehmann_from_spectra(store, ground, particle, hole, orbital, spin, omega, eta):
    frequencies, scale = grid(omega, eta)
    if type(orbital) is not int or not 0 <= orbital < ground.sector.n_orb:
        raise ValueError("Spectral orbital outside sector")
    if spin not in ("alpha", "beta"):
        raise ValueError("Spectral spin must be alpha or beta")
    ground_vector = store.get(ground.vectors_ref)[:, 0]
    green = np.zeros(len(frequencies), dtype=complex)
    coverage = {"ground": ground.coverage}
    for label, excited, operator, sign in (("particle", particle, "create", 1),
                                            ("hole", hole, "annihilate", -1)):
        if excited is None:
            coverage[label] = 1.0
            continue
        if excited.hamiltonian_id != ground.hamiltonian_id:
            raise ValueError("Lehmann spectra require the same Hamiltonian")
        mode = transition_sector(ground.sector, excited.sector, operator, orbital, spin)
        positions = {state: index for index, state in enumerate(excited.subspace.determinants)}
        image = np.zeros(len(positions), dtype=complex)
        for state, coefficient in zip(ground.subspace.determinants, ground_vector):
            applied = ladder(state, mode, operator == "create")
            destination = state ^ (1 << mode)
            if applied is not None and destination in positions:
                image[positions[destination]] += applied * coefficient
        vectors = store.get(excited.resolution_vectors_ref)
        weights = np.abs(vectors.conj().T @ image) ** 2
        energies = store.get(excited.resolution_energies_ref)
        poles = sign * (energies - ground.energies[0]) / scale
        green += (weights[None, :] / (frequencies[:, None] - poles[None, :] +
                                      1j * eta)).sum(axis=1)
        coverage[label] = excited.coverage
    spectral = -green.imag / np.pi
    return SpectralEstimate(omega=tuple(frequencies), spectral=tuple(spectral),
                            unit=omega["unit"], diagnostics={
                                "sum_rule": float(np.trapezoid(spectral, frequencies)),
                                "ground_energy": float(ground.energies[0]),
                                "coverage": coverage, "density_unit": f"1/{omega['unit']}",
                                "resolution": "complete_projected"})


def spectral_exact(store, hamiltonian, sector, orbital, spin, omega, eta):
    sector = as_sector(sector)
    ground = rayleigh_ritz(store, hamiltonian, full_subspace(store, sector))
    spectra = []
    for delta in (1, -1):
        counts = {key: getattr(sector, key) for key in ("n_alpha", "n_beta", "n_orb")}
        if spin not in ("alpha", "beta"):
            raise ValueError("Spectral spin must be alpha or beta")
        counts[f"n_{spin}"] += delta
        if not 0 <= counts[f"n_{spin}"] <= sector.n_orb:
            spectra.append(None)
        else:
            spectra.append(rayleigh_ritz(store, hamiltonian, full_subspace(store, counts)))
    result = lehmann_from_spectra(store, ground, *spectra, orbital, spin, omega, eta)
    witnesses = []
    for spectrum in (ground, *spectra):
        witnesses.append(None if spectrum is None else {
            **spectrum.model_dump(mode="json", exclude={"subspace"}),
            "sector": spectrum.sector.model_dump(mode="json"),
            "determinants_ref": store.put(np.asarray(spectrum.subspace.determinants, dtype=np.int64))})
    return result.model_copy(update={"diagnostics": {
        **result.diagnostics, "resolution_witnesses": witnesses}})
