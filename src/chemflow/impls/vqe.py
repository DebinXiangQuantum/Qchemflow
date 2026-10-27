from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from chemflow.impls.ansatz import uccsd_pool, uccsd_state
from chemflow.impls.fci import spin_square
from chemflow.impls.jw import pauli_terms
from chemflow.impls.lucj import LucjParams, lucj_state
from chemflow.impls.sector import SectorEngine
from chemflow.impls.statevector import pauli_shot_estimator
from chemflow.types import QubitOperator, Sector, StatePrep


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    s2: float
    n_iter: int
    n_energy_evals: int
    converged: bool
    state: np.ndarray
    stat_err: float | None = None
    sim_shots: int = 0
    start_energies: list[float] | None = None


def vqe(engine: SectorEngine, sector: Sector, ansatz: str = "uccsd",
        optimizer: str = "L-BFGS-B", maxiter: int = 200,
        spin_adapted: bool = False, layers: int = 1,
        initial: np.ndarray | None = None, measurement: str = "exact",
        shots: int = 1024, seed: int = 0, hamiltonian=None, store=None,
        n_starts: int = 1, init: str = "random",
        **kwargs) -> VQEResult:
    if n_starts < 1 or init not in ("random", "zero", "ccsd"):
        raise ValueError("Invalid number of starts or initialization")
    if n_starts > 1:
        size = (len(uccsd_pool(sector, spin_adapted)[1]) if ansatz == "uccsd" else
                LucjParams.zeros(sector.n_orb).size) * layers
        initializers = [initial]
        seeded_vector = None
        if ansatz == "lucj" and hamiltonian is not None and store is not None:
            from chemflow.impls.lucj import ccsd_seed_from_integrals

            seeded = ccsd_seed_from_integrals(hamiltonian, sector, store)
            if seeded is not None:
                seeded_vector = np.tile(seeded.to_vector(), layers)
        if initial is None and init == "ccsd":
            initializers.append(np.zeros(size))
        elif initial is None and init == "zero":
            if seeded_vector is not None:
                initializers.append(seeded_vector)
        else:
            if seeded_vector is not None:
                initializers.append(seeded_vector)
            initializers.append(np.zeros(size))
        rng = np.random.default_rng(seed + 1)
        while len(initializers) < n_starts:
            initializers.append(rng.normal(0, 0.12, size))
        starts = [vqe(engine, sector, ansatz=ansatz, optimizer=optimizer, maxiter=maxiter,
                      spin_adapted=spin_adapted, layers=layers, initial=values,
                      measurement=measurement, shots=shots, seed=seed + index,
                      hamiltonian=hamiltonian, store=store, init=init)
                  for index, values in enumerate(initializers[:n_starts])]
        best = min(starts, key=lambda result: result.energy)
        best.start_energies = [result.energy for result in starts]
        return best
    if ansatz not in ("uccsd", "lucj") or measurement not in ("exact", "shots"):
        raise ValueError("Unsupported ansatz or measurement")
    if measurement == "shots" and (sector.n_orb > 7 or shots < 2 or
                                   hamiltonian is None or store is None):
        raise ValueError("Shot VQE requires Hamiltonian, store, <=14 qubits and shots >= 2")
    _, generators = uccsd_pool(sector, spin_adapted) if ansatz == "uccsd" else ([], [])
    count = (len(generators) if ansatz == "uccsd" else LucjParams.zeros(sector.n_orb).size) * layers
    parameters = (np.zeros(count) if ansatz == "uccsd" or init == "zero" else
                  np.random.default_rng(seed).normal(0, 0.12, count)) if initial is None else np.asarray(
                      initial, dtype=float)
    if init == "ccsd" and ansatz == "lucj" and initial is None:
        from chemflow.impls.lucj import ccsd_seed_from_integrals

        seeded = (ccsd_seed_from_integrals(hamiltonian, sector, store)
                  if hamiltonian is not None and store is not None else None)
        if seeded is not None:
            parameters = np.tile(seeded.to_vector(), layers)
    evaluations = 0
    total_shots = 0
    operator = (QubitOperator(terms=pauli_terms(hamiltonian, store), encoding_id="sector_vqe",
                             matrix_ref="") if measurement == "shots" else None)

    def state_at(values):
        if ansatz == "uccsd":
            return uccsd_state(sector, values, generators * layers)
        size = LucjParams.zeros(sector.n_orb).size
        return lucj_state(sector, [LucjParams.from_vector(values[index:index + size], sector.n_orb)
                                   for index in range(0, count, size)])

    def objective(values):
        nonlocal evaluations, total_shots
        evaluations += 1
        state = state_at(values)
        if operator is None:
            return engine.energy(state)
        full = np.zeros(1 << (2 * sector.n_orb), dtype=complex)
        full[engine.states] = state
        prep = StatePrep(state_ref=store.put(full), encoding_id="sector_vqe",
                         hamiltonian_id=hamiltonian.id)
        samples, energy, _ = pauli_shot_estimator(operator, prep, store, shots,
                                                  seed + evaluations)
        total_shots += samples.shots
        return energy

    method = "COBYLA" if measurement == "shots" else optimizer
    result = minimize(objective, parameters, method=method,
                      options={"maxiter": maxiter, **({"ftol": 1e-12} if method == "L-BFGS-B"
                                                  else {})})
    state = state_at(result.x)
    if operator is None:
        energy, stat_err = engine.energy(state), None
    else:
        full = np.zeros(1 << (2 * sector.n_orb), dtype=complex)
        full[engine.states] = state
        prep = StatePrep(state_ref=store.put(full), encoding_id="sector_vqe",
                         hamiltonian_id=hamiltonian.id)
        samples, energy, stat_err = pauli_shot_estimator(operator, prep, store, shots,
                                                          seed + evaluations + 1)
        total_shots += samples.shots
    return VQEResult(energy, result.x, spin_square(state, sector),
                     result.get("nit", result.nfev), evaluations, result.success, state,
                     stat_err, total_shots, [energy])
