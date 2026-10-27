import numpy as np
from scipy import sparse

from chemflow.diagnostics import AdapterError
from chemflow.impls.fci import matrix
from chemflow.impls.jw import pauli_terms
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian, QubitEncoding, QubitOperator, Samples, StatePrep


def encode(hamiltonian: Hamiltonian, store: ArtifactStore) -> tuple[QubitEncoding, QubitOperator]:
    encoding = QubitEncoding(n_qubits=2 * hamiltonian.n_orb, hamiltonian_id=hamiltonian.id)
    if encoding.n_qubits > 14:
        raise AdapterError("CF-E080", None, "Dense encoding exceeds 14 qubits",
                           "Use a smaller active space or sector method")
    states = range(1 << encoding.n_qubits)
    reference = store.put(matrix(hamiltonian, store, states).toarray())
    return encoding, QubitOperator(terms=pauli_terms(hamiltonian, store),
                                   encoding_id=encoding.id, matrix_ref=reference)


def prepare_hf(hamiltonian: Hamiltonian, encoding: QubitEncoding, store: ArtifactStore,
               n_alpha: int, n_beta: int) -> StatePrep:
    if encoding.n_qubits > 14:
        raise AdapterError("CF-E080", None, "Statevector exceeds 14 qubits",
                           "Use a smaller active space")
    bits = sum(1 << index for index in range(n_alpha))
    bits |= sum(1 << (hamiltonian.n_orb + index) for index in range(n_beta))
    state = np.zeros(1 << encoding.n_qubits)
    state[bits] = 1
    return StatePrep(state_ref=store.put(state), encoding_id=encoding.id,
                     hamiltonian_id=hamiltonian.id, recipe="hartree_fock")


def expectation(operator: QubitOperator, prep: StatePrep, store: ArtifactStore) -> float:
    if operator.encoding_id != prep.encoding_id:
        raise AdapterError("CF-E090", None, "Encoding mismatch", "Re-encode the state")
    state = store.get(prep.state_ref)
    return float(np.real(np.vdot(state, store.get(operator.matrix_ref) @ state)))


def sample_configurations(prep_or_state: StatePrep | np.ndarray, shots: int, seed: int,
                          readout_flip_p: float = 0.0, *,
                          store: ArtifactStore | None = None,
                          sector=None) -> Samples:
    if shots < 1 or not 0 <= readout_flip_p <= 1:
        raise ValueError("Positive shots and readout_flip_p in [0, 1] required")
    sector_indices = isinstance(prep_or_state, StatePrep) and prep_or_state.sector is not None
    if isinstance(prep_or_state, StatePrep):
        if store is None:
            raise ValueError("StatePrep requires an artifact store")
        state = np.asarray(store.get(prep_or_state.state_ref))
        encoding_id = prep_or_state.encoding_id
        sector = prep_or_state.sector
    else:
        state = np.asarray(prep_or_state)
        encoding_id = "sector_state"
        if sector is not None:
            from chemflow.impls.fci import sector_states

            states = sector_states(sector)
            if len(state) != len(states):
                raise ValueError("Sector state length mismatch")
            embedded = np.zeros(1 << (2 * sector.n_orb), dtype=state.dtype)
            embedded[states] = state
            state = embedded
    if state.ndim != 1 or not len(state) or (sector is None and len(state) & (len(state) - 1)):
        raise ValueError("State must be a power-of-two statevector")
    probabilities = np.abs(state) ** 2
    if not np.isclose(probabilities.sum(), 1):
        raise ValueError("State must be normalized")
    rng = np.random.default_rng(seed)
    drawn = rng.choice(len(state), size=shots, p=probabilities).astype(np.int64)
    if sector_indices:
        from chemflow.impls.fci import sector_states

        drawn = np.asarray(sector_states(sector))[drawn]
    if readout_flip_p:
        width = 2 * sector.n_orb if sector_indices else len(state).bit_length() - 1
        for bit in range(width):
            drawn ^= (rng.random(shots) < readout_flip_p).astype(np.int64) << bit
    keys, counts = np.unique(drawn, return_counts=True)
    return Samples(counts={str(key): int(count) for key, count in zip(keys, counts)},
                   shots=shots, encoding_id=encoding_id, basis="computational",
                   source="quantum_simulator", seed=seed)


def ideal_energy_projective_sampler(operator: QubitOperator, prep: StatePrep,
                                    store: ArtifactStore, shots: int,
                                    seed: int) -> tuple[Samples, float, float]:
    """Model ideal projective measurement in the Hamiltonian eigenbasis."""
    if shots <= 1:
        raise AdapterError("CF-E020", None, "At least two shots required", "Increase shots")
    if operator.encoding_id != prep.encoding_id:
        raise AdapterError("CF-E090", None, "Encoding mismatch", "Re-encode the state")
    eigenvalues, eigenvectors = np.linalg.eigh(store.get(operator.matrix_ref))
    probabilities = np.abs(eigenvectors.conj().T @ store.get(prep.state_ref)) ** 2
    draws = np.random.default_rng(seed).choice(len(eigenvalues), size=shots, p=probabilities)
    counts = np.bincount(draws, minlength=len(eigenvalues))
    samples = Samples(counts={str(index): int(count) for index, count in enumerate(counts)
                              if count}, shots=shots, encoding_id=operator.encoding_id,
                      basis="energy_eigenbasis", source="quantum_simulator", seed=seed)
    values = eigenvalues[draws]
    return samples, float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(shots))


def pauli_measurement_groups(operator: QubitOperator) -> int:
    return sum(set(word) != {"I"} for word in operator.terms)


def pauli_shot_estimator(operator: QubitOperator, prep: StatePrep, store: ArtifactStore,
                         shots: int, seed: int) -> tuple[Samples, float, float]:
    if shots < 2:
        raise AdapterError("CF-E020", None, "At least two shots per Pauli term required",
                           "Increase shots")
    if operator.encoding_id != prep.encoding_id:
        raise AdapterError("CF-E090", None, "Encoding mismatch", "Re-encode the state")
    state = store.get(prep.state_ref)
    rng = np.random.default_rng(seed)
    estimate = 0.0
    variance = 0.0
    counts: dict[str, int] = {}
    paulis = {"I": sparse.eye(2, format="csr"),
              "X": sparse.csr_matrix([[0, 1], [1, 0]]),
              "Y": sparse.csr_matrix([[0, -1j], [1j, 0]]),
              "Z": sparse.diags([1, -1], format="csr", dtype=complex)}
    for word, coefficient in operator.terms.items():
        if set(word) == {"I"}:
            estimate += coefficient
            continue
        term = sparse.csr_matrix([[1]])
        for letter in reversed(word):
            term = sparse.kron(term, paulis[letter], format="csr")
        expected = float(np.real(np.vdot(state, term @ state)))
        positives = rng.binomial(shots, np.clip((1 + expected) / 2, 0, 1))
        counts[word] = int(positives)
        estimate += coefficient * (2 * positives / shots - 1)
        variance += coefficient ** 2 * max(0.0, 1 - expected ** 2) / shots
    samples = Samples(counts=counts, shots=shots * pauli_measurement_groups(operator),
                      encoding_id=operator.encoding_id, basis="pauli_terms",
                      source="quantum_simulator", seed=seed)
    return samples, estimate, float(np.sqrt(variance))


sample = pauli_shot_estimator
