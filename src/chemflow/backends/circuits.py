"""Number-preserving hardware circuits and bit-order conversion."""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from scipy.linalg import expm

from chemflow.impls.lucj import LucjParams
from chemflow.types import CircuitSpec


def _rotation(circuit: QuantumCircuit, matrix: np.ndarray, n_orb: int) -> QuantumCircuit:
    reduction = matrix.copy()
    steps = []
    for column in range(n_orb - 1):
        for bottom in range(n_orb - 1, column, -1):
            top = bottom - 1
            first, second = reduction[top, column], reduction[bottom, column]
            length = np.hypot(first, second)
            if length < 1e-14:
                continue
            cosine, sine = first / length, second / length
            eliminate = np.array([[cosine, sine], [-sine, cosine]])
            reduction[[top, bottom]] = eliminate @ reduction[[top, bottom]]
            steps.append((top, bottom, cosine, sine))
    for top, bottom, cosine, sine in reversed(steps):
        gate = np.eye(4)
        gate[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
        for offset in (0, n_orb):
            circuit.append(UnitaryGate(gate), [offset + top, offset + bottom])
    return circuit


def slater_givens(occupied: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Adjacent-orbital Givens rotations (top, bottom, cos, sin) preparing a determinant.

    ``occupied`` is the n x m matrix of the m occupied orbitals (columns). Applied in the
    returned order to the reference determinant (orbitals 0..m-1 occupied), the rotations
    prepare the determinant of ``occupied`` up to a global phase, with m(n - m) of them
    instead of the n(n - 1)/2 a general orbital rotation needs.
    """
    V = np.asarray(occupied, dtype=float).T.copy()
    m, n = V.shape
    # Rotations among the occupied orbitals are free: give V a staircase shape.
    for column in range(n - 1, n - m, -1):
        for row in range(m - 1 - (n - 1 - column)):
            first, second = V[row, column], V[row + 1, column]
            length = np.hypot(first, second)
            if length < 1e-14:
                continue
            cosine, sine = second / length, first / length
            V[[row, row + 1]] = np.array([[cosine, -sine], [sine, cosine]]) @ V[[row, row + 1]]
    steps = []
    for row in range(m):
        for column in range(n - m + row, row, -1):
            first, second = V[row, column - 1], V[row, column]
            length = np.hypot(first, second)
            if length < 1e-14:
                continue
            cosine, sine = first / length, second / length
            V[:, [column - 1, column]] = V[:, [column - 1, column]] @ np.array(
                [[cosine, -sine], [sine, cosine]])
            steps.append((column - 1, column, cosine, sine))
    return list(reversed(steps))


def _slater_preparation(circuit: QuantumCircuit, matrix: np.ndarray, n_orb: int,
                        n_alpha: int, n_beta: int) -> None:
    """Append exp(-K) applied to the HF reference (X gates already in place), per spin."""
    for offset, count in ((0, n_alpha), (n_orb, n_beta)):
        if count in (0, n_orb):
            continue
        for top, bottom, cosine, sine in slater_givens(matrix[:, :count]):
            gate = np.eye(4)
            gate[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
            circuit.append(UnitaryGate(gate), [offset + top, offset + bottom])


def xxz_bond(circuit: QuantumCircuit, first: int, second: int, a: float, b: float,
             c: float) -> None:
    """exp(i(a XX + b YY + c ZZ)) on two qubits with three CNOTs (Vatan-Williams).

    The same gates for every (a, b, c), so an XX (free-fermion) circuit and an XXZ
    (interacting) one differ only in rotation angles.
    """
    circuit.rz(np.pi / 2, second)
    circuit.cx(second, first)
    circuit.rz(np.pi / 2 - 2 * c, first)
    circuit.ry(np.pi / 2 - 2 * a, second)
    circuit.cx(first, second)
    circuit.ry(2 * b - np.pi / 2, second)
    circuit.cx(second, first)
    circuit.rz(-np.pi / 2, first)


def build_circuit(spec: CircuitSpec, *, slater_prep: bool = True) -> QuantumCircuit:
    if spec.ansatz == "xxz_trotter":
        # layers = ((J, Delta, dt, steps),): Trotterized exp(-i t H) of the open XXZ chain
        # H = sum J (XX + YY) + Delta ZZ, brick-wall order, from the Neel state |1010...>.
        if spec.n_qubits is None or spec.n_qubits < 2 or len(spec.layers) != 1:
            raise ValueError("XXZ Trotter circuits need n_qubits >= 2 and one parameter set")
        coupling, anisotropy, step, steps = spec.layers[0]
        circuit = QuantumCircuit(spec.n_qubits)
        for qubit in range(0, spec.n_qubits, 2):
            circuit.x(qubit)
        for _ in range(int(steps)):
            for parity in (0, 1):
                for qubit in range(parity, spec.n_qubits - 1, 2):
                    xxz_bond(circuit, qubit, qubit + 1, -coupling * step, -coupling * step,
                             -anisotropy * step)
        return circuit
    if spec.ansatz == "custom":
        if spec.n_qubits is None or spec.n_qubits < 1 or len(spec.layers) != 1:
            raise ValueError("Custom circuit requires qubit count and one parameter vector")
        circuit = QuantumCircuit(spec.n_qubits, spec.n_clbits + spec.n_qubits
                                 if spec.n_clbits else 0)

        def emit(gates):
            for gate in gates:
                if gate["op"] == "ry":
                    circuit.ry(float(spec.layers[0][gate["parameter"]]), gate["qubit"])
                elif gate["op"] == "cz":
                    circuit.cz(gate["control"], gate["target"])
                elif gate["op"] == "x":
                    circuit.x(gate["qubit"])
                elif gate["op"] == "measure":
                    circuit.measure(gate["qubit"], gate["bit"])
                elif gate["op"] == "if":
                    with circuit.if_test((circuit.clbits[gate["bit"]], 1)) as otherwise:
                        emit(gate["then"])
                    if gate["else"]:
                        with otherwise():
                            emit(gate["else"])
                else:
                    raise ValueError(f"Unsupported custom gate: {gate['op']}")

        emit(spec.gates)
        if spec.measurement_basis is not None:
            from chemflow.quantum.pauli import rotation_circuit

            circuit.compose(rotation_circuit(spec.measurement_basis), inplace=True)
        return circuit
    n_orb = spec.n_orb
    circuit = QuantumCircuit(2 * n_orb)
    for offset, count in ((0, spec.n_alpha), (n_orb, spec.n_beta)):
        for orbital in range(count):
            circuit.x(offset + orbital)
    if spec.ansatz == "uccsd":
        from chemflow.impls.jw import ladder, product

        occupied = [*range(spec.n_alpha), *range(n_orb, n_orb + spec.n_beta)]
        virtual = [*range(spec.n_alpha, n_orb), *range(n_orb + spec.n_beta, 2*n_orb)]
        excitations = [(holes, particles) for degree in (1, 2)
                      for holes in combinations(occupied, degree)
                      for particles in combinations(virtual, degree)
                      if sorted(index // n_orb for index in holes) ==
                      sorted(index // n_orb for index in particles)]
        for vector in spec.layers:
            for angle, (holes, particles) in zip(vector, excitations, strict=True):
                expanded = {"I" * (2*n_orb): 1}
                for orbital, create in [*((index, True) for index in particles),
                                        *((index, False) for index in reversed(holes))]:
                    expanded = product(expanded, ladder(orbital, 2*n_orb, create))
                for word, coefficient in expanded.items():
                    if abs(coefficient.imag) < 1e-14:
                        continue
                    from chemflow.quantum.pauli import rotation_circuit

                    rotation = rotation_circuit(word)
                    support = [index for index, letter in enumerate(word) if letter != "I"]
                    circuit.compose(rotation, inplace=True)
                    for control in support[:-1]:
                        circuit.cx(control, support[-1])
                    circuit.rz(-4 * float(angle) * coefficient.imag, support[-1])
                    for control in reversed(support[:-1]):
                        circuit.cx(control, support[-1])
                    circuit.compose(rotation.inverse(), inplace=True)
    if not spec.final_rotation and (spec.ansatz != "lucj" or len(spec.layers) != 1):
        raise ValueError("Omitting the final rotation needs a one-layer LUCJ circuit")
    for index, vector in enumerate(spec.layers if spec.ansatz == "lucj" else ()):
        layer = LucjParams.from_vector(np.asarray(vector), n_orb)
        if len(vector) != layer.size:
            raise ValueError("Invalid LUCJ parameter vector")
        generator = np.zeros((n_orb, n_orb))
        for angle, (left, right) in zip(layer.kparams, combinations(range(n_orb), 2)):
            generator[left, right] = angle
            generator[right, left] = -angle
        forward = QuantumCircuit(2 * n_orb)
        _rotation(forward, expm(-generator), n_orb)
        if index == 0 and slater_prep:
            # The first rotation acts on |HF>: prepare the rotated determinant directly.
            _slater_preparation(circuit, expm(-generator), n_orb, spec.n_alpha, spec.n_beta)
        else:
            circuit.compose(forward, inplace=True)
        for orbital, angle in enumerate(layer.j_same):
            for offset in (0, n_orb):
                circuit.cp(float(angle), offset + orbital, offset + orbital + 1)
        for index, angle in enumerate(layer.j_opp):
            orbital = 4 * index
            circuit.cp(float(angle), orbital, n_orb + orbital)
        if spec.final_rotation:
            circuit.compose(forward.inverse(), inplace=True)
    if spec.measurement_basis is not None:
        from chemflow.quantum.pauli import rotation_circuit

        circuit.compose(rotation_circuit(spec.measurement_basis), inplace=True)
    return circuit


# Identifies the compiled circuit a QPU job ran (part of the lqcloud job identity).
COMPILER_VERSION = "chain-layout-search-v4-construction-search"
# compile_native(search=False): identity placement, one seed (the compiler before v2).
PLAIN_COMPILER = "chain-identity-seed91-v4"
TRANSPILER_SEEDS = (91, 1, 2, 3, 4)


def twirl(native: QuantumCircuit, seed: int) -> QuantumCircuit:
    """Random Pauli frame around every CZ, merged into single-qubit gates (same CZ count)."""
    from qiskit.circuit import pauli_twirl_2q_gates

    count = native.num_qubits
    edges = [[index, index + 1] for index in range(count - 1)]
    edges += [[right, left] for left, right in edges]
    twirled = pauli_twirl_2q_gates(native, twirling_gate="cz", seed=seed)
    return transpile(twirled, basis_gates=["rz", "sx", "x", "cz"], coupling_map=edges,
                     initial_layout=list(range(count)), optimization_level=1, seed_transpiler=0)


def compile_spec(spec: CircuitSpec, *, search: bool = True) -> QuantumCircuit:
    """Native circuit for ``spec`` with the fewest CZ over the equivalent constructions.

    An LUCJ circuit's first orbital rotation can be built as the optimal Slater-determinant
    preparation (fewer Givens rotations from 12 qubits up) or as the general rotation
    network (which transpiles shorter at 8 qubits); both prepare the same state. Each is
    compiled with ``compile_native``.
    """
    candidates = [build_circuit(spec)]
    if search and spec.ansatz == "lucj" and spec.layers:
        candidates.append(build_circuit(spec, slater_prep=False))
    compiled = [compile_native(circuit, search=search,
                               optimization_level=spec.optimization_level)
                for circuit in candidates]
    best = compiled[0] if len(compiled) == 1 else min(
        compiled, key=lambda native: (native.count_ops().get("cz", 0), native.depth()))
    return best if spec.twirl_seed is None else twirl(best, spec.twirl_seed)


def chain_layouts(count: int) -> list[list[int]]:
    """Initial placements of ``count`` logical qubits on a linear chain to try.

    Identity, and for a two-spin register (alpha qubits 0..n-1, beta n..2n-1) the folded
    order: alpha orbitals run down the first half of the chain and beta orbitals up the
    second, so alpha_0 and beta_0 are neighbours and an orbital rotation's Givens network
    stays within each half.
    """
    layouts = [list(range(count))]
    if count >= 4 and count % 2 == 0:
        half = count // 2
        layouts.append([half - 1 - index for index in range(half)] +
                       [half + index for index in range(half)])
    return layouts


def compile_native(circuit: QuantumCircuit, *, search: bool = True,
                   optimization_level: int = 3) -> QuantumCircuit:
    """Transpile to rz/sx/x/cz on a linear chain, measuring every qubit.

    With ``search`` (the default) it tries each placement of ``chain_layouts`` with every
    seed of ``TRANSPILER_SEEDS`` and keeps the circuit with the fewest CZ gates, then the
    least depth. Measurements stay on the logical qubits, so the bit order of the results
    does not depend on the placement.
    """
    native = circuit.copy()
    native.barrier()
    native.measure_all()
    count = native.num_qubits
    edges = [[index, index + 1] for index in range(count - 1)]
    edges += [[right, left] for left, right in edges]
    layouts = chain_layouts(count) if search else [list(range(count))]
    seeds = TRANSPILER_SEEDS if search else TRANSPILER_SEEDS[:1]
    best = None
    for layout in layouts:
        for seed in seeds:
            candidate = transpile(native, basis_gates=["rz", "sx", "x", "cz"],
                                  coupling_map=edges, initial_layout=layout,
                                  optimization_level=optimization_level,
                                  seed_transpiler=seed)
            key = (candidate.count_ops().get("cz", 0), candidate.depth())
            if best is None or key < best[0]:
                best = (key, candidate)
    return best[1]


def memory_to_index(memory: str, *, q0_left: bool) -> int:
    bits = memory.replace(" ", "")
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("Memory must contain only binary digits")
    return int(bits[::-1] if q0_left else bits, 2)


def to_lqcloud(native: QuantumCircuit):
    from lqcloud import QuantumCircuit as CloudCircuit

    output = CloudCircuit(native.num_qubits, native.num_clbits)
    for instruction in native.data:
        name = instruction.operation.name
        qubits = [native.find_bit(bit).index for bit in instruction.qubits]
        if name == "barrier":
            output.barrier(*qubits)
        elif name == "measure":
            output.measure(qubits[0], native.find_bit(instruction.clbits[0]).index)
        elif name == "rz":
            output.rz(float(instruction.operation.params[0]), qubits[0])
        elif name == "sx":
            output.xhalf(qubits[0])
        elif name == "x":
            output.x(qubits[0])
        elif name == "cz":
            output.cz(*qubits)
        else:
            raise ValueError(f"Unsupported hardware gate: {name}")
    return output
