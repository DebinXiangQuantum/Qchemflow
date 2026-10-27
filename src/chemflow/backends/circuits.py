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


def build_circuit(spec: CircuitSpec) -> QuantumCircuit:
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
    for vector in spec.layers if spec.ansatz == "lucj" else ():
        layer = LucjParams.from_vector(np.asarray(vector), n_orb)
        if len(vector) != layer.size:
            raise ValueError("Invalid LUCJ parameter vector")
        generator = np.zeros((n_orb, n_orb))
        for angle, (left, right) in zip(layer.kparams, combinations(range(n_orb), 2)):
            generator[left, right] = angle
            generator[right, left] = -angle
        forward = QuantumCircuit(2 * n_orb)
        _rotation(forward, expm(-generator), n_orb)
        circuit.compose(forward, inplace=True)
        for orbital, angle in enumerate(layer.j_same):
            for offset in (0, n_orb):
                circuit.cp(float(angle), offset + orbital, offset + orbital + 1)
        for index, angle in enumerate(layer.j_opp):
            orbital = 4 * index
            circuit.cp(float(angle), orbital, n_orb + orbital)
        circuit.compose(forward.inverse(), inplace=True)
    if spec.measurement_basis is not None:
        from chemflow.quantum.pauli import rotation_circuit

        circuit.compose(rotation_circuit(spec.measurement_basis), inplace=True)
    return circuit


def compile_native(circuit: QuantumCircuit) -> QuantumCircuit:
    native = circuit.copy()
    native.barrier()
    native.measure_all()
    count = native.num_qubits
    edges = [[index, index + 1] for index in range(count - 1)]
    edges += [[right, left] for left, right in edges]
    return transpile(native, basis_gates=["rz", "sx", "x", "cz"], coupling_map=edges,
                    initial_layout=list(range(count)), optimization_level=3, seed_transpiler=91)


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
