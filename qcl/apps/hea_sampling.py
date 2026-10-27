"""Eight-qubit HEA sampling on a local QPU simulator."""

import qcl


@qcl.quantum_kernel
def hea(qubits: qcl.Qubits[8], theta: qcl.Params[16]):
    for index in range(8):
        qcl.ry(qubits[index], theta[index])
    for index in range(7):
        qcl.cz(qubits[index], qubits[index + 1])
    for index in range(8):
        qcl.ry(qubits[index], theta[8 + index])


@qcl.workflow
def sample_hea(qpu, parameters):
    return qcl.sample(hea, params=parameters, shots=128, seed=7).to(qpu)


if __name__ == "__main__":
    plan = sample_hea.compile(qcl.Cluster.local().qpus[0], [0.0] * 16)
    print(plan.run())
