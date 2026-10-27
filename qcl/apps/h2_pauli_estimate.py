"""Finite-shot H2 Pauli expectation on a local QPU simulator."""

import qcl


@qcl.workflow
def h2_pauli_estimate(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    operator = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.estimate(operator, qcl.kernels.hf(space),
                        n_groups=5, shots_per_group=128, seed=7).to(qpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = h2_pauli_estimate.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
        qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    print(plan.explain())
    print(plan.estimate())
    print(plan.run())
