"""Finite-shot H2 VQE on a local QPU simulator."""

import qcl


@qcl.workflow
def h2_vqe(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    qubit = qcl.qubit_hamiltonian(space.hamiltonian).to(cpu)
    return qcl.vqe.sampled(qubit, space.sector,
                           ansatz="lucj", optimizer="cobyla",
                           max_evaluations=3, shots_per_group=128,
                           n_groups=5, seed=7).to(qpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = h2_vqe.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    print(plan.explain())
    print(plan.estimate())
    print(plan.run())
