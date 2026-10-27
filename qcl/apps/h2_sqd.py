"""H2 SQD energy from a local QPU simulator."""

import qcl


@qcl.workflow
def sampled_energy(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    bits = qcl.sample(qcl.kernels.hf(space), shots=128, seed=7).to(qpu)
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector,
                               seed=7).to(cpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = sampled_energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                                   qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    print(plan.explain())
    print(plan.estimate())
    print(plan.run())
