"""CCSD-seeded LUCJ sampling followed by SQD on a local QPU simulator."""

import qcl


@qcl.workflow
def lucj_sqd(molecule, active, cpu, qpu, shots=128):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    params = qcl.ccsd.amplitudes(space).to(cpu)
    kernel = qcl.kernels.lucj(space, params=params)
    bits = qcl.sample(kernel, shots=shots, seed=19).to(qpu)
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector).to(cpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = lucj_sqd.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
        qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0],
    )
    print(plan.explain())
    print(plan.run())
