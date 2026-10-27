"""H2 HF state sampling on a local QPU simulator."""

import qcl


@qcl.workflow
def sample_hf(molecule, active, cpu, qpu):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    return qcl.sample(qcl.kernels.hf(space), shots=128, seed=7).to(qpu)


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = sample_hf.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                             qcl.Active(2, 2), cluster.cpus[0], cluster.qpus[0])
    print(plan.explain())
    print(plan.estimate())
    print(plan.run())
