"""Compute H2 three-center AO integrals and reconstruct its ERI."""

import qcl


@qcl.workflow
def density_fitting(molecule, cpu):
    three_center = qcl.int3c(molecule, "weigend").to(cpu)
    gram = qcl.contract("ijp,klp->ijkl", three_center, three_center).to(cpu)
    factors = qcl.cholesky(molecule, 1e-10).to(cpu)
    eri = qcl.contract("Pij,Pkl->ijkl", factors, factors).to(cpu)
    return three_center, gram, factors, eri


if __name__ == "__main__":
    cluster = qcl.Cluster.local()
    plan = density_fitting.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), cluster.cpus[0])
    print(plan.explain())
    three_center, gram, factors, eri = plan.run()
    print(three_center.shape, gram.shape, factors.shape, eri.shape)
