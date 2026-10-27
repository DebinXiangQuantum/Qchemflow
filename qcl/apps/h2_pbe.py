"""H2 PBE/RKS energy and frontier orbitals on a CPU lane."""

import qcl


@qcl.workflow
def pbe(molecule, cpu):
    mean_field = qcl.dft.rks(molecule, xc="pbe").to(cpu)
    return mean_field.energy, mean_field.homo, mean_field.lumo


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    plan = pbe.compile(molecule, qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
