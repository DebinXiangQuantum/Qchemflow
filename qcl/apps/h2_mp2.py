"""RHF reference followed by MP2 on an explicitly placed device."""

import qcl


@qcl.workflow
def correlated_energy(molecule, cpu):
    reference = qcl.scf.hf(molecule).to(cpu)
    return qcl.mp2(reference).to(cpu).energy


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    print(correlated_energy.compile(molecule, qcl.Cluster.local().cpus[0]).run())
