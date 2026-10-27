"""Semiempirical GFN2-xTB energy through the tblite engine."""

import qcl


@qcl.workflow
def energy(molecule, cpu):
    return qcl.engines.tblite.gfn2.energy(molecule).to(cpu)


if __name__ == "__main__":
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")
    print(energy.compile(molecule, qcl.Cluster.local().cpus[0]).run())
