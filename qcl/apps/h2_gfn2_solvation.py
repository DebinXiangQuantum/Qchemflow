"""GFN2-xTB water solvation energy from explicit gas and ALPB calculations."""

import qcl


@qcl.workflow
def solvation(molecule, cpu):
    gas = qcl.engines.tblite.gfn2.energy(molecule).to(cpu)
    water = qcl.engines.tblite.gfn2.energy(
        molecule, solvent=qcl.ALPB("water")).to(cpu)
    return water - gas


if __name__ == "__main__":
    plan = solvation.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                             qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
