"""Optimize H2 with GFN2-xTB and evaluate the resulting geometry."""

import qcl


@qcl.workflow
def optimize_h2(molecule, cpu):
    geometry = qcl.engines.tblite.gfn2.optimize(molecule).to(cpu)
    energy = qcl.engines.tblite.gfn2.energy(geometry).to(cpu)
    return geometry, energy


if __name__ == "__main__":
    plan = optimize_h2.compile(qcl.Molecule("H 0 0 0; H 0 0 1.1", "sto-3g"),
                               qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
