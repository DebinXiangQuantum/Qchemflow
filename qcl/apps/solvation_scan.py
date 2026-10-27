"""Electrostatic PCM scan with one shared molecular integral set."""

import qcl


@qcl.workflow
def solvation_scan(molecule, solvents, cpu):
    integrals = qcl.int2e(molecule).to(cpu)
    gas = qcl.scf.rhf(molecule, eri=integrals).to(cpu).energy

    def point(name, lane):
        liquid = qcl.scf.rhf(molecule, eri=integrals,
                             solvent=qcl.PCM(name)).to(lane).energy
        return liquid - gas

    return qcl.map(point, solvents).to([cpu])


if __name__ == "__main__":
    plan = solvation_scan.compile(
        qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
        ["water", "methanol", "toluene"], qcl.Cluster.local().cpus[0],
        strict_transfers=True,
    )
    print(plan.explain())
    print(plan.run())
