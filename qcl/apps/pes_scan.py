"""Rigid dihedral potential-energy scan on CPU."""

import qcl


@qcl.workflow
def rigid_dihedral_scan(molecule, atoms, angles, cpu):
    def point(angle, lane):
        geometry = molecule.set_dihedral(*atoms, angle)
        return qcl.scf.hf(geometry).to(lane).energy

    return qcl.map(point, angles).to([cpu])


if __name__ == "__main__":
    plan = rigid_dihedral_scan.compile(
        qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g"),
        (0, 1, 2, 3), [60.0, 120.0], qcl.Cluster.local().cpus[0],
    )
    print(plan.explain())
    print(plan.run())
