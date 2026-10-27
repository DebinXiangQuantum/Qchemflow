"""Sequential HF dihedral scan with constrained geometry relaxation on CPU."""

import qcl


@qcl.workflow
def relaxed_dihedral_scan(molecule, atoms, angles, cpu):
    pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                  gradient=qcl.Analytic())

    def point(previous, angle, lane):
        guess = previous.set_dihedral(*atoms, angle)
        optimized = qcl.optimize(pes, guess,
                                 fixed=[qcl.Dihedral(*atoms)]).to(lane)
        return optimized, qcl.scf.hf(optimized).to(lane).energy

    return qcl.scan(point, init=molecule, over=angles).to(cpu)


if __name__ == "__main__":
    plan = relaxed_dihedral_scan.compile(
        qcl.Molecule("H 0 0 0; O 0 0 1; O 0 1.4 1; H 1 1.4 1", "sto-3g"),
        (0, 1, 2, 3), [60.0, 120.0], qcl.Cluster.local().cpus[0],
    )
    print(plan.explain())
    for geometry, energy in plan.run():
        angle = geometry.provenance["method"]["constraints"][0]["value"]
        print(f"{angle:.1f} degrees: {energy.value:.12f} hartree")
