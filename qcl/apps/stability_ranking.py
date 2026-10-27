"""Rank two geometries using the same placed electronic-energy method."""

import qcl


@qcl.workflow
def stability_ranking(molecules, lanes):
    energies = qcl.map(lambda molecule, lane: qcl.scf.rhf(molecule).to(lane).energy,
                       molecules).to(lanes)
    return qcl.rank(energies)


if __name__ == "__main__":
    molecules = [qcl.Molecule(f"H 0 0 0; H 0 0 {distance}", "sto-3g")
                 for distance in (1.5, 0.74)]
    plan = stability_ranking.compile(molecules, qcl.Cluster.local().cpus)
    print(plan.explain())
    print(plan.run())
