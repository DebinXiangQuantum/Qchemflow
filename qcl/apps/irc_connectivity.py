"""Trace an explicit HF intrinsic reaction coordinate and check its endpoints."""

import qcl


@qcl.workflow
def irc_connectivity(transition, reactant, product, cpu):
    pes = qcl.PES(lambda molecule, lane: qcl.scf.hf(molecule).to(lane),
                  gradient=qcl.Analytic())
    path = qcl.irc(pes, transition).to(cpu)
    qcl.require(path.connects(reactant, product))
    return path


if __name__ == "__main__":
    plan = irc_connectivity.compile(
        qcl.Molecule("C 0 0 0; N 0 0 1.2; H 1.1 0 .5", "sto-3g"),
        qcl.Molecule("C 0 0 0; N 0 0 1.16; H 0 0 -1.07", "sto-3g"),
        qcl.Molecule("C 0 0 0; N 0 0 1.17; H 0 0 2.17", "sto-3g"),
        qcl.Cluster.local().cpus[0],
    )
    print(plan.explain())
