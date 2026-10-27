"""Constrained HF geometry relaxation with finite-difference PES gradients."""

import qcl


@qcl.workflow
def fixed_bond_relaxation(molecule, cpu, bond):
    pes = qcl.PES(lambda geometry, lane: qcl.scf.hf(geometry).to(lane),
                  gradient=qcl.FiniteDifference(step=1e-3))
    return qcl.optimize(pes, molecule, fixed=[qcl.Bond(*bond)]).to(cpu)


if __name__ == "__main__":
    plan = fixed_bond_relaxation.compile(
        qcl.Molecule("O 0 0 0; H 0 0 1.0; H 0.8 0 -0.6", "sto-3g"),
        qcl.Cluster.local().cpus[0], (0, 1),
    )
    print(plan.explain())
    print(plan.run())
