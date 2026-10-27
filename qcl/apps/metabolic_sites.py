"""Empirical C-H site scoring with BDE and solvent-accessible area evidence."""

import qcl


@qcl.workflow
def metabolic_sites(molecule, cpu, level="hf"):
    return qcl.metabolism.sites(molecule, level=level).to(cpu)


if __name__ == "__main__":
    molecule = qcl.Molecule(
        "C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; "
        "H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276", "sto-3g",
    )
    plan = metabolic_sites.compile(molecule, qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
