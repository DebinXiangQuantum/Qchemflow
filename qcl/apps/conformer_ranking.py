"""ETKDG/MMFF conformers ranked by CPU electronic energies."""

import qcl


@qcl.workflow
def conformer_ranking(smiles, cpu, n=6, seed=42):
    conformers = qcl.engines.rdkit.embed(smiles, n=n, seed=seed).to(cpu)
    optimized = qcl.map(lambda conformer, lane:
                        qcl.engines.rdkit.mmff_optimize(conformer).to(lane),
                        conformers).to([cpu])
    unique = qcl.engines.rdkit.dedupe(optimized, rmsd=0.5).to(cpu)
    energies = qcl.map(lambda conformer, lane:
                       qcl.scf.rhf(conformer).to(lane).energy,
                       unique).to([cpu])
    return qcl.rank(energies)


if __name__ == "__main__":
    plan = conformer_ranking.compile("CCO", qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
