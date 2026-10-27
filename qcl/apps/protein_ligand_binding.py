"""Model-defined T4 lysozyme L99A / benzene single-point MM/GBSA."""

from pathlib import Path

import qcl


@qcl.workflow
def protein_ligand_binding(receptor, ligand, cpu, frames=0, seed=0):
    return qcl.binding.mmgbsa(receptor, ligand, frames=frames, seed=seed).to(cpu)


if __name__ == "__main__":
    pdb = (Path(__file__).resolve().parents[2] / "tests" / "data" / "181L.pdb").read_text()
    plan = protein_ligand_binding.compile(
        {"pdb_text": pdb, "chains": ["A"]},
        {"source": {"pdb_text": pdb, "residue": "BNZ", "smiles": "c1ccccc1"}},
        qcl.Cluster.local().cpus[0],
    )
    print(plan.explain())
    print(plan.run(timeout_s=1800))
