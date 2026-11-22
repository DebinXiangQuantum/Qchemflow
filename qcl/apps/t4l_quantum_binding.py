"""T4 lysozyme L99A with benzene: whole-protein MM, pocket QM and a quantum QM/MM ligand.

Preparation uses the ChemFlow binding kernels: PDBFixer with PROPKA protonation at
pH 7, GAFF2/AM1-BCC benzene parameters, and relaxation of the added hydrogens near
the ligand. Everything after that is one QCL plan:

    CPU  MM/GBSA of the complex (Amber14/GBn2, the whole protein)
    CPU  counterpoise MP2 interactions of benzene with every pocket side chain
    CPU  AutoDock Vina scores of an alkylbenzene series in the cavity
    CPU  QM/MM RHF and CASCI(6,6) with exact FCI
    GPU  QM/MM CASCI with GPU FCI, the exact reference for the larger space
    GPU  QM/MM CASCI with SQD: CCSD-seeded LUCJ circuits sampled on a GPU
         statevector simulator, then diagonalized in the sampled subspace, with and
         without the single-excitation extension of the leading configurations

The QM region is the benzene; the protein (Amber14, electrostatic embedding) is MM.
Each QM/MM energy is computed for the bound pose and for the ligand moved 100 A out of
the protein, so ``bound - separated`` is the vertical QM/MM interaction energy at that
level, with the same geometries for every solver.

    python qcl/apps/t4l_quantum_binding.py --out runs/applications/t4l --active 14 14
"""

import argparse
import io
import json
import time
from pathlib import Path

import qcl

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "tests" / "data" / "181L.pdb"
AD_TYPES = {"C": "C", "N": "N", "O": "OA", "S": "SA"}
HARTREE_KCAL = 627.509474
# Benzene and alkylbenzenes in the L99A cavity, docked into the same box.
SERIES = {
    "benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "ethylbenzene": "CCc1ccccc1",
    "n-propylbenzene": "CCCc1ccccc1", "n-butylbenzene": "CCCCc1ccccc1",
    "p-xylene": "Cc1ccc(C)cc1", "indole": "c1ccc2[nH]ccc2c1", "phenol": "Oc1ccccc1",
}


def prepare(out: Path) -> dict:
    """Protonated, hydrogen-relaxed complex PDB, ligand force field and docking receptor."""
    from openmm import app, unit

    from chemflow.domains import binding
    from chemflow.runtime.artifacts import read_text_artifact

    text = PDB.read_text()
    protein = binding.protein_structure(text, ph=7.0, chains=["A"])
    ligand = binding.ligand_structure({"pdb_text": text, "residue": "BNZ",
                                       "smiles": "c1ccccc1"})
    combined = binding.complex(protein, ligand)
    parameters = binding.ligand_parameters(ligand)
    evaluated = binding.mmgbsa(combined, parameters)
    pdb_text = read_text_artifact(evaluated.diagnostics["evaluated_pdb_ref"])
    ffxml = out / "benzene-gaff2.xml"
    ffxml.write_text(read_text_artifact(parameters["parameters_ref"]))
    (out / "complex.pdb").write_text(pdb_text)

    structure = app.PDBFile(io.StringIO(pdb_text))
    positions = [list(map(float, row))
                 for row in structure.positions.value_in_unit(unit.nanometer)]
    qm_atoms = [atom.index for atom in structure.topology.atoms()
                if atom.residue.name == "BNZ"]
    separated = [[x + 10.0, y, z] if index in set(qm_atoms) else [x, y, z]
                 for index, (x, y, z) in enumerate(positions)]

    receptor, ligand_xyz = [], []
    for line in text.splitlines():
        if line.startswith("HETATM") and line[17:20] == "BNZ":
            ligand_xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        element = line[76:78].strip()
        if line.startswith("ATOM") and line[21] == "A" and element in AD_TYPES:
            receptor.append(f"{line[:54]}{1.0:6.2f}{0.0:6.2f}    {0.0:+6.3f} "
                            f"{AD_TYPES[element]:<2}")
    center = [sum(column) / len(ligand_xyz) for column in zip(*ligand_xyz, strict=True)]
    return {"pdb_text": pdb_text, "ffxml": str(ffxml), "positions": positions,
            "separated": separated, "qm_atoms": qm_atoms,
            "receptor_pdbqt": "\n".join(receptor) + "\n", "center": center,
            "n_atoms": structure.topology.getNumAtoms(),
            "n_residues": sum(1 for residue in structure.topology.residues()
                              if residue.name != "BNZ"),
            "protein_charge": protein["formal_charge"],
            "protonation_method": protein["protonation_method"]}


@qcl.workflow
def t4l_quantum_binding(system, cpu, gpus, active, basis="6-31g*", shots=100000,
                        mp2_basis="6-31g*", series=None):
    receptor = {"pdb_text": PDB.read_text(), "chains": ["A"]}
    ligand = {"source": {"pdb_text": PDB.read_text(), "residue": "BNZ",
                         "smiles": "c1ccccc1"}}
    forcefields = ("amber14-all.xml", system["ffxml"])

    def qmmm(positions, gpu, **level):
        device = gpu if level.get("solver") in ("gpu_fci", "sqd") else cpu
        return qcl.qmmm.energy(system["pdb_text"], positions, system["qm_atoms"],
                               forcefields=forcefields, basis=basis, **level).to(device)

    sampled = {"solver": "sqd", "sampler": "gpu", "shots": shots, "n_batches": 40, "seed": 3}
    levels = {"rhf": {}, "casci66_fci": {"method": "casci", "cas": (6, 6)}}
    levels["casci66_sqd"] = {"method": "casci", "cas": (6, 6), **sampled}
    levels["casci66_sqd_ext"] = {**levels["casci66_sqd"], "extend": True}
    levels["casci_large_gpu_fci"] = {"method": "casci", "cas": tuple(active),
                                     "solver": "gpu_fci"}
    levels["casci_large_sqd"] = {"method": "casci", "cas": tuple(active), **sampled}
    levels["casci_large_sqd_ext"] = {**levels["casci_large_sqd"], "extend": True}
    qmmm_rows = {}
    for index, (name, level) in enumerate(levels.items()):
        gpu = gpus[index % len(gpus)]
        qmmm_rows[name] = {"bound": qmmm(system["positions"], gpu, **level),
                           "separated": qmmm(system["separated"], gpu, **level)}
    return {
        "mmgbsa": qcl.binding.mmgbsa(receptor, ligand).to(cpu),
        "fragments": qcl.binding.fragments(receptor, ligand, method="mp2",
                                           basis=mp2_basis).to(cpu),
        "docking": {name: qcl.drug.dock(smiles, system["receptor_pdbqt"],
                                        center=system["center"], box=(16, 16, 16),
                                        seed=1).to(cpu)
                    for name, smiles in (series or {}).items()},
        "qmmm": qmmm_rows,
    }


def _row(value):
    return {"value": value.value, "unit": value.unit, "provenance": value.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--active", type=int, nargs=2, default=(14, 14),
                        metavar=("ORBITALS", "ELECTRONS"))
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=6 * 3600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    system = prepare(args.out)
    prepared = time.perf_counter() - started
    print(f"prepared {system['n_residues']} residues, {system['n_atoms']} atoms, "
          f"protein charge {system['protein_charge']:+d} ({system['protonation_method']}) "
          f"in {prepared:.0f} s")

    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        raise SystemExit("This application needs a GPU")
    plan = t4l_quantum_binding.compile(system, cluster.cpus[0], cluster.gpus, args.active,
                                       basis=args.basis, shots=args.shots, series=SERIES)
    print(plan.explain())
    started = time.perf_counter()
    result = plan.run(timeout_s=args.timeout)
    wall = time.perf_counter() - started

    print(f"\nMM/GBSA (Amber14/GBn2, whole protein): {result['mmgbsa'].value:8.2f} kcal/mol")
    fragments = result["fragments"]
    print(f"Pocket MP2/{args.basis} counterpoise sum: "
          f"{fragments.value * HARTREE_KCAL:8.2f} kcal/mol")
    for residue, energy in sorted(fragments.provenance["per_residue"].items(),
                                  key=lambda item: item[1]):
        print(f"    {residue:8s} {energy * HARTREE_KCAL:8.2f}")
    print("\nVina scores (kcal/mol):")
    for name, score in sorted(result["docking"].items(), key=lambda item: item[1].value):
        print(f"    {name:16s} {score.value:7.2f}")
    print(f"\nQM/MM benzene/{args.basis} in the protein field (hartree; interaction kcal/mol):")
    for name, row in result["qmmm"].items():
        bound, separated = row["bound"].value, row["separated"].value
        dets = [row[state].provenance["qm_solver"].get("n_determinants", "")
                for state in ("bound", "separated")]
        print(f"    {name:20s} {bound:16.8f} {separated:16.8f} "
              f"{(bound - separated) * HARTREE_KCAL:9.3f}   {dets[0]} {dets[1]}")
    print(f"\nprepare {prepared:.0f} s, plan wall {wall:.0f} s, "
          f"worker CPU {plan.cpu_core_s:.0f} core-s")

    output = {"system": {key: system[key] for key in
                         ("n_atoms", "n_residues", "protein_charge", "protonation_method",
                          "qm_atoms", "center")},
              "active": list(args.active), "basis": args.basis, "shots": args.shots,
              "prepare_s": prepared, "wall_s": wall, "cpu_core_s": plan.cpu_core_s,
              "plan_hash": plan.hash, "profile": plan.profile(),
              "mmgbsa": _row(result["mmgbsa"]), "fragments": _row(fragments),
              "docking": {name: _row(value) for name, value in result["docking"].items()},
              "qmmm": {name: {state: _row(value) for state, value in row.items()}
                       for name, row in result["qmmm"].items()}}
    (args.out / "result.json").write_text(json.dumps(output, indent=1, default=str))


if __name__ == "__main__":
    main()
