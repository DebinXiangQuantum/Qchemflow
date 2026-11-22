"""Diels-Alder cycloaddition, butadiene + ethylene -> cyclohexene: barrier with quantum solvers.

The prototype C-C bond-forming reaction of synthesis, as one QCL plan:

    CPU  HF/6-31G* geometries: Sella transition-state search with an IRC that must
         connect the reactants to cyclohexene, and optimizations of cyclohexene,
         s-trans butadiene and ethylene
    CPU  RHF, MP2 and frozen-core CCSD(T) single points (the classical ladder)
    CPU  CASCI(pi) with exact FCI: six pi electrons at the transition state and in
         cyclohexene, (4e,4o) butadiene and (2e,2o) ethylene, so the barrier is
         size-consistent
    QPU  CCSD-seeded LUCJ circuits for each of those spaces and for a 28-qubit
         CASCI(14e,14o) at the transition state and product, sampled on GPU-hosted
         statevector simulators; SQD on the CPU
    GPU  CASCI(14e,14o) with exact FCI, the reference for the 28-qubit SQD
    CPU  DMRG of the same CASCI(14e,14o) Hamiltonians

Starting structures come from one RDKit embedding of cyclohexene: the TS guess
moves the ethylene unit out until the forming bonds are about 2.2 A, and the
reactant complex used only for the IRC connectivity check moves it 2 A further.
Active orbitals are the canonical RHF orbitals nearest the Fermi level.

    QCL_POLICY_FILE=qcl/apps/applications-policy.toml \\
        python qcl/apps/diels_alder.py --json diels_alder.json
"""

import argparse
import json
import time

import numpy as np

import qcl

HARTREE_KCAL = 627.509474
PI_SPACES = {"transition": (6, 6), "product": (6, 6), "butadiene": (4, 4), "ethylene": (2, 2)}


def starting_geometries():
    """Atom-mapped TS guess, reactant complex and product, plus the free reactants.

    Cyclohexene ``C1C=CCCC1`` numbers the butadiene carbons 0-3 and the ethylene
    carbons 4 and 5; the bonds 3-4 and 5-0 form in the reaction.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def embed(smiles):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        options = AllChem.ETKDGv3()
        options.randomSeed = 7
        AllChem.EmbedMolecule(molecule, options)
        AllChem.MMFFOptimizeMolecule(molecule)
        return molecule, molecule.GetConformer().GetPositions()

    def text(molecule, coordinates):
        return "; ".join(f"{atom.GetSymbol()} {x:.8f} {y:.8f} {z:.8f}"
                         for atom, (x, y, z) in zip(molecule.GetAtoms(), coordinates,
                                                    strict=True))

    ring, positions = embed("C1C=CCCC1")
    ethylene = sorted({4, 5} | {neighbor.GetIdx() for index in (4, 5)
                                for neighbor in ring.GetAtomWithIdx(index).GetNeighbors()
                                if neighbor.GetSymbol() == "H"})
    direction = positions[[4, 5]].mean(0) - positions[[0, 1, 2, 3]].mean(0)
    direction /= np.linalg.norm(direction)

    def shifted(distance):
        moved = positions.copy()
        moved[ethylene] += distance * direction
        return moved

    forming = np.linalg.norm(positions[3] - positions[4])
    butadiene, butadiene_xyz = embed("C=CC=C")       # s-trans, the reactant minimum
    ethene, ethene_xyz = embed("C=C")
    return {"guess": text(ring, shifted(2.2 - forming)),
            "complex": text(ring, shifted(2.2 - forming + 2.0)),
            "product": text(ring, positions), "butadiene": text(butadiene, butadiene_xyz),
            "ethylene": text(ethene, ethene_xyz)}


def frozen_core(atoms, basis):
    """All orbitals but the carbon 1s cores, with their electrons."""
    from pyscf import gto

    molecule = gto.M(atom=atoms, basis=basis, verbose=0)
    cores = sum(1 for index in range(molecule.natm) if molecule.atom_charge(index) > 2)
    return qcl.Active(molecule.nelectron - 2 * cores, molecule.nao_nr() - cores)


@qcl.workflow
def diels_alder(geometries, cpu, lanes, *, basis="6-31g*", large=(14, 14), shots=200000,
                bond_dimension=500, seed=5):
    def molecule(name):
        return qcl.Molecule(geometries[name], basis)

    structures = {
        "transition": qcl.engines.sella.saddle(molecule("guess"), molecule("complex"),
                                               molecule("product")).to(cpu),
        "product": qcl.optimize(molecule("product")).to(cpu),
        "butadiene": qcl.optimize(molecule("butadiene")).to(cpu),
        "ethylene": qcl.optimize(molecule("ethylene")).to(cpu),
    }
    rows = {}
    for index, (name, structure) in enumerate(structures.items()):
        gpu, qpu = lanes[index % len(lanes)]
        mean_field = qcl.scf.hf(structure).to(cpu)
        composition = geometries["guess" if name == "transition" else name]
        valence = qcl.active_space(mean_field, frozen_core(composition, basis)).to(cpu)
        row = {"rhf": mean_field.energy, "mp2": qcl.mp2(mean_field).to(cpu).energy,
               "ccsd_t": qcl.classical.solve(valence.hamiltonian, valence.sector,
                                             method="ccsd_t").to(cpu).energy}
        spaces = {"pi": PI_SPACES[name]}
        if name in ("transition", "product"):
            spaces["large"] = tuple(large)
        for label, active in spaces.items():
            space = qcl.active_space(mean_field, qcl.Active(*active)).to(cpu)
            amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
            bits = qcl.sample(qcl.kernels.lucj(space, params=amplitudes), shots=shots,
                              seed=seed + index).to(qpu).to(cpu)
            row[f"sqd_{label}"] = qcl.sqd.diagonalize(space.hamiltonian, bits,
                                                      space.sector).to(cpu)
            row[f"sqd_{label}_raw"] = qcl.sqd.diagonalize(
                space.hamiltonian, bits, space.sector, recovery=False,
                extend=False).to(cpu)
            if label == "pi":
                row["casci_pi"] = qcl.fci.solve(space.hamiltonian,
                                                space.sector).to(cpu).energy
            else:
                row["casci_large"] = qcl.fci.solve(space.hamiltonian.to(gpu),
                                                   space.sector.to(gpu)).to(gpu).energy
                row["dmrg_large"] = qcl.dmrg.solve(space.hamiltonian, space.sector,
                                                   bond_dimension=bond_dimension
                                                   ).to(cpu).energy
        rows[name] = row
    return {"structures": structures, "energies": rows}


def lanes_for(cluster, qubits):
    if not cluster.gpus:
        raise SystemExit("This application needs a GPU")
    return [(gpu, qcl.Device(f"sim-{gpu.id}", "qpu", qubits=qubits, simulator_host=gpu.id,
                             simulator_method="statevector")) for gpu in cluster.gpus]


def forming_bonds(atoms):
    rows = np.array([[float(value) for value in atom.split()[1:4]]
                     for atom in atoms.replace("\n", ";").split(";") if atom.strip()])
    return float(np.linalg.norm(rows[3] - rows[4])), float(np.linalg.norm(rows[5] - rows[0]))


def _atoms(value):
    """Atoms of a returned structure: a Molecule, or the molecule of a GeometryResult."""
    return getattr(value, "molecule", value).atoms


def _dets(value):
    return value.provenance.get("subspace_dimension")


def _estimate(value):
    return {"value": value.value, "unit": value.unit, "provenance": value.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--large", type=int, nargs=2, default=(14, 14),
                        metavar=("ELECTRONS", "ORBITALS"))
    parser.add_argument("--shots", type=int, default=200000)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    cluster = qcl.Cluster.local()
    plan = diels_alder.compile(starting_geometries(), cluster.cpus[0],
                               lanes_for(cluster, 2 * args.large[1]), basis=args.basis,
                               large=tuple(args.large), shots=args.shots)
    print(plan.explain())
    started = time.perf_counter()
    result = plan.run(timeout_s=args.timeout)
    wall = time.perf_counter() - started
    rows = result["energies"]
    structures = {name: _atoms(value) for name, value in result["structures"].items()}
    first, second = forming_bonds(structures["transition"])
    output = {"basis": args.basis, "large": list(args.large), "shots": args.shots,
              "structures": structures, "forming_bonds_A": [first, second], "wall_s": wall,
              "cpu_core_s": plan.cpu_core_s, "plan_hash": plan.hash, "profile": plan.profile(),
              "energies": {name: {key: _estimate(value) for key, value in row.items()}
                           for name, row in rows.items()}}
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1, default=str)
    print(f"transition state forming bonds (A): {first:.3f} {second:.3f}")

    def relative(key):
        reactants = rows["butadiene"][key].value + rows["ethylene"][key].value
        return (HARTREE_KCAL * (rows["transition"][key].value - reactants),
                HARTREE_KCAL * (rows["product"][key].value - reactants))

    print(f"\n{'method':>12} {'barrier':>9} {'reaction':>9}   (kcal/mol, electronic, vs "
          "s-trans butadiene + ethylene)")
    for key in ("rhf", "mp2", "ccsd_t", "casci_pi", "sqd_pi", "sqd_pi_raw"):
        print(f"{key:>12} {relative(key)[0]:9.2f} {relative(key)[1]:9.2f}")
    print("\nSQD and DMRG minus exact CASCI (mEh), subspace determinants in brackets:")
    for name, row in rows.items():
        exact = row["casci_pi"].value
        line = (f"   {name:10s} pi SQD {1000 * (row['sqd_pi'].value - exact):8.3f} "
                f"[{_dets(row['sqd_pi'])}] raw {1000 * (row['sqd_pi_raw'].value - exact):8.3f} "
                f"[{_dets(row['sqd_pi_raw'])}]")
        if "casci_large" in row:
            exact = row["casci_large"].value
            line += (f" | CAS{tuple(args.large)} SQD "
                     f"{1000 * (row['sqd_large'].value - exact):8.3f} [{_dets(row['sqd_large'])}]"
                     f" raw {1000 * (row['sqd_large_raw'].value - exact):8.3f} "
                     f"[{_dets(row['sqd_large_raw'])}]"
                     f" DMRG {1000 * (row['dmrg_large'].value - exact):8.4f}")
        print(line)
    print(f"\nplan wall {wall:.0f} s, worker CPU {plan.cpu_core_s:.0f} core-s")


if __name__ == "__main__":
    main()
