"""Hydrogen chain H_N: equation of state of a correlated 1D solid with quantum and classical solvers.

The equally spaced hydrogen chain is the standard benchmark for correlated
materials (Motta et al., Phys. Rev. X 7, 031059 (2017)): it crosses from a weakly
correlated metal-like chain at short spacing to a strongly correlated Mott-like
insulator when stretched, where single-reference methods break down. Every
spacing R is one pipeline over three device kinds, all orbitals active:

    CPU  RHF/STO-6G, CCSD(T), DMRG (block2), EWF embedding (atomic fragments,
         bath natural orbitals, FCI fragment solver), CCSD amplitudes, SQD
    QPU  CCSD-seeded LUCJ circuit on 2N qubits, sampled on a GPU-hosted
         statevector simulator (no charge; a cloud QPU takes the same program)
    GPU  exact FCI in the full space, the reference for every other method

SQD is reported twice from the same samples: with configuration recovery and the
single-excitation extension (the default), and from the sampled valid
configurations alone (``raw``), which is what the circuit contributes. CCSD does not
converge for the stretched chain (R > 2.4 bohr for H12 and H14), so CCSD(T) is
requested only up to ``--ccsd-max``.

    QCL_POLICY_FILE=qcl/apps/applications-policy.toml \\
        python qcl/apps/hydrogen_chain.py --atoms 12 --json hchain.json
"""

import argparse
import json
import time

import qcl

BOHR = 0.529177210903  # angstrom
DISTANCES = (1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6)  # bohr
METHODS = ("rhf", "ccsd_t", "ewf_bno", "dmrg", "sqd", "sqd_raw")


def chain(n_atoms, spacing_bohr, basis):
    atoms = "; ".join(f"H 0 0 {index * spacing_bohr * BOHR:.10f}" for index in range(n_atoms))
    return qcl.Molecule(atoms, basis)


@qcl.workflow
def hydrogen_chain(distances, n_atoms, cpu, lanes, *, basis="sto-6g", shots=100000,
                   bond_dimension=512, ccsd_max=2.4, seed=11):
    curve = []
    for index, distance in enumerate(distances):
        gpu, qpu = lanes[index % len(lanes)]
        mean_field = qcl.scf.hf(chain(n_atoms, distance, basis)).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(n_atoms, n_atoms)).to(cpu)
        operator, sector = space.hamiltonian, space.sector

        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        kernel = qcl.kernels.lucj(space, params=amplitudes)
        bits = qcl.sample(kernel, shots=shots, seed=seed + index).to(qpu).to(cpu)
        point = {
            "rhf": mean_field.energy,
            "ewf_bno": qcl.embedding.ewf(operator, sector, bath="bno").to(cpu),
            "dmrg": qcl.dmrg.solve(operator, sector,
                                   bond_dimension=bond_dimension).to(cpu).energy,
            "sqd": qcl.sqd.diagonalize(operator, bits, sector).to(cpu),
            "sqd_raw": qcl.sqd.diagonalize(operator, bits, sector, recovery=False,
                                           extend=False).to(cpu),
            "fci": qcl.fci.solve(operator.to(gpu), sector.to(gpu)).to(gpu).energy,
        }
        if distance <= ccsd_max:
            point["ccsd_t"] = qcl.classical.solve(operator, sector,
                                                  method="ccsd_t").to(cpu).energy
        curve.append(point)
    return curve


def lanes_for(cluster, qubits):
    if not cluster.gpus:
        raise SystemExit("This application needs a GPU")
    return [(gpu, qcl.Device(f"sim-{gpu.id}", "qpu", qubits=qubits, simulator_host=gpu.id,
                             simulator_method="statevector")) for gpu in cluster.gpus]


def minimum(distances, energies):
    """Spacing and energy at the vertex of a parabola through the lowest three points."""
    lowest = min(range(len(energies)), key=energies.__getitem__)
    lowest = min(max(lowest, 1), len(energies) - 2)
    (x0, x1, x2), (y0, y1, y2) = distances[lowest - 1:lowest + 2], energies[lowest - 1:lowest + 2]
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 +
         x0 * x1 * (x0 - x1) * y2) / denominator
    return -b / (2 * a), c - b * b / (4 * a)


def _estimate(value):
    return {"value": value.value, "unit": value.unit, "provenance": value.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atoms", type=int, default=12)
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES,
                        help="spacings in bohr")
    parser.add_argument("--basis", default="sto-6g")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--bond-dimension", type=int, default=512)
    parser.add_argument("--ccsd-max", type=float, default=2.4,
                        help="largest spacing (bohr) with a CCSD(T) point")
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    cluster = qcl.Cluster.local()
    plan = hydrogen_chain.compile(tuple(args.distances), args.atoms, cluster.cpus[0],
                                  lanes_for(cluster, 2 * args.atoms), basis=args.basis,
                                  shots=args.shots, bond_dimension=args.bond_dimension,
                                  ccsd_max=args.ccsd_max)
    print(plan.explain())
    started = time.perf_counter()
    curve = plan.run(timeout_s=args.timeout)
    wall = time.perf_counter() - started

    print(f"\nH{args.atoms}/{args.basis}: FCI energy per atom (hartree) and error vs FCI "
          "(mEh per chain); SQD subspace determinants")
    print(f"{'R/bohr':>7} {'FCI E/N':>12} " + " ".join(f"{name:>9}" for name in METHODS) +
          f" {'SQD dets':>9} {'raw dets':>9}")
    for distance, point in zip(args.distances, curve, strict=True):
        exact = point["fci"].value
        print(f"{distance:7.2f} {exact / args.atoms:12.6f} " + " ".join(
            f"{1000 * (point[name].value - exact):9.3f}" if name in point else f"{'n/c':>9}"
            for name in METHODS) +
            f" {point['sqd'].provenance['subspace_dimension']:>9}"
            f" {point['sqd_raw'].provenance['subspace_dimension']:>9}")
    minima = {}
    for name in ("fci", "sqd", "dmrg", "rhf"):
        spacing, energy = minimum(list(args.distances), [point[name].value for point in curve])
        minima[name] = {"spacing_bohr": spacing, "energy_per_atom": energy / args.atoms}
        print(f"{name:>5} minimum: R = {spacing:.3f} bohr, E/N = {energy / args.atoms:.6f}")
    print(f"\nplan wall {wall:.0f} s, worker CPU {plan.cpu_core_s:.0f} core-s")

    output = {"atoms": args.atoms, "basis": args.basis, "distances_bohr": args.distances,
              "shots": args.shots, "bond_dimension": args.bond_dimension, "wall_s": wall,
              "cpu_core_s": plan.cpu_core_s, "plan_hash": plan.hash, "profile": plan.profile(),
              "minima": minima,
              "curve": [{name: _estimate(value) for name, value in point.items()}
                        for point in curve]}
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
