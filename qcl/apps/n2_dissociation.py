"""N2 triple-bond dissociation with GPU mean fields, QPU sampling and CPU SQD.

Breaking the N2 triple bond is a standard multireference problem: restricted
Hartree-Fock fails badly as the bond stretches, and the full-valence (10e, 8o)
active space becomes strongly correlated. Each bond length is one pipeline
spread over three device kinds:

    GPU  RHF/cc-pVDZ mean field (GPU4PySCF); CASCI reference in the active space
    CPU  active-space Hamiltonian, CCSD amplitudes, SQD diagonalization
    QPU  LUCJ circuit seeded by CCSD, sampled on a QPU or a QPU simulator

Every cross-device value is an explicit ``.to()`` copy; the plan is compiled
with ``strict_transfers=True``, so an implicit transfer is a compile error.
Bond lengths are dealt round-robin over (GPU, QPU) lanes. The SQD - CASCI
difference is computed after the run: QCL refuses to subtract Estimates from
different methods (QCL-E103), because the result would carry no method.

Run from the repository root with the bundled operator policy:

    QCL_POLICY_FILE=qcl/apps/n2-policy.toml python qcl/apps/n2_dissociation.py
    QCL_POLICY_FILE=... python qcl/apps/n2_dissociation.py --cluster cluster.toml

Without ``--cluster`` the local cluster is used: one CPU, every visible GPU and
one statevector QPU simulator per GPU. With a cluster file, QPUs listed there
(for example a cloud QPU) are paired with its GPUs.
"""

import argparse
import json

import qcl

DISTANCES = (0.9, 1.0, 1.098, 1.2, 1.4, 1.6, 1.9, 2.2)  # angstrom
FULL_VALENCE = qcl.Active(10, 8)                         # 16 qubits


@qcl.workflow
def n2_curve(distances, cpu, lanes, basis="cc-pvdz", active=FULL_VALENCE,
             shots=4000, seed=11):
    curve = []
    for index, distance in enumerate(distances):
        gpu, qpu = lanes[index % len(lanes)]
        molecule = qcl.Molecule(f"N 0 0 0; N 0 0 {distance}", basis)

        mean_field = qcl.scf.rhf(molecule).to(gpu)
        space = qcl.active_space(mean_field.to(cpu), active).to(cpu)

        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        kernel = qcl.kernels.lucj(space, params=amplitudes)
        bits = qcl.sample(kernel, shots=shots, seed=seed + index).to(qpu)
        sqd = qcl.sqd.diagonalize(space.hamiltonian, bits.to(cpu), space.sector).to(cpu)

        casci = qcl.fci.solve(space.hamiltonian.to(gpu),
                              space.sector.to(gpu)).to(gpu).energy
        curve.append({"rhf": mean_field.energy, "sqd": sqd, "casci": casci})
    return curve


def lanes_from(cluster):
    if not cluster.gpus or not cluster.qpus:
        raise SystemExit("This application needs at least one GPU and one QPU (or simulator)")
    count = max(len(cluster.gpus), len(cluster.qpus))
    return [(cluster.gpus[index % len(cluster.gpus)], cluster.qpus[index % len(cluster.qpus)])
            for index in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster", help="cluster.toml (default: the local cluster)")
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--active", type=int, nargs=2, default=(10, 8),
                        metavar=("ELECTRONS", "ORBITALS"),
                        help="active space (default: full valence, 10 8)")
    parser.add_argument("--json", help="write the curve with provenance to this file")
    args = parser.parse_args()

    cluster = qcl.Cluster.load(args.cluster) if args.cluster else qcl.Cluster.local()
    plan = n2_curve.compile(tuple(args.distances), cluster.cpus[0], lanes_from(cluster),
                            active=qcl.Active(*args.active), shots=args.shots,
                            strict_transfers=True)
    print(plan.explain())
    print(plan.estimate())
    curve = plan.run()

    print(f"\n{'R/Å':>6} {'E(RHF)':>14} {'E(SQD)':>14} {'±':>8} {'E(CASCI)':>14} "
          f"{'SQD−CASCI/mEh':>14} {'subspace':>9}")
    for distance, point in zip(args.distances, curve, strict=True):
        sqd = point["sqd"]
        print(f"{distance:6.3f} {point['rhf'].value:14.6f} {sqd.value:14.6f} "
              f"{sqd.variance ** 0.5:8.1e} {point['casci'].value:14.6f} "
              f"{1000 * (sqd.value - point['casci'].value):14.3f} "
              f"{sqd.provenance['subspace_dimension']:>9}")
    print(f"\nWorker CPU time: {plan.cpu_core_s:.1f} core-s")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([{"distance": distance,
                        **{key: {"value": value.value, "unit": value.unit,
                                 "variance": value.variance,
                                 "provenance": value.provenance}
                           for key, value in point.items()}}
                       for distance, point in zip(args.distances, curve, strict=True)],
                      handle, indent=2, default=str)


if __name__ == "__main__":
    main()
