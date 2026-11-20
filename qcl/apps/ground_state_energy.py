"""Molecular ground-state energy: exact active-space CI, optionally with a QPU SQD estimate.

The catalogue application ``ground_state_energy``. The exact active-space energy
(CASCI) is always computed; with a QPU (hardware or simulator) the same active
space is also solved by CCSD-seeded LUCJ sampling and SQD, and both are returned
so that the sampled estimate is reported next to its exact reference.

    python qcl/apps/ground_state_energy.py "H 0 0 0; H 0 0 0.74" --basis sto-3g --active 2 2
    QCL_POLICY_FILE=qcl/apps/local-policy.toml \\
        python qcl/apps/ground_state_energy.py "N 0 0 0; N 0 0 1.098" --basis cc-pvdz \\
        --active 6 6 --qpu
"""

import argparse

import qcl


@qcl.workflow
def ground_state_energy(molecule, active, cpu, qpu=None, shots=4000, seed=7):
    mean_field = qcl.scf.hf(molecule).to(cpu)
    space = qcl.active_space(mean_field, active).to(cpu)
    result = {"hf": mean_field.energy,
              "casci": qcl.fci.solve(space.hamiltonian, sector=space.sector).to(cpu).energy}
    if qpu is not None:
        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        kernel = qcl.kernels.lucj(space, params=amplitudes)
        bits = qcl.sample(kernel, shots=shots, seed=seed).to(qpu)
        result["sqd"] = qcl.sqd.diagonalize(space.hamiltonian, bits.to(cpu),
                                            space.sector).to(cpu)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("atoms")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--active", type=int, nargs=2, required=True,
                        metavar=("ELECTRONS", "ORBITALS"))
    parser.add_argument("--qpu", action="store_true", help="add an SQD estimate")
    parser.add_argument("--shots", type=int, default=4000)
    args = parser.parse_args()
    cluster = qcl.Cluster.local()
    plan = ground_state_energy.compile(
        qcl.Molecule(args.atoms, args.basis), qcl.Active(*args.active), cluster.cpus[0],
        cluster.qpus[0] if args.qpu else None, shots=args.shots, strict_transfers=True)
    print(plan.explain())
    for name, value in plan.run().items():
        print(f"{name:6s} {value.value:.10f} {value.unit}")


if __name__ == "__main__":
    main()
