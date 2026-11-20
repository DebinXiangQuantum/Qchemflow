"""Sample-based quantum diagonalization on a real QPU, against an ideal simulator.

For every molecule the same CCSD-seeded LUCJ circuit is sampled twice: on the
hardware QPU and on a noiseless statevector simulator. Each sample set is then
diagonalized by SQD in two ways:

    full   configuration recovery + single-excitation extension (the default)
    raw    sampled valid configurations only (no recovery, no extension)

so the report can separate what the device contributes from what the classical
post-processing contributes. The exact active-space reference (CASCI) is
computed on the GPU. Devices:

    GPU  RHF mean field (GPU4PySCF), CASCI reference
    CPU  active-space Hamiltonian, CCSD amplitudes, four SQD diagonalizations
    QPU  hardware sampling (cloud QPU, e.g. lqcloud MQ02)
    SIM  ideal sampling (statevector simulator hosted on a GPU)

``--schedule`` picks how the plan executes: ``sequential`` (one node at a time),
``lanes`` (every device works concurrently, one node per device at a time) or
``compiled`` (the qcl-sched schedule: lanes plus dead/common-node elimination,
critical-path priorities, several CPU nodes and cloud-QPU jobs at once).
``--overlap`` is short for ``--schedule compiled``.

    QCL_POLICY_FILE=qcl/apps/qpu-experiment-policy.toml \\
        python qcl/apps/qpu_sqd_experiment.py --system n2 --active 10 8 \\
        --distances 1.098 1.6 2.2 --shots 4000 --qpu mq02 --json out.json

The QPU credential is read from LQCLOUD_API_KEY; ``--qpu sim`` replaces the
hardware by a second simulator (no charge) for dry runs.
"""

import argparse
import json
import time

import qcl

SYSTEMS = {
    # name: (geometry template with {r}, basis)
    "h2": ("H 0 0 0; H 0 0 {r}", "sto-3g"),
    "n2": ("N 0 0 0; N 0 0 {r}", "cc-pvdz"),
}


def sqd(space, bits, cpu, **options):
    """SQD of the active-space Hamiltonian in the subspace ``bits`` span, on ``cpu``."""
    return qcl.sqd.diagonalize(space.hamiltonian, bits, space.sector, **options).to(cpu)


@qcl.workflow
def hardware_sqd(molecules, cpu, gpus, qpu, simulator, active, shots, ideal_shots, seed=11):
    rows = []
    for index, molecule in enumerate(molecules):
        gpu = gpus[index % len(gpus)]
        mean_field = qcl.scf.rhf(molecule).to(gpu)
        space = qcl.active_space(mean_field.to(cpu), active).to(cpu)
        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        kernel = qcl.kernels.lucj(space, params=amplitudes)

        hardware = qcl.sample(kernel, shots=shots, seed=seed + index).to(qpu).to(cpu)
        ideal = qcl.sample(kernel, shots=ideal_shots, seed=seed + index).to(simulator).to(cpu)

        rows.append({
            "rhf": mean_field.energy,
            "casci": qcl.fci.solve(space.hamiltonian.to(gpu),
                                   space.sector.to(gpu)).to(gpu).energy,
            "sqd_qpu": sqd(space, hardware, cpu),
            "sqd_qpu_raw": sqd(space, hardware, cpu, recovery=False, extend=False),
            "sqd_sim": sqd(space, ideal, cpu),
            "sqd_sim_raw": sqd(space, ideal, cpu, recovery=False, extend=False),
            "bits_qpu": hardware,
            "bits_sim": ideal,
        })
    return rows


def devices(qpu_kind, qubits):
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        raise SystemExit("This experiment needs a GPU")
    simulator = qcl.Device("sim-ideal", "qpu", qubits=qubits, simulator_host=cluster.gpus[0].id,
                           simulator_method="statevector")
    if qpu_kind == "mq02":
        qpu = qcl.Device("mq02", "qpu", qubits=24, site="remote", billing="metered",
                         provider="lqcloud", backend="MQ02")
    else:
        qpu = qcl.Device("sim-standin", "qpu", qubits=qubits,
                         simulator_host=cluster.gpus[-1].id, simulator_method="statevector")
    return cluster.cpus[0], cluster.gpus, qpu, simulator


def _estimate(value):
    return {"value": value.value, "unit": value.unit, "variance": value.variance,
            "provenance": value.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="h2")
    parser.add_argument("--distances", type=float, nargs="+", required=True)
    parser.add_argument("--active", type=int, nargs=2, required=True,
                        metavar=("ELECTRONS", "ORBITALS"))
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--ideal-shots", type=int, default=100000)
    parser.add_argument("--qpu", choices=("mq02", "sim"), default="sim")
    parser.add_argument("--overlap", action="store_true", help="same as --schedule compiled")
    parser.add_argument("--schedule", choices=("sequential", "lanes", "compiled"),
                        default="sequential")
    parser.add_argument("--processes", choices=("auto", "on", "off"), default="auto",
                        help="CPU lane processes for compiled schedules")
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    template, basis = SYSTEMS[args.system]
    molecules = tuple(qcl.Molecule(template.format(r=r), basis) for r in args.distances)
    cpu, gpus, qpu, simulator = devices(args.qpu, 2 * args.active[1])
    plan = hardware_sqd.compile(molecules, cpu, gpus, qpu, simulator, qcl.Active(*args.active),
                                args.shots, args.ideal_shots, strict_transfers=True)
    estimate = plan.estimate()
    print(json.dumps(estimate, indent=1, default=str))
    mode = "compiled" if args.overlap else args.schedule
    slots = ({lane: 1 for lane in plan.schedule()["slots"]} if mode == "lanes" else None)
    processes = (False if mode == "lanes" else
                 {"auto": None, "on": True, "off": False}[args.processes])
    schedule = (plan.schedule(slots=slots, processes=processes) if mode != "sequential"
                else None)
    if schedule is not None:
        predicted = schedule["predicted"]
        print(f"schedule: predicted {predicted['makespan_s']:.1f} s of "
              f"{predicted['sequential_s']:.1f} s sequential, critical path "
              f"{predicted['critical_path_s']:.1f} s, slots {schedule['slots']}")
    started = time.perf_counter()
    rows = plan.run(timeout_s=args.timeout, overlap=mode != "sequential", slots=slots,
                    processes=processes)
    wall = time.perf_counter() - started

    print(f"\n{'R/Å':>6} {'CASCI':>12} " + " ".join(
        f"{name:>13}" for name in ("QPU", "QPU raw", "SIM", "SIM raw")) + "   (mEh vs CASCI)")
    for distance, row in zip(args.distances, rows, strict=True):
        reference = row["casci"].value
        print(f"{distance:6.3f} {reference:12.6f} " + " ".join(
            f"{1000 * (row[key].value - reference):13.3f}"
            for key in ("sqd_qpu", "sqd_qpu_raw", "sqd_sim", "sqd_sim_raw")))
    print(f"\nwall {wall:.1f} s, worker CPU {plan.cpu_core_s:.1f} core-s")

    output = {
        "system": args.system, "basis": basis, "active": args.active,
        "shots": args.shots, "ideal_shots": args.ideal_shots, "qpu": args.qpu,
        "schedule_mode": mode, "processes": args.processes, "schedule": schedule,
        "wall_s": wall, "cpu_core_s": plan.cpu_core_s,
        "plan_hash": plan.hash, "estimate": estimate, "profile": plan.profile(),
        "points": [{"distance": distance,
                    **{key: _estimate(value) for key, value in row.items()
                       if not key.startswith("bits")},
                    **{key: {"counts": value.counts, "shots": value.shots,
                             "source": value.source, "device": value.device,
                             "job_ids": list(value.job_ids), "provenance": value.provenance}
                       for key, value in row.items() if key.startswith("bits")}}
                   for distance, row in zip(args.distances, rows, strict=True)],
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
