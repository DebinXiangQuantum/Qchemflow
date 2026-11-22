"""Error budget of stored MQ02 SQD runs: readout, post-selection and idle decoherence.

Offline: reads the hardware and ideal bitstrings that ``qcl/apps/qpu_sqd_experiment.py``
stored, the QPU job ledger (each job's physical layout) and a device calibration
snapshot. It spends no QPU time. For every geometry it reports

    F_raw       Hellinger fidelity of the hardware distribution to the ideal one
    F_ro        that fidelity predicted from readout error alone: the ideal
                distribution pushed through the calibrated per-qubit confusion matrices
    F_mit       fidelity after tensored readout-error inversion of the hardware counts
    F_post      fidelity after keeping only shots in the target (N_alpha, N_beta) sector
    F_mit+post  both
    valid       fraction of hardware shots in the target sector, raw and mitigated

and for each circuit the native CZ count, the ASAP-scheduled duration and per-qubit idle
time under assumed gate durations (the calibration has none), with the fidelity that a
product of independent error channels predicts:

    gates       product of the calibrated CZ and single-qubit (XEB) gate fidelities
    +readout    times the readout-only factor F_ro
    +T2*        times exp(-t/T2*) per qubit over its active window (free evolution)
    +T2 echo    the same with the echo T2: what ideal dynamical decoupling could reach

    python qcl/apps/qpu_error_budget.py --runs runs/qpu --ledger ~/.chemflow/qpu/MQ02 \\
        --calibration runs/qpu/mq02_calibration.json --json error_budget.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

GEOMETRY = {"h2": "H 0 0 0; H 0 0 {r}", "n2": "N 0 0 0; N 0 0 {r}"}


def dense(counts, n_qubits):
    vector = np.zeros(1 << n_qubits)
    for key, count in counts.items():
        vector[int(key)] += count
    return vector / vector.sum()


def hellinger(first, second):
    return float(np.sqrt(first * second).sum() ** 2)


def confusion(calibration, layout):
    """Per logical qubit, M[measured, prepared] from the physical qubit's readout fidelities."""
    metrics = {item["id"]: item for item in calibration["qubit_metrics"]}
    matrices = []
    for physical in layout:
        f0, f1 = metrics[physical]["measure_f0"], metrics[physical]["measure_f1"]
        matrices.append(np.array([[f0, 1 - f1], [1 - f0, f1]]))
    return matrices


def apply_per_qubit(vector, matrices):
    """Apply one 2x2 matrix to each qubit axis; logical qubit i is bit i of the index."""
    n_qubits = len(matrices)
    tensor = vector.reshape((2,) * n_qubits)
    for qubit, matrix in enumerate(matrices):
        axis = n_qubits - 1 - qubit                  # C order: the last axis is bit 0
        tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def mitigate(vector, matrices):
    """Tensored inversion, then the nearest distribution by clipping and renormalizing."""
    corrected = apply_per_qubit(vector, [np.linalg.inv(matrix) for matrix in matrices])
    corrected = np.clip(corrected, 0, None)
    return corrected / corrected.sum()


def sector_mask(n_orb, n_alpha, n_beta):
    index = np.arange(1 << (2 * n_orb), dtype=np.int64)
    alpha, beta = index & ((1 << n_orb) - 1), index >> n_orb
    return (np.bitwise_count(alpha) == n_alpha) & (np.bitwise_count(beta) == n_beta)


def post_select(vector, mask):
    kept = np.where(mask, vector, 0.0)
    return kept / kept.sum() if kept.sum() else kept


def layouts(ledger):
    found = {}
    for path in Path(ledger).expanduser().glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("job_id") and record.get("layout"):
            found[record["job_id"]] = record["layout"]
    return found


def native_circuit(run, point):
    from chemflow.backends.circuits import build_circuit, compile_native
    from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals
    from chemflow.impls.pyscf_impls import build_molecule
    from chemflow.runtime.artifacts import ArtifactStore
    from chemflow.types import CircuitSpec
    from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

    electrons, orbitals = run["active"]
    parameters = point["bits_qpu"]["provenance"].get("kernel_params")
    if parameters is None:
        store = ArtifactStore()
        data = build_molecule(GEOMETRY[run["system"]].format(r=point["distance"]),
                              ElectronicModel(basis=run["basis"]), store,
                              active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
        seed = ccsd_seed_from_integrals(data.hamiltonian, data.sector, store)
        parameters = (seed or LucjParams.zeros(orbitals)).to_vector().tolist()
    spec = CircuitSpec(ansatz="lucj", n_orb=orbitals, n_alpha=electrons // 2,
                       n_beta=electrons // 2, layers=[parameters])
    return compile_native(build_circuit(spec))


def schedule(native, one_qubit_ns, cz_ns):
    """ASAP schedule: total duration, and per qubit its active window and busy time (ns)."""
    durations = {"rz": 0.0, "sx": one_qubit_ns, "x": 2 * one_qubit_ns, "cz": cz_ns}
    clock = [0.0] * native.num_qubits
    first = [None] * native.num_qubits
    busy = [0.0] * native.num_qubits
    gates = {"cz": [], "1q": []}
    for instruction in native.data:
        name = instruction.operation.name
        if name not in durations:
            continue
        qubits = [native.find_bit(qubit).index for qubit in instruction.qubits]
        start = max(clock[qubit] for qubit in qubits)
        for qubit in qubits:
            if first[qubit] is None and durations[name]:
                first[qubit] = start
            clock[qubit] = start + durations[name]
            busy[qubit] += durations[name]
        if name == "cz":
            gates["cz"].append(tuple(qubits))
        elif durations[name]:
            gates["1q"].append(qubits[0])
    end = max(clock)
    windows = [end - (start if start is not None else end) for start in first]
    return end, windows, busy, gates


def predicted(calibration, layout, gates, windows, readout_factor):
    metrics = {item["id"]: item for item in calibration["qubit_metrics"]}
    couplers = {tuple(sorted(item["qubits"])): item["cz_fidelity"]
                for item in calibration["coupler_metrics"]}
    gate = 1.0
    for left, right in gates["cz"]:
        gate *= couplers[tuple(sorted((layout[left], layout[right])))]
    for qubit in gates["1q"]:
        gate *= metrics[layout[qubit]].get("xeb_fidelity", 1.0)
    free = math.exp(-sum(window / 1000 / metrics[layout[qubit]]["t2_star"]
                         for qubit, window in enumerate(windows)))
    echo = math.exp(-sum(window / 1000 / metrics[layout[qubit]]["t2"]
                         for qubit, window in enumerate(windows)))
    return {"gates": gate, "gates_readout": gate * readout_factor,
            "with_t2_star": gate * readout_factor * free,
            "with_t2_echo": gate * readout_factor * echo}


def analyse(path, calibration, jobs, one_qubit_ns, cz_ns):
    from chemflow.backends.calibration import best_layout

    run = json.loads(Path(path).read_text())
    electrons, orbitals = run["active"]
    n_qubits = 2 * orbitals
    mask = sector_mask(orbitals, electrons // 2, electrons // 2)
    native = native_circuit(run, run["points"][0])
    duration, windows, busy, gates = schedule(native, one_qubit_ns, cz_ns)
    rows = []
    for point in run["points"]:
        hardware = point["bits_qpu"]
        layout = next((jobs[job] for job in hardware.get("job_ids", []) if job in jobs),
                      None) or best_layout({"qubits": calibration["qubits"],
                                            "properties": calibration}, n_qubits)
        matrices = confusion(calibration, layout)
        ideal = dense(point["bits_sim"]["counts"], n_qubits)
        measured = dense(hardware["counts"], n_qubits)
        mitigated = mitigate(measured, matrices)
        readout_only = apply_per_qubit(ideal, matrices)
        ideal_sector = post_select(ideal, mask)
        rows.append({
            "distance": point["distance"], "layout": layout,
            "F_raw": hellinger(measured, ideal), "F_ro": hellinger(readout_only, ideal),
            "F_mit": hellinger(mitigated, ideal),
            "F_post": hellinger(post_select(measured, mask), ideal_sector),
            "F_mit_post": hellinger(post_select(mitigated, mask), ideal_sector),
            "valid_raw": float(measured[mask].sum()), "valid_mit": float(mitigated[mask].sum()),
            "valid_readout_only": float(readout_only[mask].sum()),
            "valid_uniform": float(mask.mean()),
        })
    readout_factor = float(np.mean([row["F_ro"] for row in rows]))
    first_layout = rows[0]["layout"]
    return {"file": Path(path).name, "system": run["system"], "active": run["active"],
            "qubits": n_qubits, "shots": run["shots"],
            "cz": len(gates["cz"]), "one_qubit_gates": len(gates["1q"]),
            "duration_us": duration / 1000, "mean_window_us": float(np.mean(windows)) / 1000,
            "mean_idle_fraction": float(np.mean([1 - b / w if w else 0
                                                 for b, w in zip(busy, windows, strict=True)])),
            "predicted": predicted(calibration, first_layout, gates, windows, readout_factor),
            "points": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True, help="directory of *_mq02.json")
    parser.add_argument("--ledger", type=Path, required=True, help="QPU job ledger directory")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--one-qubit-ns", type=float, default=20.0)
    parser.add_argument("--cz-ns", type=float, nargs="+", default=(40.0, 100.0))
    parser.add_argument("--json")
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    jobs = layouts(args.ledger)
    order = ["h2_mq02.json", "n2_44_mq02.json", "n2_66_mq02.json", "n2_108_mq02.json",
             "n2_1010_mq02.json", "n2_1212_mq02.json"]
    report = {"calibration": calibration.get("last_update"),
              "one_qubit_ns": args.one_qubit_ns, "runs": {}}
    for cz_ns in args.cz_ns:
        report["runs"][str(cz_ns)] = [analyse(args.runs / name, calibration, jobs,
                                              args.one_qubit_ns, cz_ns)
                                      for name in order if (args.runs / name).exists()]
    for cz_ns, runs in report["runs"].items():
        print(f"\nCZ {cz_ns} ns, single-qubit {args.one_qubit_ns} ns, calibration "
              f"{report['calibration']}")
        print(f"{'run':>18} {'qb':>3} {'CZ':>4} {'T/us':>6} {'idle':>5} | "
              f"{'F_raw':>7} {'F_ro':>6} {'F_mit':>7} {'F_post':>7} {'F_m+p':>7} | "
              f"{'valid':>6} {'v_mit':>6} {'v_unif':>6} | "
              f"{'gates':>7} {'+ro':>7} {'+T2*':>9} {'+T2echo':>9}")
        for run in runs:
            mean = {key: float(np.mean([row[key] for row in run["points"]]))
                    for key in ("F_raw", "F_ro", "F_mit", "F_post", "F_mit_post",
                                "valid_raw", "valid_mit", "valid_uniform")}
            model = run["predicted"]
            print(f"{run['file'][:-5]:>18} {run['qubits']:3d} {run['cz']:4d} "
                  f"{run['duration_us']:6.2f} {run['mean_idle_fraction']:5.2f} | "
                  f"{mean['F_raw']:7.4f} {mean['F_ro']:6.3f} {mean['F_mit']:7.4f} "
                  f"{mean['F_post']:7.4f} {mean['F_mit_post']:7.4f} | "
                  f"{mean['valid_raw']:6.3f} {mean['valid_mit']:6.3f} "
                  f"{mean['valid_uniform']:6.3f} | {model['gates']:7.3f} "
                  f"{model['gates_readout']:7.3f} {model['with_t2_star']:9.2e} "
                  f"{model['with_t2_echo']:9.2e}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
