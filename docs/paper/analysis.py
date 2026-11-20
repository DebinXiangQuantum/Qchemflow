"""Figures, tables and controls for the QCL paper from recorded QPU experiment runs.

Inputs are the JSON files written by ``qcl/apps/qpu_sqd_experiment.py``. The script
adds one control that is not part of the QCL program: SQD on uniformly random
bitstrings (the same number of shots as the hardware run), which shows how much of
an SQD energy the classical post-processing recovers without any quantum signal.

    python docs/paper/analysis.py runs/qpu/h2_mq02.json runs/qpu/n2_108_mq02.json \\
        runs/qpu/n2_1010_mq02.json --out docs/paper
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
QPU, SIM, UNIFORM = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
CHEMICAL_ACCURACY_MEH = 1.6
GEOMETRY = {"h2": "H 0 0 0; H 0 0 {r}", "n2": "N 0 0 0; N 0 0 {r}"}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.markersize": 4, "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def hellinger_fidelity(first, second):
    total_first, total_second = sum(first.values()), sum(second.values())
    overlap = sum(math.sqrt(count / total_first * second.get(key, 0) / total_second)
                  for key, count in first.items())
    return overlap ** 2


def in_sector(mask, n_orb, n_alpha, n_beta):
    half = (1 << n_orb) - 1
    return (mask & half).bit_count() == n_alpha and (mask >> n_orb).bit_count() == n_beta


def uniform_control(run, point, seed=2026):
    """SQD (full and raw) on uniformly random bitstrings with the hardware shot count."""
    from chemflow.impls.pyscf_impls import build_molecule
    from chemflow.impls.sqd import sqd_sector
    from chemflow.runtime.artifacts import ArtifactStore
    from chemflow.types import Samples
    from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

    store = ArtifactStore()
    electrons, orbitals = run["active"]
    data = build_molecule(GEOMETRY[run["system"]].format(r=point["distance"]),
                          ElectronicModel(basis=run["basis"]), store,
                          active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 1 << (2 * orbitals), size=run["shots"], dtype=np.int64)
    keys, counts = np.unique(draws, return_counts=True)
    samples = Samples(counts={str(k): int(c) for k, c in zip(keys, counts, strict=True)},
                      shots=run["shots"], encoding_id="uniform", basis="computational",
                      source="quantum_simulator", seed=seed, device="uniform")
    result = {}
    for name, options in (("full", {}), ("raw", {"recovery": False, "with_ext": False})):
        energy, diagnostics = sqd_sector(data, data.sector, store, samples, seed=7, **options)
        result[name] = {"value": float(energy),
                        "subspace_dimension": diagnostics["subspace_dimension"],
                        "sector_dimension": diagnostics["sector_dimension"]}
    return result


def summarize(run, controls=True, cached=None):
    electrons, orbitals = run["active"]
    n_alpha = n_beta = electrons // 2
    rows = []
    for point in run["points"]:
        reference = point["casci"]["value"]
        hardware = point["bits_qpu"]["counts"]
        ideal = point["bits_sim"]["counts"]
        hardware_shots = sum(hardware.values())
        valid = sum(count for key, count in hardware.items()
                    if in_sector(int(key), orbitals, n_alpha, n_beta)) / hardware_shots
        hf_mask = ((1 << n_alpha) - 1) | (((1 << n_beta) - 1) << orbitals)
        row = {
            "distance": point["distance"], "casci": reference, "rhf": point["rhf"]["value"],
            "valid_fraction": valid,
            "hellinger": hellinger_fidelity(hardware, ideal),
            "hf_weight_qpu": hardware.get(str(hf_mask), 0) / hardware_shots,
            "hf_weight_sim": ideal.get(str(hf_mask), 0) / sum(ideal.values()),
            "unique_qpu": len(hardware), "unique_sim": len(ideal),
            "job_ids": point["bits_qpu"]["job_ids"],
            "sector_dimension": point["sqd_qpu"]["provenance"]["sector_dimension"],
        }
        for key in ("sqd_qpu", "sqd_qpu_raw", "sqd_sim", "sqd_sim_raw"):
            row[key] = point[key]["value"]
            row[key + "_error_meh"] = 1000 * (point[key]["value"] - reference)
            row[key + "_dimension"] = point[key]["provenance"]["subspace_dimension"]
        previous = next((item for item in (cached or ())
                         if abs(item["distance"] - point["distance"]) < 1e-9 and
                         "sqd_uniform" in item), None)
        if previous is not None:
            for key in ("sqd_uniform", "sqd_uniform_raw"):
                row[key] = previous[key]
                row[key + "_error_meh"] = previous[key + "_error_meh"]
                row[key + "_dimension"] = previous[key + "_dimension"]
        elif controls:
            control = uniform_control(run, point)
            for name, value in control.items():
                suffix = "" if name == "full" else "_raw"
                row["sqd_uniform" + suffix] = value["value"]
                row["sqd_uniform" + suffix + "_error_meh"] = 1000 * (value["value"] - reference)
                row["sqd_uniform" + suffix + "_dimension"] = value["subspace_dimension"]
        rows.append(row)
    return rows


def circuit_cost(run, calibration):
    """Native CZ count of the first point's circuit, and the calibration fidelity product."""
    from chemflow.backends.calibration import best_layout
    from chemflow.backends.circuits import build_circuit, compile_native
    from chemflow.types import CircuitSpec

    point = run["points"][0]
    parameters = point["bits_qpu"]["provenance"].get("kernel_params")
    electrons, orbitals = run["active"]
    if parameters is None:
        from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals
        from chemflow.impls.pyscf_impls import build_molecule
        from chemflow.runtime.artifacts import ArtifactStore
        from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

        store = ArtifactStore()
        data = build_molecule(GEOMETRY[run["system"]].format(r=point["distance"]),
                              ElectronicModel(basis=run["basis"]), store,
                              active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
        seed = ccsd_seed_from_integrals(data.hamiltonian, data.sector, store)
        parameters = (seed or LucjParams.zeros(orbitals)).to_vector().tolist()
    spec = CircuitSpec(ansatz="lucj", n_orb=orbitals, n_alpha=electrons // 2,
                       n_beta=electrons // 2, layers=[parameters])
    native = compile_native(build_circuit(spec))
    layout = best_layout({"qubits": calibration["qubits"], "properties": {
        "qubit_metrics": calibration["qubit_metrics"],
        "coupler_metrics": calibration["coupler_metrics"]}}, native.num_qubits)
    fidelity = {tuple(sorted(item["qubits"])): item["cz_fidelity"]
                for item in calibration["coupler_metrics"]}
    product = 1.0
    for instruction in native.data:
        if instruction.operation.name == "cz":
            first, second = (native.find_bit(qubit).index for qubit in instruction.qubits)
            product *= fidelity[tuple(sorted((layout[first], layout[second])))]
    return int(native.count_ops().get("cz", 0)), product


def billing(run):
    total, currency, jobs = 0.0, None, 0
    for point in run["points"]:
        bill = point["bits_qpu"]["provenance"].get("billing")
        if bill:
            total += bill["cost"]
            currency = bill["currency"]
            jobs += 1
    return total, currency, jobs


FLOOR_MEH = 1e-3   # |error| below this is drawn at the floor and labelled exact


def figure_curve(run, rows, path):
    import matplotlib.pyplot as plt

    distance = [row["distance"] for row in rows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True,
                                         gridspec_kw={"height_ratios": [1.2, 1]})
    top.plot(distance, [row["rhf"] for row in rows], color=MUTED, linestyle="--",
             label="RHF")
    top.plot(distance, [row["casci"] for row in rows], color=INK, label="CASCI (exact)")
    top.plot(distance, [row["sqd_sim"] for row in rows], color=SIM, marker="s",
             linestyle="none", markersize=3.5, label="SQD, ideal samples")
    top.plot(distance, [row["sqd_qpu"] for row in rows], color=QPU, marker="o",
             linestyle="none", markerfacecolor="none", label="SQD, MQ02 samples")
    top.set_ylabel("Energy / Eh")
    top.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5)
    for key, color, marker, label in (
            ("sqd_qpu", QPU, "o", "MQ02 samples"), ("sqd_sim", SIM, "s", "ideal samples"),
            ("sqd_uniform", UNIFORM, "^", "uniform random bits")):
        bottom.plot(distance, [max(abs(row[key + "_error_meh"]), FLOOR_MEH) for row in rows],
                    color=color, marker=marker, markersize=4 if marker != "^" else 3,
                    markerfacecolor="none" if key != "sqd_sim" else color, label=label)
    bottom.axhline(CHEMICAL_ACCURACY_MEH, color=MUTED, linewidth=0.8, linestyle=":")
    bottom.annotate("chemical accuracy", (distance[-1], CHEMICAL_ACCURACY_MEH),
                    textcoords="offset points", xytext=(0, 3), ha="right", fontsize=6,
                    color=MUTED)
    bottom.annotate("exact (< 10$^{-3}$ mEh)", (distance[-1], FLOOR_MEH),
                    textcoords="offset points", xytext=(0, 4), ha="right", fontsize=6,
                    color=MUTED)
    bottom.set_yscale("log")
    bottom.set_ylim(FLOOR_MEH / 2, None)
    bottom.set_ylabel("|SQD − CASCI| / mEh")
    bottom.set_xlabel("N–N distance / Å")
    figure.legend(*bottom.get_legend_handles_labels(), loc="lower center", ncol=3,
                  fontsize=6.5, bbox_to_anchor=(0.55, -0.04))
    figure.align_ylabels()
    figure.subplots_adjust(bottom=0.16)
    figure.savefig(path)
    plt.close(figure)


def figure_ablation(runs, path):
    import matplotlib.pyplot as plt

    labels, groups = [], []
    for name, rows in runs:
        for row in rows:
            labels.append(f"{name}\n{row['distance']:.2f} Å")
            groups.append(row)
    keys = [("sqd_qpu_raw", QPU, "MQ02, samples only"),
            ("sqd_sim_raw", SIM, "ideal, samples only"),
            ("sqd_uniform_raw", UNIFORM, "uniform, samples only")]
    full = [("sqd_qpu", QPU), ("sqd_sim", SIM), ("sqd_uniform", UNIFORM)]
    figure, axis = plt.subplots(figsize=(7.0, 2.6))
    width = 0.26
    positions = np.arange(len(groups))
    for offset, ((key, color, label), (full_key, _)) in enumerate(zip(keys, full, strict=True)):
        x = positions + (offset - 1) * width
        axis.bar(x, [max(row[key + "_error_meh"], 1e-3) for row in groups], width * 0.9,
                 color=color, alpha=0.35, label=label)
        axis.scatter(x, [max(row[full_key + "_error_meh"], 1e-3) for row in groups], s=12,
                     color=color, zorder=3, edgecolor="white", linewidth=0.6)
    axis.scatter([], [], s=12, color=MUTED, label="dot: with recovery + extension")
    axis.axhline(CHEMICAL_ACCURACY_MEH, color=MUTED, linewidth=0.8, linestyle=":")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, fontsize=5.5)
    axis.set_xlabel("N$_2$ active space (electrons, orbitals) and bond length")
    axis.set_ylabel("SQD − CASCI / mEh")
    axis.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    figure.savefig(path)
    plt.close(figure)


def figure_hardware(runs, path):
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    # One point per run at its first geometry, against the native two-qubit gate count.
    ordered = sorted(runs, key=lambda item: item[2])
    gates = [item[3] for item in ordered]
    fidelity = [np.mean([row["hellinger"] for row in item[1]]) for item in ordered]
    left.plot(gates, fidelity, color=QPU, marker="o", label="measured (mean over geometries)")
    left.plot(gates, [item[4] for item in ordered], color=MUTED, linestyle="--", marker="s",
              label="calibration product of CZ fidelities")
    for gate, value, item in zip(gates, fidelity, ordered, strict=True):
        left.annotate(f"{item[2]}q", (gate, value), textcoords="offset points", xytext=(4, 4),
                      fontsize=6, color=MUTED)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("native CZ gates in the LUCJ circuit")
    left.set_ylabel("Hellinger fidelity to ideal")
    left.legend(loc="lower left")
    qubits = [item[2] for item in ordered]
    valid = [np.mean([row["valid_fraction"] for row in item[1]]) for item in ordered]
    uniform = [item[1][0]["sector_dimension"] / 2 ** item[2] for item in ordered]
    right.plot(qubits, valid, color=QPU, marker="o", label="MQ02")
    right.plot(qubits, uniform, color=UNIFORM, marker="^", linestyle="--",
               label="uniform random bitstrings")
    right.axhline(1.0, color=SIM, linewidth=1.0, label="ideal (all shots in sector)")
    right.set_xlabel("qubits")
    right.set_ylabel("fraction of shots in the (Nα, Nβ) sector")
    right.set_yscale("log")
    right.legend(loc="upper right", fontsize=6.5)
    figure.savefig(path)
    plt.close(figure)


def figure_distribution(rows, path, run):
    import matplotlib.pyplot as plt

    point = next(p for p in run["points"] if abs(p["distance"] - 0.74) < 1e-6)
    hardware, ideal = point["bits_qpu"]["counts"], point["bits_sim"]["counts"]
    keys = sorted({*hardware, *ideal}, key=lambda key: -ideal.get(key, 0) - hardware.get(key, 0))
    width = 2 * run["active"][1]
    figure, axis = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(keys))
    for offset, (counts, color, label) in enumerate(((ideal, SIM, "ideal simulator"),
                                                    (hardware, QPU, "MQ02"))):
        total = sum(counts.values())
        axis.bar(x + (offset - 0.5) * 0.4, [counts.get(key, 0) / total for key in keys], 0.38,
                 color=color, label=label)
    axis.set_xticks(x, [format(int(key), f"0{width}b")[::-1] for key in keys], rotation=90,
                    fontsize=6, family="monospace")
    axis.set_ylabel("probability")
    axis.set_yscale("log")
    axis.legend()
    figure.savefig(path)
    plt.close(figure)


def figure_timeline(run, path):
    import matplotlib.pyplot as plt

    profile = [entry for entry in run["profile"] if "start_s" in entry]
    lanes = sorted({entry["device"] for entry in profile},
                   key=lambda lane: ("cpu" not in lane, "gpu" not in lane, lane))
    figure, axis = plt.subplots(figsize=(7.0, 0.32 * len(lanes) + 0.9))
    colors = {"cpu": INK, "gpu": SIM, "mq02": QPU, "sim": UNIFORM}
    for row, lane in enumerate(lanes):
        color = next(value for key, value in colors.items() if lane.startswith(key))
        spans = [(entry["start_s"], entry["end_s"] - entry["start_s"])
                 for entry in profile if entry["device"] == lane]
        axis.broken_barh(spans, (row - 0.35, 0.7), facecolors=color, edgecolor="white",
                         linewidth=0.4)
    busy = sum(entry["end_s"] - entry["start_s"] for entry in profile)
    span = max(entry["end_s"] for entry in profile) - min(entry["start_s"] for entry in profile)
    axis.set_yticks(range(len(lanes)), lanes)
    axis.set_xlabel(f"worker time / s   (sum of node times {busy:.1f} s in {span:.1f} s)")
    axis.grid(axis="y", visible=False)
    figure.savefig(path)
    plt.close(figure)
    return busy, span


def latex_table(rows, path, caption_label):
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             (r"$R$/\AA & $E_\mathrm{CASCI}$/Eh & \multicolumn{3}{c}{SQD $-$ CASCI / mEh} & "
              r"$F_\mathrm{H}$ & valid & dim./sector \\"),
             r"\cmidrule(lr){3-5}", r" & & MQ02 & ideal & uniform & & & \\", r"\midrule"]
    def meh(value):
        return f"{0.0 if abs(value) < 0.005 else value:.2f}"

    for row in rows:
        lines.append(
            f"{row['distance']:.3f} & {row['casci']:.6f} & {meh(row['sqd_qpu_error_meh'])} & "
            f"{meh(row['sqd_sim_error_meh'])} & {meh(row['sqd_uniform_error_meh'])} & "
            f"{row['hellinger']:.3f} & {row['valid_fraction']:.3f} & "
            f"{row['sqd_qpu_dimension']}/{row['sector_dimension']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="docs/paper")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--calibration", default="runs/qpu/mq02_calibration.json")
    args = parser.parse_args()
    _style()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summaries, hardware_runs, macros = {}, [], []
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    previous_path = out / "results.json"
    previous_results = (json.loads(previous_path.read_text(encoding="utf-8"))
                        if previous_path.is_file() else {})
    total_cost, currency, total_jobs = 0.0, None, 0
    for filename in args.runs:
        run = json.loads(Path(filename).read_text(encoding="utf-8"))
        electrons, orbitals = run["active"]
        tag = f"{run['system']}_{electrons}{orbitals}"
        cached = previous_results.get(tag, {}).get("rows")
        rows = summarize(run, controls=not args.no_controls, cached=cached)
        summaries[tag] = {"system": run["system"], "basis": run["basis"],
                          "active": run["active"], "shots": run["shots"],
                          "wall_s": run["wall_s"], "cpu_core_s": run["cpu_core_s"],
                          "plan_hash": run["plan_hash"], "rows": rows}
        cost, run_currency, jobs = billing(run)
        total_cost += cost
        total_jobs += jobs
        currency = run_currency or currency
        label = f"{run['system'].upper()} CAS({electrons},{orbitals})"
        gates, predicted = circuit_cost(run, calibration)
        summaries[tag]["cz_gates"] = gates
        summaries[tag]["predicted_fidelity"] = predicted
        hardware_runs.append((label, rows, 2 * orbitals, gates, predicted))
        if run["system"] == "n2":
            figure_curve(run, rows, out / "figures" / f"curve_{tag}.pdf")
            latex_table(rows, out / "tables" / f"{tag}.tex", label)
            busy, span = figure_timeline(run, out / "figures" / f"timeline_{tag}.pdf")
            summaries[tag]["timeline"] = {"busy_s": busy, "span_s": span}
        else:
            figure_distribution(rows, out / "figures" / f"distribution_{tag}.pdf", run)
            latex_table(rows, out / "tables" / f"{tag}.tex", label)
    figure_hardware(hardware_runs, out / "figures" / "hardware_fidelity.pdf")
    figure_ablation([(label.split()[1][3:].strip("()"), rows)
                     for label, rows, *_ in hardware_runs if "N2" in label],
                    out / "figures" / "ablation.pdf")
    macros.append(f"\\newcommand{{\\qpuCost}}{{{total_cost:.1f}}}")
    macros.append(f"\\newcommand{{\\qpuCurrency}}{{{currency}}}")
    macros.append(f"\\newcommand{{\\qpuJobs}}{{{total_jobs}}}")
    (out / "tables" / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(summaries, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    for tag, summary in summaries.items():
        print(tag, f"wall {summary['wall_s']:.1f}s")
        for row in summary["rows"]:
            print(f"  {row['distance']:.3f} qpu {row['sqd_qpu_error_meh']:8.3f} "
                  f"sim {row['sqd_sim_error_meh']:8.3f} "
                  f"uni {row.get('sqd_uniform_error_meh', float('nan')):8.3f} | raw qpu "
                  f"{row['sqd_qpu_raw_error_meh']:8.2f} sim {row['sqd_sim_raw_error_meh']:8.2f} "
                  f"uni {row.get('sqd_uniform_raw_error_meh', float('nan')):8.2f} | "
                  f"F {row['hellinger']:.3f} valid {row['valid_fraction']:.3f} "
                  f"dim {row['sqd_qpu_dimension']}/{row['sector_dimension']}")
    print(f"QPU: {total_jobs} jobs, {total_cost:.2f} {currency}")


if __name__ == "__main__":
    main()
