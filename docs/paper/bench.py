"""Scheduling benchmark table and timelines from runs/qpu/bench2 (see bench2/run.sh).

    python docs/paper/bench.py runs/qpu/bench2 --out docs/paper
"""

import argparse
import json
from pathlib import Path

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"cpu": "#2a78d6", "gpu": "#eb6834", "mq02": "#1baf7a", "sim": "#eda100"}
MODES = (("sequential_auto", "sequential"), ("lanes_off", "lanes (threads)"),
         ("compiled_off", "compiled (threads)"), ("compiled_on", "compiled (lane processes)"))
CASES = {"1010": "N$_2$ CAS(10,10), 4 geometries", "1212": "N$_2$ CAS(12,12), 2 geometries"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def busy(run):
    return sum(entry["duration_s"] for entry in run["profile"])


def timeline(axis, run, title):
    profile = [entry for entry in run["profile"] if "start_s" in entry]
    if not profile:  # sequential runs have no start times: lay the nodes end to end
        clock, profile = 0.0, []
        for entry in run["profile"]:
            profile.append({**entry, "start_s": clock, "end_s": clock + entry["duration_s"]})
            clock += entry["duration_s"]
    lanes = sorted({entry["device"] for entry in profile},
                   key=lambda lane: (not lane.startswith("cpu"), not lane.startswith("gpu"), lane))
    for row, lane in enumerate(lanes):
        color = next(value for key, value in COLORS.items() if lane.startswith(key))
        spans = [(entry["start_s"], entry["end_s"] - entry["start_s"])
                 for entry in profile if entry["device"] == lane]
        axis.broken_barh(spans, (row - 0.36, 0.72), facecolors=color, edgecolor="white",
                         linewidth=0.3)
    axis.set_yticks(range(len(lanes)), lanes)
    axis.set_title(title, loc="left")
    axis.grid(axis="y", visible=False)


def main():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bench")
    parser.add_argument("--out", default="docs/paper")
    args = parser.parse_args()
    mpl.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "axes.edgecolor": MUTED})
    bench, out = Path(args.bench), Path(args.out)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             (r"Workload & execution & wall/s & node time/s & predicted/s & speed-up & "
              r"max $|\Delta E|$/Eh \\"), r"\midrule"]
    macros, summary = [], {}
    for tag, label in CASES.items():
        runs = {name: load(bench / f"{name}_{tag}.json") for name, _ in MODES}
        reference = runs["sequential_auto"]
        if reference is None:
            continue
        for index, (name, title) in enumerate(MODES):
            run = runs[name]
            if run is None:
                continue
            deviation = max(abs(point[key]["value"] - base[key]["value"])
                            for point, base in zip(run["points"], reference["points"],
                                                   strict=True)
                            for key in ("rhf", "casci"))
            predicted = (run["schedule"]["predicted"]["makespan_s"]
                         if run.get("schedule") else None)
            speedup = reference["wall_s"] / run["wall_s"]
            summary[f"{name}_{tag}"] = {"wall_s": run["wall_s"], "busy_s": busy(run),
                                        "predicted_s": predicted, "speedup": speedup,
                                        "max_deviation": deviation,
                                        "slots": (run.get("schedule") or {}).get("slots")}
            lines.append(f"{label if index == 0 else ''} & {title} & {run['wall_s']:.1f} & "
                         f"{busy(run):.1f} & "
                         f"{'' if predicted is None else f'{predicted:.1f}'} & "
                         f"{speedup:.2f}$\\times$ & {deviation:.1e} \\\\")
            macros.append(f"\\newcommand{{\\bench{name.replace('_', '').capitalize()}"
                          f"{'A' if tag == '1010' else 'B'}}}{{{speedup:.2f}}}")
        lines.append(r"\midrule")
    fresh = {mode: load(bench / f"mq02_{mode}_1010.json") for mode in ("sequential", "compiled")}
    if all(fresh.values()):
        base = fresh["sequential"]
        for mode, run in fresh.items():
            predicted = (run["schedule"]["predicted"]["makespan_s"]
                         if run.get("schedule") else None)
            speedup = base["wall_s"] / run["wall_s"]
            summary[f"mq02_{mode}_1010"] = {"wall_s": run["wall_s"], "busy_s": busy(run),
                                            "predicted_s": predicted, "speedup": speedup}
            title = "sequential" if mode == "sequential" else "compiled (lane processes)"
            lines.append(f"{'MQ02, CAS(10,10), 4 geometries, new jobs' if mode == 'sequential' else ''}"
                         f" & {title} & {run['wall_s']:.1f} & {busy(run):.1f} & "
                         f"{'' if predicted is None else f'{predicted:.1f}'} & "
                         f"{speedup:.2f}$\\times$ & -- \\\\")
            macros.append(f"\\newcommand{{\\benchMq{mode.capitalize()}}}{{{speedup:.2f}}}")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "bench.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "tables" / "bench_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (out / "bench.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    sequential = load(bench / "sequential_auto_1212.json")
    compiled = load(bench / "compiled_on_1212.json")
    if sequential and compiled:
        figure, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
        timeline(axes[0], sequential, f"sequential: {sequential['wall_s']:.0f} s")
        timeline(axes[1], compiled, f"compiled schedule: {compiled['wall_s']:.0f} s")
        axes[1].set_xlabel("time in the worker / s")
        (out / "figures").mkdir(parents=True, exist_ok=True)
        figure.savefig(out / "figures" / "bench_timeline.pdf")
        plt.close(figure)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
