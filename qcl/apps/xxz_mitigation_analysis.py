"""Mitigation estimators for xxz_dynamics.py runs: raw, depolarizing rescale and affine.

The free-fermion twins (Delta = 0) have exactly computable observables; from them it fits
a global depolarizing fidelity f (noisy - mixed = f (exact - mixed), mixed: the
maximally mixed S^z = 0 state) and an affine map per observable class, and applies both
to the interacting circuit.

    python qcl/apps/xxz_mitigation_analysis.py xxz.json
"""

import json
import sys

import numpy as np
from qiskit.quantum_info import Statevector
from xxz_dynamics import exact_observables, observables, spec

from chemflow.backends.circuits import build_circuit

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    d = json.load(handle)
summary = []
for row in d["rows"]:
    N, k = row["qubits"], row["steps"]
    exact, noisy = {}, {}
    for name, cnts in row["counts"].items():
        J, D = (1.0, 1.0) if name == "target" else (float(name.split("J")[1]), 0.0)
        exact[name] = exact_observables(Statevector.from_instruction(build_circuit(spec(N, J, D, k))).probabilities(), N)
        noisy[name] = observables(None, cnts, N, True)
    mixed = {"z": 0.0, "zz": -1.0 / (N - 1)}           # maximally mixed state in the S^z = 0 sector
    twins = [n for n in exact if n != "target"]
    vec = lambda o, names: np.concatenate([o[n]["z"] for n in names] + [o[n]["zz"] for n in names])
    ref_t = np.concatenate([exact["target"]["z"], exact["target"]["zz"]])
    raw_t = np.concatenate([noisy["target"]["z"], noisy["target"]["zz"]])
    out = {"raw": raw_t}
    # global depolarizing fidelity f from twins: noisy - mixed = f (exact - mixed)
    for label, names in (("f_all_twins", twins), ("f_same_J", ["twin_J1.0"])):
        mix = np.concatenate([np.full(N * len(names), 0.0), np.full((N - 1) * len(names), mixed["zz"])])
        e, n = vec(exact, names) - mix, vec(noisy, names) - mix
        f = float(e @ n / (e @ e))
        mix_t = np.concatenate([np.zeros(N), np.full(N - 1, mixed["zz"])])
        out[label] = mix_t + (raw_t - mix_t) / f
        out[label + "_value"] = f
    # affine per class from all twins
    aff = []
    for key, size in (("z", N), ("zz", N - 1)):
        x = np.concatenate([noisy[n][key] for n in twins]); y = np.concatenate([exact[n][key] for n in twins])
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        aff.append(a * noisy["target"][key] + b)
    out["affine"] = np.concatenate(aff)
    sign = np.array([(-1) ** (s + 1) for s in range(N)])
    line = {"N": N, "steps": k, "cz": row["cz"], "m_s_exact": float(np.mean(sign * exact["target"]["z"]))}
    for label in ("raw", "f_all_twins", "f_same_J", "affine"):
        est = out[label]
        line[label] = {"rms_z": float(np.sqrt(np.mean((est[:N] - ref_t[:N]) ** 2))),
                       "rms_zz": float(np.sqrt(np.mean((est[N:] - ref_t[N:]) ** 2))),
                       "m_s": float(np.mean(sign * est[:N]))}
    line["f_same_J"]["f"] = out["f_same_J_value"]; line["f_all_twins"]["f"] = out["f_all_twins_value"]
    summary.append(line)
    parts = []
    for label in ("raw", "f_same_J", "f_all_twins", "affine"):
        item = line[label]
        fidelity = f" f={item['f']:.2f}" if "f" in item else ""
        parts.append(f"{label}: Z {item['rms_z']:.3f} ZZ {item['rms_zz']:.3f} "
                     f"m_s {item['m_s']:+.3f}{fidelity}")
    print(f"N={N} steps={k} CZ={row['cz']} m_s exact {line['m_s_exact']:+.3f} | " +
          " | ".join(parts))
with open(path.replace(".json", "_estimators.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=1)
