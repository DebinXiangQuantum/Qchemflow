"""Interacting spin-chain dynamics on MQ02, mitigated by training on free-fermion twins.

The open XXZ chain H = sum_i J (X_i X_i+1 + Y_i Y_i+1) + Delta Z_i Z_i+1 is native to
MQ02's linear chain: one Trotter step is a brick wall of nearest-neighbour bonds, three
CZ each, no routing. From the Neel state the circuit for Delta != 0 is interacting
dynamics (not a fermionic Gaussian circuit). With Delta = 0 the same gates give the XX
model, free fermions after Jordan-Wigner, whose observables are classically computable at
any size.

Every bond uses the same three-CNOT block for any couplings and the circuits are compiled
without two-qubit resynthesis (optimization level 1). The free-fermion twins therefore
run the gates of the interacting circuit and see its noise. Following Montanaro &
Stanisic (arXiv:2102.02120, "Error mitigation by training with fermionic linear optics"),
a linear correction per observable class is fitted on the twins (J = 0.6, 1.0, 1.4,
Delta = 0) and applied to the interacting circuit (J = 1, Delta = 1).

Observables come straight from the Z-basis samples: the site magnetizations <Z_i>, the
neighbour correlations <Z_i Z_i+1> and the staggered magnetization. Total S^z is
conserved, so shots outside the initial magnetization sector can be discarded.

    . ~/.config/chemflow/lqcloud.env
    python qcl/apps/xxz_dynamics.py --sizes 12 24 --json xxz.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from chemflow.backends.circuits import build_circuit, compile_spec
from chemflow.backends.lqcloud import LqcloudBackend
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import CircuitSpec, Sector, StatePrep

STEP = 0.25
TRAINING_COUPLINGS = (0.6, 1.0, 1.4)
DEPTHS = {8: (1, 2, 3, 4, 6), 12: (1, 2, 3, 4), 16: (1, 2, 3), 24: (1, 2, 3)}


def spec(n_qubits, coupling, anisotropy, steps):
    return CircuitSpec(ansatz="xxz_trotter", n_orb=n_qubits, n_alpha=n_qubits // 2, n_beta=0,
                       n_qubits=n_qubits, layers=((coupling, anisotropy, STEP, steps),),
                       optimization_level=1)


def observables(probabilities: np.ndarray | None, counts: dict | None, n_qubits: int,
                post_select: bool) -> dict:
    """<Z_i>, <Z_i Z_i+1> and the staggered magnetization from probabilities or counts."""
    if probabilities is not None:
        return exact_observables(probabilities, n_qubits)
    else:
        index = np.array([int(key) for key in counts], dtype=np.int64)
        weights = np.array([counts[key] for key in counts], dtype=float)
    bits = ((index[:, None] >> np.arange(n_qubits)) & 1).astype(np.int8)
    if post_select:
        keep = bits.sum(axis=1) == n_qubits // 2
        bits, weights = bits[keep], weights[keep]
    weights = weights / weights.sum()
    spins = 1 - 2 * bits.astype(float)
    z = weights @ spins
    zz = weights @ (spins[:, :-1] * spins[:, 1:])
    sign = np.array([(-1) ** (site + 1) for site in range(n_qubits)])
    return {"z": z, "zz": zz, "staggered": float(np.mean(sign * z))}


def exact_observables(probabilities: np.ndarray, n_qubits: int) -> dict:
    """The same observables from a full probability vector, through marginals."""
    tensor = probabilities.reshape((2,) * n_qubits)        # axis n-1-i is qubit i

    def axis(qubit):
        return n_qubits - 1 - qubit

    z = np.empty(n_qubits)
    for qubit in range(n_qubits):
        marginal = tensor.sum(axis=tuple(a for a in range(n_qubits) if a != axis(qubit)))
        z[qubit] = marginal[0] - marginal[1]
    zz = np.empty(n_qubits - 1)
    for qubit in range(n_qubits - 1):
        keep = sorted((axis(qubit), axis(qubit + 1)))
        marginal = tensor.sum(axis=tuple(a for a in range(n_qubits) if a not in keep))
        zz[qubit] = marginal[0, 0] + marginal[1, 1] - marginal[0, 1] - marginal[1, 0]
    sign = np.array([(-1) ** (site + 1) for site in range(n_qubits)])
    return {"z": z, "zz": zz, "staggered": float(np.mean(sign * z))}


def fit(training: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Scale a minimizing sum |a y_noisy - y_exact|^2 over the training observables."""
    noisy = np.concatenate([item[0] for item in training])
    exact = np.concatenate([item[1] for item in training])
    return float(noisy @ exact / (noisy @ noisy))


def _simulate_native(item, shots):
    """Noiseless samples of the compiled (and twirled) native circuit, for dry runs."""
    from qiskit_aer import AerSimulator

    native = compile_spec(item)
    counts = AerSimulator().run(native, shots=shots, seed_simulator=1).result().get_counts()
    return {str(int(bits.replace(" ", ""), 2)): value for bits, value in counts.items()}


def rms(values, reference):
    return float(np.sqrt(np.mean((np.asarray(values) - np.asarray(reference)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=(12, 24))
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--json", required=True)
    parser.add_argument("--twirls", type=int, default=1,
                        help="Pauli-twirled instances per circuit (shots split evenly)")
    parser.add_argument("--dry-run", action="store_true",
                        help="noiseless simulation of the compiled native circuits")
    args = parser.parse_args()
    backend = LqcloudBackend(backend="MQ02")
    store = ArtifactStore()
    report = {"step": STEP, "training_couplings": TRAINING_COUPLINGS, "rows": []}
    for n_qubits in args.sizes:
        for steps in DEPTHS[n_qubits]:
            circuits = {"target": spec(n_qubits, 1.0, 1.0, steps),
                        **{f"twin_J{c}": spec(n_qubits, c, 0.0, steps)
                           for c in TRAINING_COUPLINGS}}
            cz = {name: compile_spec(item).count_ops().get("cz", 0)
                  for name, item in circuits.items()}
            if len(set(cz.values())) != 1:
                raise SystemExit(f"Twins and target compile differently: {cz}")
            ideal = {name: Statevector.from_instruction(build_circuit(item)).probabilities()
                     for name, item in circuits.items()}
            row = {"qubits": n_qubits, "steps": steps, "cz": cz["target"], "jobs": {}}
            measured = {}
            for name, item in circuits.items():
                instances = ([item] if args.twirls <= 1 else
                             [item.model_copy(update={"twirl_seed": seed})
                              for seed in range(args.twirls)])
                per_instance = args.shots // len(instances)
                merged, jobs = {}, []
                for instance in instances:
                    if args.dry_run:
                        counts = _simulate_native(instance, per_instance)
                    else:
                        prep = StatePrep(state_ref="", encoding_id="spin_chain",
                                         hamiltonian_id="xxz",
                                         sector=Sector(n_alpha=n_qubits // 2, n_beta=0,
                                                       n_orb=n_qubits),
                                         recipe=f"xxz:{name}", circuit=instance)
                        started = time.perf_counter()
                        result = backend.sample(store, prep, per_instance)
                        counts = dict(result.value.counts)
                        jobs.append({"job_ids": list(result.job_ids),
                                     "wall_s": time.perf_counter() - started,
                                     "billing": result.diagnostics.get("billing"),
                                     "native_ops": result.diagnostics.get("native_ops")})
                    for key, value in counts.items():
                        merged[str(key)] = merged.get(str(key), 0) + value
                measured[name] = merged
                row["jobs"][name] = jobs
            for post in (False, True):
                label = "post" if post else "raw"
                exact = {name: observables(ideal[name], None, n_qubits, False)
                         for name in circuits}
                noisy = {name: observables(None, measured[name], n_qubits, post)
                         for name in circuits}
                scale = {key: fit([(noisy[name][key], exact[name][key])
                                   for name in circuits if name != "target"])
                         for key in ("z", "zz")}
                target, reference = noisy["target"], exact["target"]
                row[label] = {
                    "scale": scale,
                    "z_rms": rms(target["z"], reference["z"]),
                    "z_rms_trained": rms(scale["z"] * target["z"], reference["z"]),
                    "zz_rms": rms(target["zz"], reference["zz"]),
                    "zz_rms_trained": rms(scale["zz"] * target["zz"], reference["zz"]),
                    "staggered": target["staggered"],
                    "staggered_trained": scale["z"] * target["staggered"],
                    "staggered_exact": reference["staggered"],
                    "z_exact": reference["z"].tolist(), "z_noisy": target["z"].tolist(),
                    "zz_exact": reference["zz"].tolist(), "zz_noisy": target["zz"].tolist(),
                }
            row["counts"] = measured if not args.dry_run else {}
            report["rows"].append(row)
            print(f"N={n_qubits:2d} steps={steps} CZ={row['cz']:3d} | " + " | ".join(
                f"{label}: Z rms {row[label]['z_rms']:.3f} -> {row[label]['z_rms_trained']:.3f}, "
                f"ZZ rms {row[label]['zz_rms']:.3f} -> {row[label]['zz_rms_trained']:.3f}, "
                f"m_s {row[label]['staggered']:+.3f} -> {row[label]['staggered_trained']:+.3f} "
                f"(exact {row[label]['staggered_exact']:+.3f})" for label in ("raw", "post")),
                flush=True)
            Path(args.json).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
