"""Jastrow-off twin circuits on MQ02: does the hardware carry the LUCJ correlation signal?

The LUCJ circuit is U^dagger J U |HF>. Its twin, with the same gates and the Jastrow
phases set to zero, is U^dagger U |HF> = |HF>: an exact, same-structure calibration of the
noise of this gate sequence (the different decompositions of U and U^dagger keep the
transpiler from cancelling it). The CCSD-seeded LUCJ output is 97-99.97 % Hartree-Fock,
so the test is on the rest: the off-HF part of the hardware distribution, raw and with
the twin's subtracted, against the ideal off-HF distribution. The Jastrow is
line-native (only the opposite-spin pair adjacent on the chain), which saves a third of
the CZ at 12 qubits for a 0.6 % change of the state.

    . ~/.config/chemflow/lqcloud.env
    python qcl/apps/twin_mitigation.py --da da/result.json --json twin.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from chemflow.backends.lqcloud import LqcloudBackend
from chemflow.impls.fci import sector_states
from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals, lucj_state
from chemflow.impls.pyscf_impls import build_molecule
from chemflow.impls.sector import hf_vector
from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import CircuitSpec, StatePrep
from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

SYSTEMS = {"butadiene": (4, 4), "transition": (6, 6)}


def hellinger(first: dict, second: dict) -> float:
    total_first, total_second = sum(first.values()), sum(second.values())
    if not total_first or not total_second:
        return 0.0
    return sum(math.sqrt(max(value, 0) / total_first * max(second.get(key, 0), 0) / total_second)
               for key, value in first.items()) ** 2


def in_sector(counts, n_orb, n_alpha):
    mask = (1 << n_orb) - 1
    return {key: value for key, value in counts.items()
            if (key & mask).bit_count() == n_alpha and (key >> n_orb).bit_count() == n_alpha}


def analyse(counts, twin, ideal, hf, n_orb, n_alpha):
    """Fidelities of the full and the off-HF distributions, raw and twin-corrected."""
    total, twin_total = sum(counts.values()), sum(twin.values())
    p = {key: value / total for key, value in counts.items()}
    t = {key: value / twin_total for key, value in twin.items()}
    weight = t.get(hf, 0.0)                      # twin: ideal is |HF>, so P(HF) ~ its fidelity
    corrected = {key: (p.get(key, 0.0) - t.get(key, 0.0)) / weight
                 for key in set(p) | set(t)}
    corrected[hf] = corrected.get(hf, 0.0) + 1.0
    corrected = {key: value for key, value in corrected.items() if value > 0}
    off = {key: value for key, value in ideal.items() if key != hf}
    raw_valid = in_sector(p, n_orb, n_alpha)
    corrected_valid = in_sector(corrected, n_orb, n_alpha)
    return {
        "twin_hf_weight": weight, "hf_weight": p.get(hf, 0.0), "ideal_hf_weight": ideal[hf],
        "F": hellinger(p, ideal), "F_post": hellinger(raw_valid, ideal),
        "F_twin": hellinger(corrected, ideal), "F_twin_post": hellinger(corrected_valid, ideal),
        "F_off": hellinger({k: v for k, v in raw_valid.items() if k != hf}, off),
        "F_off_twin": hellinger({k: v for k, v in corrected_valid.items() if k != hf}, off),
        "F_off_uniform": hellinger({k: 1.0 for k in off}, off),
        "F_off_twin_only": hellinger({k: v for k, v in in_sector(t, n_orb, n_alpha).items()
                                      if k != hf}, off),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--da", required=True)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    structures = json.loads(Path(args.da).read_text())["structures"]
    backend = LqcloudBackend(backend="MQ02")
    report = {}
    for name, (electrons, orbitals) in SYSTEMS.items():
        store = ArtifactStore()
        data = build_molecule(structures[name], ElectronicModel(basis="6-31g*"), store,
                              active=ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals))
        seed = ccsd_seed_from_integrals(data.hamiltonian, data.sector, store)
        line = LucjParams(norb=orbitals, kparams=seed.kparams, j_same=seed.j_same,
                          j_opp=np.array([seed.j_opp[0]] + [0.0] * (len(seed.j_opp) - 1)))
        twin = LucjParams(norb=orbitals, kparams=seed.kparams, j_same=0 * seed.j_same,
                          j_opp=0 * seed.j_opp)
        states = sector_states(data.sector)
        state = lucj_state(data.sector, [line])
        ideal = {int(states[index]): float(abs(amplitude) ** 2)
                 for index, amplitude in enumerate(state) if abs(amplitude) ** 2 > 1e-14}
        hf = int(states[int(np.argmax(hf_vector(data.sector)))])
        runs = {}
        for label, layer in (("lucj", line), ("twin", twin)):
            spec = CircuitSpec(ansatz="lucj", n_orb=orbitals, n_alpha=electrons // 2,
                               n_beta=electrons // 2, layers=(tuple(layer.to_vector()),))
            prep = StatePrep(state_ref="", encoding_id="sector_state",
                             hamiltonian_id=data.hamiltonian.id, sector=data.sector,
                             recipe=f"twin-experiment:{label}", circuit=spec)
            parts = []
            for repetition in range(args.repeats):
                started = time.perf_counter()
                result = backend.sample(store, prep, args.shots, repetition=repetition)
                parts.append({"counts": {int(k): v for k, v in result.value.counts.items()},
                              "job_ids": list(result.job_ids), "wall_s": time.perf_counter() - started,
                              "diagnostics": {k: v for k, v in result.diagnostics.items()
                                              if k in ("layout", "native_ops", "depth", "billing")}})
            runs[label] = parts
        rows = [analyse(runs["lucj"][index]["counts"], runs["twin"][index]["counts"], ideal, hf,
                        orbitals, electrons // 2) for index in range(args.repeats)]
        report[name] = {"qubits": 2 * orbitals, "ideal_hf_weight": ideal[hf], "rows": rows,
                        "cz": {label: runs[label][0]["diagnostics"].get("native_ops", {}).get("cz")
                               for label in runs},
                        "jobs": {label: [part["job_ids"] for part in parts]
                                 for label, parts in runs.items()},
                        "counts": {label: [{str(k): v for k, v in part["counts"].items()}
                                           for part in parts] for label, parts in runs.items()}}
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        print(name, report[name]["cz"], json.dumps({k: round(v, 4) for k, v in mean.items()}),
              flush=True)
    Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
