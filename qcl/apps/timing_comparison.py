"""Exact solution on our server against the quantum route, for every hardware active space.

Measured on this server:
- exact FCI with PySCF on the CPU (``--threads`` threads) and with ChemFlow's CUDA FCI
  on one GPU (12 and more orbitals);
- the quantum route's classical steps: the CCSD seed, LUCJ circuit construction and
  transpilation, and SQD on the hardware samples (the same options as the plan).

Measured on the QPU, from the job ledger: device execution time and submit-to-result
wall time of every job. The table also projects the QPU time for the shot counts the
simulator runs used, with the MQ02 job-time model (0.709 s + 0.3985 ms per shot,
at most 50,000 shots per job, from 3,773 jobs).

    python qcl/apps/timing_comparison.py --out hw
"""

import argparse
import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np
from hardware_analysis import _ledger, _rebuild, job_timing

SIMULATOR_SHOTS = {"diels_alder": 200000, "t4l": 100000, "hchain": 100000, "large": 200000}


def mq02_job_seconds(shots: int) -> tuple[int, float]:
    """Jobs and device seconds for ``shots`` with the MQ02 model."""
    jobs = -(-shots // 50000)
    per_job = shots / jobs
    return jobs, jobs * (0.709 + 0.0003985 * per_job)


def _time(function, repeat=1):
    best = None
    for _ in range(repeat):
        started = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    return value, best


def exact_times(data, store, n_orb):
    from chemflow.impls.pyscf_impls import fci_pyscf

    times = {}
    (energy, *_), times["fci_cpu_s"] = _time(lambda: fci_pyscf(data, store))
    times["fci_cpu_energy"] = float(energy)
    if n_orb >= 12:
        import cupy as cp

        from chemflow.backends.gpu import gpu_fci_ground

        gpu_fci_ground(store, data, data.sector.model_dump())  # warm the kernels
        cp.cuda.Device().synchronize()
        _, times["fci_gpu_s"] = _time(
            lambda: gpu_fci_ground(store, data, data.sector.model_dump()))
    return times


def quantum_classical_times(data, store, counts, shots, n_orb, large):
    from chemflow.backends.circuits import build_circuit, compile_native
    from chemflow.impls.lucj import LucjParams, ccsd_seed_from_integrals
    from chemflow.impls.sqd import sqd_sector
    from chemflow.types import CircuitSpec, Samples

    seed, ccsd_s = _time(lambda: ccsd_seed_from_integrals(data.hamiltonian, data.sector, store))
    params = (seed or LucjParams.zeros(n_orb)).to_vector().tolist()
    spec = CircuitSpec(ansatz="lucj", n_orb=n_orb, n_alpha=data.sector.n_alpha,
                       n_beta=data.sector.n_beta, layers=(tuple(params),))
    native, compile_s = _time(lambda: compile_native(build_circuit(spec)))
    samples = Samples(counts=counts, shots=shots, encoding_id="sector_state",
                      basis="computational", source="qpu", seed=0)
    options = {"n_batches": 4, "max_iter": 2} if large else {}
    times = {"ccsd_seed_s": ccsd_s, "compile_s": compile_s,
             "cz": int(native.count_ops().get("cz", 0))}
    (_, diagnostics), times["sqd_raw_s"] = _time(lambda: sqd_sector(
        data, data.sector, store, samples, recovery=False, with_ext=False))
    times["sqd_raw_dims"] = diagnostics["subspace_dimension"]
    if n_orb < 14:
        (_, diagnostics), times["sqd_full_s"] = _time(lambda: sqd_sector(
            data, data.sector, store, samples, **options))
        times["sqd_full_dims"] = diagnostics["subspace_dimension"]
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", default=str(Path.home() / ".chemflow" / "qpu"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    from chemflow.runtime.artifacts import ArtifactStore

    out = Path(args.out)
    ledger = _ledger(Path(args.ledger))
    metadata = json.loads((out / "metadata.json").read_text())
    rows = []
    for group, systems in metadata.items():
        payload = json.loads((out / f"{group}.json").read_text())
        for name, spec in systems.items():
            electrons, n_orb = spec["active"]
            store = ArtifactStore()
            data = _rebuild(group, name, spec, store)
            bits = payload["rows"][name]["base_bits"]
            jobs = [job_timing(ledger[job]) for job in bits["job_ids"] if job in ledger]
            row = {"group": group, "system": name, "qubits": 2 * n_orb,
                   "sector": comb(n_orb, electrons // 2) ** 2, "qpu_jobs": jobs,
                   "qpu_shots": bits["shots"],
                   "qpu_device_s": sum(job["device_s"] or 0 for job in jobs),
                   "qpu_wall_s": sum(job["wall_s"] or 0 for job in jobs),
                   **exact_times(data, store, n_orb),
                   **quantum_classical_times(data, store, bits["counts"], bits["shots"],
                                             n_orb, n_orb >= 12)}
            projected = SIMULATOR_SHOTS[group]
            row["projected_shots"] = projected
            row["projected_jobs"], row["projected_device_s"] = mq02_job_seconds(projected)
            per_job_overhead = (np.median([job["wall_s"] - job["device_s"] for job in jobs
                                           if job["wall_s"] and job["device_s"]])
                                if jobs else 0.0)
            row["projected_wall_s"] = (row["projected_device_s"] +
                                       row["projected_jobs"] * float(per_job_overhead))
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "qpu_jobs"},
                             default=float), flush=True)
    (out / "timing.json").write_text(json.dumps(rows, indent=1, default=float))
    _print(rows)


def _print(rows):
    print(f"\n{'system':24s} {'q':>3} {'FCI cpu':>8} {'FCI gpu':>8} | {'QPU dev':>8} "
          f"{'QPU wall':>8} {'SQD':>8} {'seed+tr':>8} {'route':>8} | {'proj. wall':>10}")
    for row in rows:
        sqd = row.get("sqd_full_s", row["sqd_raw_s"])
        route = row["ccsd_seed_s"] + row["compile_s"] + row["qpu_wall_s"] + sqd
        print(f"{row['group'] + '/' + row['system']:24s} {row['qubits']:3d} "
              f"{row['fci_cpu_s']:8.2f} {row.get('fci_gpu_s', float('nan')):8.2f} | "
              f"{row['qpu_device_s']:8.2f} {row['qpu_wall_s']:8.2f} {sqd:8.2f} "
              f"{row['ccsd_seed_s'] + row['compile_s']:8.2f} {route:8.2f} | "
              f"{row['projected_wall_s'] + sqd:10.2f}")


if __name__ == "__main__":
    main()
