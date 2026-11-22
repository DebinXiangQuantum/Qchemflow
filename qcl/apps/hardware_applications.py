"""The quantum parts of the three applications on real QPUs, with mitigation variants.

Each active space below is sampled on lqcloud hardware with the CCSD-seeded LUCJ circuit
and diagonalized by SQD, next to the exact CASCI energy and ideal (noiseless) samples of
the same circuit:

    diels_alder  pi spaces of ethylene (4 qubits), butadiene (8), the transition state
                 and cyclohexene (12 each), on MQ02: a hardware barrier and reaction energy
    t4l          benzene CASCI(6,6) in the field of the whole T4 lysozyme (12 qubits),
                 bound and separated, on MQ02: a hardware QM/MM interaction energy
    hchain       H12 in STO-6G at 1.8 and 3.2 bohr (24 qubits, the whole of MQ02)
    large        CASCI(14,14) (28 qubits) of the Diels-Alder transition state and of the
                 embedded benzene (bound, separated), on QZ03 (35-qubit grid)
    readout      all-0 and all-1 circuits on every layout width, for readout calibration

Mitigation variants (each a separate job; the circuit-level ones are exact):

    base     the LUCJ circuit as before
    open     final orbital rotation omitted, about half the CZ gates; SQD rotates the
             Hamiltonian to the sampled orbital basis
    dd       platform dynamical decoupling (CPMG, 4 pulses per idle window)
    open_dd  both

Readout inversion and particle-number post-selection are applied offline by ``analyse``.

    . ~/.config/chemflow/lqcloud.env
    QCL_POLICY_FILE=qcl/apps/hardware-policy.toml CUDA_VISIBLE_DEVICES=2,3 \\
        python qcl/apps/hardware_applications.py run --out hw --da da/result.json --t4l t4l
    python qcl/apps/hardware_applications.py analyse --out hw

``--qpu sim`` replaces every QPU by a GPU-hosted statevector simulator (no charge).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import qcl

sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = {"base": (True, "none"), "open": (False, "none"), "dd": (True, "cpmg"),
            "open_dd": (False, "cpmg")}
GROUPS = ("readout", "diels_alder", "t4l", "hchain", "large")
SPACINGS_BOHR = (1.8, 3.2)


def systems(group, args):
    """name -> (mean-field builder spec, active (electrons, orbitals), backend, variants)."""
    if group == "diels_alder":
        structures = json.loads(Path(args.da).read_text())["structures"]
        pi = {"ethylene": (2, 2), "butadiene": (4, 4), "transition": (6, 6), "product": (6, 6)}
        return {name: ({"atoms": structures[name], "basis": "6-31g*"}, pi[name], "MQ02",
                       tuple(VARIANTS)) for name in pi}
    if group == "t4l":
        system = _t4l(args)
        return {state: ({"qmmm": system, "positions": system[key]}, (6, 6), "MQ02",
                        ("base", "open_dd"))
                for state, key in (("bound", "positions"), ("separated", "separated"))}
    if group == "hchain":
        from hydrogen_chain import chain

        return {f"R{spacing}": ({"atoms": chain(12, spacing, "sto-6g").atoms,
                                 "basis": "sto-6g"}, (12, 12), "MQ02", ("base", "open_dd"))
                for spacing in SPACINGS_BOHR}
    if group == "large":
        structures = json.loads(Path(args.da).read_text())["structures"]
        system = _t4l(args)
        return {"da_transition": ({"atoms": structures["transition"], "basis": "6-31g*"},
                                  (14, 14), "QZ03", ("base", "open_dd")),
                **{f"t4l_{state}": ({"qmmm": system, "positions": system[key]}, (14, 14),
                                    "QZ03", ("base", "open_dd"))
                   for state, key in (("bound", "positions"), ("separated", "separated"))}}
    raise ValueError(group)


def _t4l(args):
    from t4l_quantum_binding import prepare

    folder = Path(args.t4l)
    folder.mkdir(parents=True, exist_ok=True)
    return prepare(folder)


def qpus(backend, qubits, dd, copies, simulate, gpu):
    if simulate:
        return [qcl.Device(f"sim-{backend.lower()}-{dd}-{index}", "qpu", qubits=qubits,
                           simulator_host=gpu.id, simulator_method="statevector")
                for index in range(copies)]
    return [qcl.Device(f"{backend.lower()}-{dd}-{index}", "qpu", qubits=qubits,
                       site="remote", billing="metered", provider="lqcloud",
                       backend=backend).with_options(dd=dd) for index in range(copies)]


@qcl.workflow
def hardware(specs, cpu, gpus, *, shots, repeats, ideal_shots, simulate, seed=17):
    rows = {}
    for index, (name, (source, active, backend, variants)) in enumerate(specs.items()):
        gpu = gpus[index % len(gpus)]
        if "qmmm" in source:
            system = source["qmmm"]
            mean_field = qcl.qmmm.scf(system["pdb_text"], source["positions"],
                                      system["qm_atoms"],
                                      forcefields=("amber14-all.xml", system["ffxml"]),
                                      basis="6-31g*").to(cpu)
        else:
            mean_field = qcl.scf.hf(qcl.Molecule(source["atoms"], source["basis"])).to(cpu)
        space = qcl.active_space(mean_field, qcl.Active(*active)).to(cpu)
        amplitudes = qcl.ccsd.amplitudes(space).to(cpu)
        qubits = 2 * active[1]
        exact = (qcl.fci.solve(space.hamiltonian, space.sector).to(cpu) if active[1] <= 8 else
                 qcl.fci.solve(space.hamiltonian.to(gpu), space.sector.to(gpu)).to(gpu))
        large = active[1] >= 12
        # Full SQD on 24+ noisy qubits fills most of the sector: bound its cost.
        full = {"n_batches": 4, "max_iter": 2} if large else {}
        row = {"casci": exact.energy, "rhf": mean_field.energy}
        simulator = qcl.Device(f"ideal-{gpu.id}", "qpu", qubits=qubits,
                               simulator_host=gpu.id, simulator_method="statevector")
        for final in sorted({VARIANTS[variant][0] for variant in variants}):
            kernel = qcl.kernels.lucj(space, params=amplitudes, final_rotation=final)
            ideal = qcl.sample(kernel, shots=ideal_shots, seed=seed + index).to(
                simulator).to(cpu)
            label = "full" if final else "open"
            row[f"ideal_{label}_bits"] = ideal
            row[f"ideal_{label}_raw"] = qcl.sqd.diagonalize(
                space.hamiltonian, ideal, space.sector, recovery=False, extend=False).to(cpu)
        for variant in variants:
            final, dd = VARIANTS[variant]
            kernel = qcl.kernels.lucj(space, params=amplitudes, final_rotation=final)
            count = 1 if large else repeats
            per_job = shots if not large else 20000
            devices = qpus(backend, 35 if backend == "QZ03" else 24, dd, count, simulate, gpu)
            bits = qcl.sample(kernel, shots=[per_job] * count, seed=seed + index).to(
                devices).to(cpu)
            row[f"{variant}_bits"] = bits
            row[f"{variant}_raw"] = qcl.sqd.diagonalize(
                space.hamiltonian, bits, space.sector, recovery=False, extend=False).to(cpu)
            if active[1] < 14:
                row[f"{variant}_sqd"] = qcl.sqd.diagonalize(
                    space.hamiltonian, bits, space.sector, **full).to(cpu)
        rows[name] = row
    return rows


def calibration_kernels(width):
    @qcl.quantum_kernel
    def zeros(qubits: qcl.Qubits[width], angle: qcl.Params[1]):
        qcl.ry(qubits[0], angle[0])

    @qcl.quantum_kernel
    def ones(qubits: qcl.Qubits[width], angle: qcl.Params[1]):
        for index in range(width):
            qcl.ry(qubits[index], angle[0])

    return zeros, ones


@qcl.workflow
def readout(widths, cpu, gpu, *, shots, simulate):
    rows = {}
    for backend, width in widths:
        zeros, ones = calibration_kernels(width)
        device = qpus(backend, 35 if backend == "QZ03" else 24, "none", 1, simulate, gpu)[0]
        rows[f"{backend}_{width}"] = {
            "zeros": qcl.sample(zeros, params=[0.0], shots=shots, seed=1).to(device).to(cpu),
            "ones": qcl.sample(ones, params=[math.pi], shots=shots, seed=1).to(device).to(cpu)}
    return rows


def _encode(value):
    if isinstance(value, qcl.Bitstrings):
        return {"counts": value.counts, "shots": value.shots, "source": value.source,
                "device": value.device, "job_ids": list(value.job_ids),
                "parts": list(value.parts), "provenance": value.provenance}
    return {"value": value.value, "unit": value.unit, "provenance": value.provenance}


def _write_metadata(out, group, specs):
    """How to rebuild each Hamiltonian offline (hardware_analysis uniform controls)."""
    path = out / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    entries = {}
    for name, (source, active, _backend, _variants) in specs.items():
        if "qmmm" in source:
            key = next(key for key in ("positions", "separated")
                       if source["qmmm"][key] is source["positions"])
            entries[name] = {"qmmm": {field: source["qmmm"][field] for field in
                                      ("pdb_text", "ffxml", "qm_atoms", "positions",
                                       "separated")},
                             "positions_key": key, "active": list(active)}
        else:
            entries[name] = {"atoms": source["atoms"], "basis": source["basis"],
                             "active": list(active)}
    metadata[group] = entries
    path.write_text(json.dumps(metadata))


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cluster = qcl.Cluster.local()
    if not cluster.gpus:
        raise SystemExit("This program needs a GPU for the ideal samples")
    for group in args.groups:
        target = out / f"{group}.json"
        if target.exists() and not args.force:
            print(f"{group}: already done ({target})")
            continue
        started = time.perf_counter()
        if group == "readout":
            widths = [("MQ02", width) for width in (4, 8, 12, 24)] + [("QZ03", 28)]
            plan = readout.compile(widths, cluster.cpus[0], cluster.gpus[0],
                                   shots=args.shots, simulate=args.qpu == "sim")
        else:
            specs = systems(group, args)
            _write_metadata(out, group, specs)
            plan = hardware.compile(specs, cluster.cpus[0], cluster.gpus,
                                    shots=args.shots, repeats=args.repeats,
                                    ideal_shots=args.ideal_shots,
                                    simulate=args.qpu == "sim")
        estimate = plan.estimate()
        print(f"{group}: {estimate['shots']} shots, forecast QPU cost {estimate['qpu_cost']} "
              f"{estimate['currency']}, QPU {estimate['qpu_s']} s", flush=True)
        result = plan.run(timeout_s=args.timeout)
        wall = time.perf_counter() - started
        payload = {"group": group, "qpu": args.qpu, "wall_s": wall,
                   "cpu_core_s": plan.cpu_core_s, "plan_hash": plan.hash,
                   "estimate": estimate, "profile": plan.profile(),
                   "rows": {name: {key: _encode(value) for key, value in row.items()}
                            for name, row in result.items()}}
        target.write_text(json.dumps(payload, indent=1, default=str))
        print(f"{group}: done in {wall:.0f} s -> {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--out", required=True)
    runner.add_argument("--da", required=True, help="diels_alder.py result JSON")
    runner.add_argument("--t4l", required=True, help="folder for the prepared T4L system")
    runner.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    runner.add_argument("--qpu", choices=("real", "sim"), default="real")
    runner.add_argument("--shots", type=int, default=8000)
    runner.add_argument("--repeats", type=int, default=2)
    runner.add_argument("--ideal-shots", type=int, default=1000000)
    runner.add_argument("--timeout", type=float, default=6 * 3600)
    runner.add_argument("--force", action="store_true")
    analyser = commands.add_parser("analyse")
    analyser.add_argument("--out", required=True)
    analyser.add_argument("--ledger", default=str(Path.home() / ".chemflow" / "qpu"))
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        from hardware_analysis import analyse

        analyse(Path(args.out), Path(args.ledger))


if __name__ == "__main__":
    main()
