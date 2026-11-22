"""Offline analysis of hardware_applications.py runs (no QPU time).

Per hardware sample set (one job each):
- Hellinger fidelity to the ideal samples of the same circuit variant;
- the fraction of shots in the target (N_alpha, N_beta) sector;
- the same two after tensored readout inversion with the measured calibration circuits,
  and after particle-number post-selection.

Per variant:
- SQD errors against exact CASCI, with and without recovery and extension;
- a control: SQD on as many uniformly random bitstrings as the device returned.

Per job: device execution time, submit-to-result wall time and the platform's charge,
from the QPU ledger. Application quantities (Diels-Alder barrier and reaction energy,
QM/MM interaction energy) are formed from the hardware SQD energies.
"""

import json
import math
from pathlib import Path

import numpy as np

HARTREE_KCAL = 627.509474
MEH = 1000.0


def _ledger(root: Path) -> dict[str, dict]:
    records = {}
    for path in root.glob("*/*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("job_id"):
            records[record["job_id"]] = record
    return records


def _seconds(start, end):
    from datetime import datetime

    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def job_timing(record: dict) -> dict:
    status = record.get("status_info", {})
    return {"job_id": record.get("job_id"), "backend": record.get("backend"),
            "shots": record.get("shots"), "layout": record.get("layout"),
            "cz": (record.get("native_ops") or {}).get("cz"), "depth": record.get("depth"),
            "device_s": status.get("execution_time"),
            "wall_s": _seconds(record.get("submitted_at"), record.get("completed_at")),
            "cost": record.get("cost"), "currency": record.get("currency"),
            "dd": record.get("dynamic_decoupling"), "dd_applied": record.get("dd_applied")}


def dense(counts: dict, n_qubits: int) -> np.ndarray:
    vector = np.zeros(1 << n_qubits)
    for key, count in counts.items():
        vector[int(key)] += count
    return vector / vector.sum()


def sparse_hellinger(first: dict, second: dict) -> float:
    total_first, total_second = sum(first.values()), sum(second.values())
    return sum(math.sqrt(count / total_first * second.get(key, 0) / total_second)
               for key, count in first.items()) ** 2


def uniform_fidelity(counts: dict, n_qubits: int) -> float:
    """Hellinger fidelity of uniformly random bitstrings to the distribution ``counts``.

    The noise floor of the fidelity: a spread-out ideal distribution scores higher
    against noise than a concentrated one, so fidelities are compared against it.
    """
    total = sum(counts.values())
    return sum(math.sqrt(count / total) for count in counts.values()) ** 2 / 2 ** n_qubits


def signal_fraction(valid: float, uniform_valid: float) -> float:
    """Weight of the ideal state in a globally depolarized output, from the in-sector
    fraction: valid = s * 1 + (1 - s) * uniform_valid. Particle number is conserved by
    every LUCJ variant, so this does not depend on the measured orbital basis."""
    return (valid - uniform_valid) / (1 - uniform_valid)


def sector_fraction(counts: dict, n_orb: int, n_alpha: int, n_beta: int) -> float:
    mask = (1 << n_orb) - 1
    valid = sum(count for key, count in counts.items()
                if (int(key) & mask).bit_count() == n_alpha and
                (int(key) >> n_orb).bit_count() == n_beta)
    return valid / sum(counts.values())


def uniform_sector_fraction(n_orb: int, n_alpha: int, n_beta: int) -> float:
    return math.comb(n_orb, n_alpha) * math.comb(n_orb, n_beta) / 4 ** n_orb


def confusion_from_calibration(zeros: dict, ones: dict, n_qubits: int) -> list[np.ndarray]:
    """Per logical qubit M[measured, prepared] from the all-0 and all-1 circuits."""
    def marginal_one(counts):
        total = sum(counts.values())
        return [sum(count for key, count in counts.items() if int(key) >> qubit & 1) / total
                for qubit in range(n_qubits)]

    flip_up, stay_up = marginal_one(zeros), marginal_one(ones)
    return [np.array([[1 - up, 1 - one], [up, one]]) for up, one in zip(flip_up, stay_up,
                                                                       strict=True)]


def apply_per_qubit(vector: np.ndarray, matrices: list[np.ndarray]) -> np.ndarray:
    n_qubits = len(matrices)
    tensor = vector.reshape((2,) * n_qubits)
    for qubit, matrix in enumerate(matrices):
        axis = n_qubits - 1 - qubit
        tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def mitigated(vector: np.ndarray, matrices: list[np.ndarray]) -> np.ndarray:
    corrected = np.clip(apply_per_qubit(vector, [np.linalg.inv(m) for m in matrices]), 0, None)
    return corrected / corrected.sum()


def sector_mask(n_orb: int, n_alpha: int, n_beta: int) -> np.ndarray:
    index = np.arange(1 << (2 * n_orb), dtype=np.int64)
    return ((np.bitwise_count(index & ((1 << n_orb) - 1)) == n_alpha) &
            (np.bitwise_count(index >> n_orb) == n_beta))


def post_select(vector: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kept = np.where(mask, vector, 0.0)
    return kept / kept.sum() if kept.sum() else kept


def _hellinger(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(first * second).sum() ** 2)


def uniform_control(hamiltonian_data, store, shots: int, *, extend: bool, seed: int = 2026,
                    **options) -> tuple[float, int]:
    """SQD energy and subspace size from ``shots`` uniformly random bitstrings."""
    from collections import Counter

    from chemflow.impls.sqd import sqd_sector
    from chemflow.types import Samples

    n_orb = hamiltonian_data.sector.n_orb
    draws = np.random.default_rng(seed).integers(0, 1 << (2 * n_orb), size=shots)
    counts = Counter(str(int(value)) for value in draws)
    samples = Samples(counts=dict(counts), shots=shots, encoding_id="uniform",
                      basis="computational", source="quantum_simulator", seed=seed)
    energy, diagnostics = sqd_sector(hamiltonian_data, hamiltonian_data.sector, store, samples,
                                     recovery=extend, with_ext=extend, seed=7, **options)
    return float(energy), int(diagnostics["subspace_dimension"])


def _rebuild(group: str, name: str, metadata: dict, store):
    """The active-space Hamiltonian of one system, rebuilt exactly as the plan built it."""
    from chemflow.impls.pyscf_impls import build_molecule
    from chemflow.types.model import ActiveSpaceSpec, ElectronicModel

    electrons, orbitals = metadata["active"]
    active = ActiveSpaceSpec(n_elec=electrons, n_orb=orbitals)
    if "atoms" in metadata:
        return build_molecule(metadata["atoms"], ElectronicModel(basis=metadata["basis"]),
                              store, active=active)
    import io
    from types import SimpleNamespace

    import openmm
    from openmm import app

    from chemflow.domains.qmmm import mm_system
    from chemflow.impls.pyscf_impls import active_from_scf
    from chemflow.plugins.openmm import QMMMCalculator

    system = metadata["qmmm"]
    mm = mm_system(store, system["pdb_text"], ["amber14-all.xml", system["ffxml"]])
    topology = app.PDBFile(io.StringIO(system["pdb_text"])).topology
    calculator = QMMMCalculator(topology, openmm.XmlSerializer.deserialize(
        store.get(mm.system_ref).tobytes().decode()), system["qm_atoms"], "6-31g*")
    mean_field, environment = calculator.embedded_scf(system[metadata["positions_key"]])
    mean_field.chemflow_environment_energy = environment
    identity = SimpleNamespace(atom=name, unit="angstrom", charge=0, spin=0)
    return active_from_scf(SimpleNamespace(mf=mean_field, mol=identity,
                                           model=ElectronicModel(basis="6-31g*")),
                           electrons, orbitals, store)


def analyse_group(payload: dict, ledger: dict, calibration: dict, metadata: dict) -> dict:
    from chemflow.runtime.artifacts import ArtifactStore

    out = {}
    for name, row in payload["rows"].items():
        some_bits = next(value for key, value in row.items() if key.endswith("_bits"))
        method = some_bits["provenance"]["method"]
        electrons, n_orb = method["active_electrons"], method["active_orbitals"]
        n_alpha = n_beta = electrons // 2
        n_qubits = 2 * n_orb
        casci = row["casci"]["value"]
        entry = {"qubits": n_qubits, "casci": casci, "rhf": row["rhf"]["value"],
                 "uniform_valid": uniform_sector_fraction(n_orb, n_alpha, n_beta),
                 "variants": {}}
        dense_ok = n_qubits <= 24
        mask = sector_mask(n_orb, n_alpha, n_beta) if dense_ok else None
        for key in [k for k in row if k.endswith("_bits") and not k.startswith("ideal_")]:
            variant = key[:-5]
            final = variant in ("base", "dd")
            ideal = row["ideal_full_bits" if final else "ideal_open_bits"]["counts"]
            ideal_dense = dense(ideal, n_qubits) if dense_ok else None
            floor = uniform_fidelity(ideal, n_qubits)
            parts = []
            for part in row[key]["parts"]:
                record = next((ledger[job] for job in part["job_ids"] if job in ledger), {})
                timing = job_timing(record) if record else {}
                valid = sector_fraction(part["counts"], n_orb, n_alpha, n_beta)
                stats = {"shots": part["shots"], "F": sparse_hellinger(part["counts"], ideal),
                         "F_uniform": floor, "valid": valid,
                         "signal": signal_fraction(valid, entry["uniform_valid"]), **timing}
                backend = record.get("backend")
                cal = calibration.get(f"{backend}_{n_qubits}") if record else None
                if dense_ok:
                    measured = dense(part["counts"], n_qubits)
                    stats["F_post"] = _hellinger(post_select(measured, mask),
                                                 post_select(ideal_dense, mask))
                    if cal is not None and cal["layout"] == record.get("layout"):
                        fixed = mitigated(measured, cal["matrices"])
                        stats["F_readout"] = _hellinger(fixed, ideal_dense)
                        stats["valid_readout"] = float(fixed[mask].sum())
                        stats["F_readout_post"] = _hellinger(post_select(fixed, mask),
                                                             post_select(ideal_dense, mask))
                    elif cal is not None:
                        stats["readout_note"] = "calibration layout differs"
                parts.append(stats)
            result = {"parts": parts,
                      "raw_error_meh": MEH * (row[f"{variant}_raw"]["value"] - casci),
                      "raw_dims": row[f"{variant}_raw"]["provenance"]["subspace_dimension"]}
            if f"{variant}_sqd" in row:
                result["sqd_error_meh"] = MEH * (row[f"{variant}_sqd"]["value"] - casci)
                result["sqd_dims"] = row[f"{variant}_sqd"]["provenance"]["subspace_dimension"]
                result["sector_dims"] = row[f"{variant}_sqd"]["provenance"]["sector_dimension"]
            entry["variants"][variant] = result
        for label in ("full", "open"):
            if f"ideal_{label}_raw" in row:
                entry[f"ideal_{label}_raw_error_meh"] = MEH * (
                    row[f"ideal_{label}_raw"]["value"] - casci)
                entry[f"ideal_{label}_raw_dims"] = row[f"ideal_{label}_raw"]["provenance"][
                    "subspace_dimension"]
        if name in metadata:
            store = ArtifactStore()
            data = _rebuild(payload["group"], name, metadata[name], store)
            shots = sum(part["shots"] for part in row["base_bits"]["parts"])
            entry["uniform_raw_error_meh"] = MEH * (uniform_control(
                data, store, shots, extend=False)[0] - casci)
            if n_qubits <= 12:
                energy, dims = uniform_control(data, store, shots, extend=True)
                entry["uniform_sqd_error_meh"] = MEH * (energy - casci)
                entry["uniform_sqd_dims"] = dims
        out[name] = entry
    return out


def calibrations(readout_payload: dict | None, ledger: dict) -> dict:
    found = {}
    if readout_payload is None:
        return found
    for name, row in readout_payload["rows"].items():
        width = int(name.split("_")[1])
        records = [next((ledger[job] for job in row[state]["job_ids"] if job in ledger), {})
                   for state in ("zeros", "ones")]
        layouts = {tuple(record.get("layout") or ()) for record in records}
        found[name] = {"layout": list(next(iter(layouts))) if len(layouts) == 1 else None,
                       "matrices": confusion_from_calibration(row["zeros"]["counts"],
                                                              row["ones"]["counts"], width),
                       "timing": [job_timing(record) for record in records if record]}
    return found


def applications(results: dict, reference: dict | None) -> dict:
    derived = {}
    da = results.get("diels_alder")
    if da:
        for variant in da["transition"]["variants"]:
            for kind in ("sqd", "raw"):
                key = f"{kind}_error_meh"
                if key not in da["transition"]["variants"][variant]:
                    continue

                def energy(name, variant=variant, key=key):
                    item = da[name]
                    return item["casci"] + item["variants"][variant][key] / MEH

                reactants = energy("butadiene") + energy("ethylene")
                derived[f"diels_alder/{variant}/{kind}"] = {
                    "barrier_kcal": HARTREE_KCAL * (energy("transition") - reactants),
                    "reaction_kcal": HARTREE_KCAL * (energy("product") - reactants)}
        exact = da["butadiene"]["casci"] + da["ethylene"]["casci"]
        derived["diels_alder/casci"] = {
            "barrier_kcal": HARTREE_KCAL * (da["transition"]["casci"] - exact),
            "reaction_kcal": HARTREE_KCAL * (da["product"]["casci"] - exact)}
    for group, (bound, separated) in (("t4l", ("bound", "separated")),
                                       ("large", ("t4l_bound", "t4l_separated"))):
        rows = results.get(group)
        if not rows or bound not in rows:
            continue
        derived[f"{group}/casci"] = {"interaction_kcal": HARTREE_KCAL * (
            rows[bound]["casci"] - rows[separated]["casci"])}
        for variant in rows[bound]["variants"]:
            for kind in ("sqd", "raw"):
                key = f"{kind}_error_meh"
                if key in rows[bound]["variants"][variant]:
                    derived[f"{group}/{variant}/{kind}"] = {"interaction_kcal": HARTREE_KCAL * (
                        rows[bound]["casci"] - rows[separated]["casci"] +
                        (rows[bound]["variants"][variant][key] -
                         rows[separated]["variants"][variant][key]) / MEH)}
    return derived


def analyse(out: Path, ledger_root: Path) -> dict:
    ledger = _ledger(ledger_root)
    loaded = {path.stem: json.loads(path.read_text()) for path in out.glob("*.json")
              if path.stem in ("readout", "diels_alder", "t4l", "hchain", "large")}
    calibration = calibrations(loaded.get("readout"), ledger)
    metadata = json.loads((out / "metadata.json").read_text()) if (
        out / "metadata.json").exists() else {}
    results = {group: analyse_group(payload, ledger, calibration, metadata.get(group, {}))
               for group, payload in loaded.items() if group != "readout"}
    report = {"systems": results, "applications": applications(results, None),
              "readout": {name: {"layout": item["layout"], "timing": item["timing"],
                                 "f0": [float(m[0, 0]) for m in item["matrices"]],
                                 "f1": [float(m[1, 1]) for m in item["matrices"]]}
                          for name, item in calibration.items()},
              "plans": {group: {"wall_s": payload["wall_s"], "profile": payload["profile"]}
                        for group, payload in loaded.items()}}
    (out / "analysis.json").write_text(json.dumps(report, indent=1, default=float))
    _print(report)
    return report


def _print(report: dict) -> None:
    for group, rows in report["systems"].items():
        print(f"\n== {group}")
        for name, entry in rows.items():
            print(f"{name} ({entry['qubits']} q): uniform valid {entry['uniform_valid']:.3f}; "
                  f"ideal raw {entry.get('ideal_full_raw_error_meh', float('nan')):.3f} mEh; "
                  f"uniform raw {entry.get('uniform_raw_error_meh', float('nan')):.1f} / "
                  f"full {entry.get('uniform_sqd_error_meh', float('nan')):.3f} mEh")
            for variant, result in entry["variants"].items():
                parts = result["parts"]
                mean = {key: float(np.mean([part[key] for part in parts if key in part]))
                        for key in ("F", "F_uniform", "valid", "signal", "F_post", "F_readout",
                                    "valid_readout",
                                    "F_readout_post", "device_s", "wall_s", "cz")
                        if any(key in part and part[key] is not None for part in parts)}
                print(f"   {variant:8s} " + " ".join(f"{key}={value:.4g}"
                                                     for key, value in mean.items()) +
                      f" | raw {result['raw_error_meh']:.3f} mEh [{result['raw_dims']}]" +
                      (f" | SQD {result['sqd_error_meh']:.4f} mEh [{result['sqd_dims']}]"
                       if "sqd_error_meh" in result else ""))
    print("\n== applications")
    for name, values in report["applications"].items():
        print(f"   {name:32s} " + " ".join(f"{key}={value:.3f}" for key, value in values.items()))
