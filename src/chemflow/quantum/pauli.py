"""Qubit-zero-first Pauli measurement with covariance-aware uncertainties."""

import numpy as np


def qwc_groups(terms):
    groups = []
    widths = {len(word) for word in terms}
    if len(widths) > 1 or any(set(word) - set("IXYZ") for word in terms):
        raise ValueError("Invalid Pauli words")
    for word in sorted(terms):
        if not set(word) - {"I"}:
            continue
        for group in groups:
            if all(left == right or "I" in (left, right)
                   for left, right in zip(word, group["basis"], strict=True)):
                break
        else:
            group = {"basis": "I" * len(word), "terms": []}
            groups.append(group)
        group["terms"].append(word)
        group["basis"] = "".join(left if left != "I" else right for left, right in
                                 zip(word, group["basis"], strict=True))
    return groups


def rotation_circuit(basis):
    from qiskit import QuantumCircuit

    if set(basis) - set("IXYZ"):
        raise ValueError("Invalid measurement basis")
    circuit = QuantumCircuit(len(basis))
    for qubit, letter in enumerate(basis):
        if letter == "Y":
            circuit.sdg(qubit)
        if letter in "XY":
            circuit.h(qubit)
    return circuit


def estimate(terms, groups, counts, readout=None):
    energy = sum(value for word, value in terms.items() if not set(word) - {"I"})
    per_term = {word: 1.0 for word in terms if not set(word) - {"I"}}
    variances = []
    for group, measured in zip(groups, counts, strict=True):
        width = len(group["basis"])
        shots = sum(measured.values())
        if shots <= 0 or any(value < 0 for value in measured.values()):
            raise ValueError("Counts must have positive total and nonnegative entries")
        scores = np.zeros(len(measured))
        probabilities = np.array(list(measured.values()), dtype=float) / shots
        keys = [key.replace(" ", "") for key in measured]
        if any(len(key) != width or set(key) - {"0", "1"} for key in keys):
            raise ValueError("Counts must be qubit-zero-right bitstrings")
        for word in group["terms"]:
            outcomes = np.ones(len(keys))
            for qubit, letter in enumerate(word):
                if letter == "I":
                    continue
                p01, p10 = (readout or {}).get(qubit, (0, 0))
                if not (0 <= p01 <= 1 and 0 <= p10 <= 1) or abs(1-p01-p10) < 1e-12:
                    raise ValueError("Invalid or singular readout calibration")
                signs = np.array([1 - 2 * int(key[-1-qubit]) for key in keys])
                outcomes *= (signs - (p10-p01)) / (1-p01-p10)
            per_term[word] = float(probabilities @ outcomes)
            scores += terms[word] * outcomes
        mean = float(probabilities @ scores)
        variance = float(probabilities @ (scores - mean)**2) / max(1, shots-1)
        energy += mean
        variances.append(variance)
    return {"energy": float(energy), "stat_err": float(np.sqrt(sum(variances))),
            "per_term": per_term, "per_group_stat_err": np.sqrt(variances).tolist()}


def allocate_shots(total_shots, groups, standard_errors=None):
    """Deterministic largest-remainder allocation, at least one shot per group."""
    count = len(groups)
    if type(total_shots) is not int or total_shots < count:
        raise ValueError("Insufficient shot budget")
    if not count:
        return []
    weights = np.ones(count) if standard_errors is None else np.asarray(standard_errors, float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid group uncertainties")
    if not weights.sum():
        weights = np.ones(count)
    quotas = (total_shots-count) * weights / weights.sum()
    allocated = np.floor(quotas).astype(int) + 1
    for index in sorted(range(count), key=lambda index: (-(quotas[index] % 1), index))[
            :total_shots-int(allocated.sum())]:
        allocated[index] += 1
    return allocated.tolist()
