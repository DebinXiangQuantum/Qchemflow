"""Scrubbed device calibration and contiguous-chain selection."""

import json
from importlib.resources import files
from itertools import pairwise

import numpy as np


def load_calibration(backend: str = "MQ02") -> dict:
    if backend != "MQ02":
        raise ValueError(f"Unknown calibration: {backend}")
    return json.loads(files("chemflow.backends.data").joinpath(
        "mq02_calibration_20260925.json").read_text(encoding="utf-8"))


def best_layout(config: dict, n: int, beam: int = 4000) -> list[int]:
    """The best-scoring chain of ``n`` coupled physical qubits for a linear circuit.

    The score adds the log readout fidelities of the qubits and eight times the log CZ
    fidelities of the links. On a chain device (couplings i, i+1 only) the candidates are
    its contiguous windows; on any other coupling graph a beam search over simple paths
    keeps the ``beam`` best partial paths.
    """
    if n < 1 or n > config["qubits"]:
        raise ValueError("Invalid layout width")
    qubits = {item["id"]: item for item in config["properties"]["qubit_metrics"]}
    couplers = {tuple(sorted(item["qubits"])): item["cz_fidelity"]
                for item in config["properties"]["coupler_metrics"] if "qubits" in item}

    def site(index):
        return float(np.log(max(qubits[index].get("measure_f0", .95) *
                                qubits[index].get("measure_f1", .95), 1e-9)))

    def link(left, right):
        return float(8 * np.log(max(couplers.get(tuple(sorted((left, right))), .01), 1e-9)))

    def score(path):
        return sum(site(index) for index in path) + sum(
            link(left, right) for left, right in pairwise(path))

    edges = {tuple(sorted(edge)) for edge in
             (config.get("topology") or {}).get("coupling_map") or []}
    chain = {(index, index + 1) for index in range(config["qubits"] - 1)}
    if not edges or edges <= chain:
        return max((list(range(start, start + n)) for start in range(config["qubits"] - n + 1)),
                   key=score)
    neighbours: dict[int, list[int]] = {}
    for left, right in edges:
        neighbours.setdefault(left, []).append(right)
        neighbours.setdefault(right, []).append(left)
    paths = [(site(index), (index,)) for index in sorted(neighbours) if index in qubits]
    for _ in range(n - 1):
        grown = [(value + site(nxt) + link(path[-1], nxt), (*path, nxt))
                 for value, path in paths for nxt in neighbours[path[-1]]
                 if nxt not in path and nxt in qubits]
        if not grown:
            raise ValueError(f"No chain of {n} coupled qubits on this device")
        grown.sort(key=lambda item: -item[0])
        paths = grown[:beam]
    return list(max(paths, key=lambda item: item[0])[1])
