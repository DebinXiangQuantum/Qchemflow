"""Scrubbed device calibration and contiguous-chain selection."""

import json
from importlib.resources import files
from itertools import pairwise

import numpy as np


def load_calibration(backend: str = "MQ02") -> dict:
    if backend != "MQ02":
        raise ValueError(f"Unknown calibration: {backend}")
    return json.loads(files("chemflow.backends.data").joinpath(
        "mq02_calibration_20260925.json").read_text())


def best_layout(config: dict, n: int) -> list[int]:
    if n < 1 or n > config["qubits"]:
        raise ValueError("Invalid layout width")
    qubits = {item["id"]: item for item in config["properties"]["qubit_metrics"]}
    couplers = {tuple(sorted(item["qubits"])): item["cz_fidelity"]
                for item in config["properties"]["coupler_metrics"] if "qubits" in item}

    def score(path):
        return sum(np.log(max(qubits[index].get("measure_f0", .95) *
                              qubits[index].get("measure_f1", .95), 1e-9)) for index in path) + sum(
            8 * np.log(max(couplers.get((left, right), .01), 1e-9))
            for left, right in pairwise(path))

    return max((list(range(start, start + n)) for start in range(config["qubits"] - n + 1)),
               key=score)
