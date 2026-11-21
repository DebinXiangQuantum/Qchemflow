"""Reviewer regression tests for the full-project review FULL-1 (R77–R82; written by Claude;
implementers must not edit). Findings came from a read-only Codex (gpt-6-astra) review and were
reproduced by the reviewer before being recorded. R79 revised for R1 (QPU as hardware)."""
from __future__ import annotations

import pytest


def _base(tools):
    empty = tools.call("modeler", "plan.new", {})["data"]["plan_id"]
    return tools.call("modeler", "plan.base", {"plan_id": empty,
                                               "active": {"n_elec": 2, "n_orb": 2}})["data"]["plan_id"]


def test_r82_cz_noise_applies_in_both_qubit_orders() -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from chemflow.backends.aer_noise import noise_model

    config = {"properties": {
        "qubit_metrics": [{"id": 0, "measure_f0": 1, "measure_f1": 1},
                          {"id": 1, "measure_f0": 1, "measure_f1": 1}],
        "coupler_metrics": [{"qubits": [0, 1], "cz_fidelity": 0.0}]}}
    noise = noise_model(config, [0, 1])
    for pair in ((0, 1), (1, 0)):
        circuit = QuantumCircuit(2)
        circuit.cz(*pair)
        circuit.measure_all()
        counts = AerSimulator(noise_model=noise).run(circuit, shots=2000,
                                                      seed_simulator=3).result().get_counts()
        assert counts.get("00", 0) < 1000, f"CZ{pair} ran noiseless"
