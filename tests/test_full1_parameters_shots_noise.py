

from chemflow.backends.aer_noise import noise_model


def test_all_layout_couplers_have_bidirectional_cz_noise():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    calibration = {"properties": {
        "qubit_metrics": [{"id": qubit} for qubit in (10, 11, 12)],
        "coupler_metrics": [{"qubits": [10, 12], "cz_fidelity": 0.0}]}}
    simulator = AerSimulator(noise_model=noise_model(calibration, [10, 11, 12]))
    for pair in ((0, 2), (2, 0)):
        circuit = QuantumCircuit(3)
        circuit.cz(*pair)
        circuit.measure_all()
        counts = simulator.run(circuit, shots=2000, seed_simulator=3).result().get_counts()
        assert counts.get("000", 0) < 1000
