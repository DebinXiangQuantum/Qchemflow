"""Circuit-level simulation with a snapshot of physical-device errors."""

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

from chemflow.backends.calibration import best_layout
from chemflow.backends.circuits import build_circuit, compile_native, memory_to_index
from chemflow.diagnostics import AdapterError
from chemflow.registry import ImplementationSpec
from chemflow.runtime import ExecutionResult
from chemflow.types import BackendKind, Samples


def noise_model(calibration: dict, layout: list[int]) -> NoiseModel:
    noise = NoiseModel()
    physical = {item["id"]: item for item in calibration["properties"]["qubit_metrics"]}
    couplers = {tuple(sorted(item["qubits"])): item for item in
                calibration["properties"]["coupler_metrics"] if "qubits" in item}
    for logical, physical_id in enumerate(layout):
        metrics = physical[physical_id]
        fidelity_zero = metrics.get("measure_f0", 1.0)
        fidelity_one = metrics.get("measure_f1", 1.0)
        if fidelity_zero < 1 or fidelity_one < 1:
            noise.add_readout_error(ReadoutError([[fidelity_zero, 1 - fidelity_zero],
                                                  [1 - fidelity_one, fidelity_one]]), [logical])
        strength = max(0, min(1, 2 * (1 - metrics.get("xeb_fidelity", 1.0))))
        for gate in ("sx", "x"):
            error = depolarizing_error(strength, 1) if strength else None
            timing = metrics.get("timing", {})
            duration = timing.get("xy", 0) * (2 if gate == "x" else 1) / 1000
            if metrics.get("t1") and metrics.get("t2") and duration:
                thermal = thermal_relaxation_error(metrics["t1"], metrics["t2"], duration)
                error = error.compose(thermal) if error else thermal
            if error:
                noise.add_quantum_error(error, gate, [logical])
    for pair, metrics in couplers.items():
        if not all(qubit in layout for qubit in pair):
            continue
        first, second = (layout.index(qubit) for qubit in pair)
        strength = max(0, min(1, 4 * (1 - metrics.get("cz_fidelity", 1.0)) / 3))
        if strength:
            noise.add_quantum_error(depolarizing_error(strength, 2), "cz",
                                    [first, second])
            noise.add_quantum_error(depolarizing_error(strength, 2), "cz", [second, first])
    return noise


def register_noisy_aer(registry, calibration: dict, identifier: str = "aer_noisy_mq02") -> None:
    def sample(store, prep, shots, seed=0, **kwargs):
        if prep.circuit is None:
            raise AdapterError("CF-E080", None, "State has no hardware circuit",
                               "Prepare HF or LUCJ")
        native = compile_native(build_circuit(prep.circuit))
        layout = best_layout(calibration, native.num_qubits)
        simulator = AerSimulator(noise_model=noise_model(calibration, layout))
        counts = simulator.run(native, shots=shots, seed_simulator=seed).result().get_counts()
        device = (f"aer:noise({calibration['name']}@"
                  f"{calibration['properties']['last_update']})")
        samples = Samples(counts={str(memory_to_index(bits, q0_left=False)): count
                                  for bits, count in counts.items()}, shots=shots,
                          encoding_id=prep.encoding_id, basis="computational",
                          source="quantum_simulator", seed=seed, device=device)
        return ExecutionResult(samples, device=device)

    registry.register(ImplementationSpec(id=identifier, version="1", target="sample_configurations",
        backend_kind=BackendKind.quantum_simulator, effects="random", adapter=sample,
        execution_role="emulator", emulates="sample_configurations",
        params=frozenset({"shots", "seed"}),
        resource_model=lambda shots=None, **kwargs: {"sim_shots": shots}))
