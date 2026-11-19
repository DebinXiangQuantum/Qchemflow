import inspect
import sys
from types import SimpleNamespace

import pytest

from chemflow import build
from chemflow.backends import gpu
from chemflow.backends.aer_noise import noise_model, register_noisy_aer
from chemflow.backends.lqcloud import LqcloudBackend, register_lqcloud
from chemflow.check import check
from chemflow.diagnostics import AdapterError
from chemflow.impls.statevector import pauli_measurement_groups
from chemflow.ir.schema import Node, Program
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor


def test_every_registration_rejects_identity_parameters(monkeypatch, tmp_path):
    registry = molecular_registry()
    register_noisy_aer(registry, {})
    register_lqcloud(registry, LqcloudBackend(state_dir=tmp_path))
    monkeypatch.setattr(gpu, "cuda_available", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    gpu.register_gpu(registry)
    with build.program("registration-inputs") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        scf = builder.node("scf", molecule=molecule, model=model)
        sector = builder.sector(n_alpha=1, n_beta=1, n_orb=2)
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        samples = builder.node("sample_configurations", prep=prep, shots=8, seed=0)
        energy = builder.ground_energy(hamiltonian, sector)
    sources = {"molecule": molecule.id, "model": model.id, "hamiltonian": hamiltonian.id,
               "scf": scf.id, "sector": sector.id, "encoding": encoding.id, "operator": encoding.id,
               "prep": prep.id, "samples": samples.id, "left": energy.id, "right": energy.id}
    identities = {"method", "sampler", "solver", "backend"}
    for identifier, implementation in registry.implementations.items():
        assert not identities.intersection(inspect.signature(implementation.adapter).parameters)
        assert isinstance(implementation.params, frozenset)
        assert not identities.intersection(implementation.params)
        for identity in identities:
            parameters = {identity: "forged"}
            if implementation.target == "molecule":
                parameters["atoms"] = "H 0 0 0"
            if implementation.target == "active_space":
                parameters.update(n_elec=2, n_orb=2)
            ports = registry.operators[f"chemflow.{implementation.target}@1"].inputs
            if implementation.implements in registry.refinements:
                ports = {**ports, **registry.refinements[implementation.implements].inputs}
            program = Program(name="identity", nodes=[*builder.to_ir().nodes,
                Node(id="probe", op=implementation.target, binding=identifier, params=parameters,
                     refinement=implementation.implements,
                     inputs={port: sources[port] for port in ports})])
            diagnostics = check(program, registry)
            assert any(item.code == "CF-E080" and identity in item.message and
                       "Unsupported implementation parameters" in item.message
                       for item in diagnostics), (identifier, identity, diagnostics)


def sample_program(binding, shots=10):
    with build.program("shot-cost") as builder:
        hamiltonian = builder.hamiltonian(builder.molecule("H 0 0 0; H 0 0 0.74"),
                                          builder.electronic_model(basis="sto-3g"))
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        sample = builder.node("sample", operator=encoding, prep=prep, shots=shots, seed=3)
        builder.bind(sample, binding)
    return builder.to_ir(), encoding.id, sample.id


@pytest.mark.parametrize("binding", ["pauli_shot_estimator", "ideal_energy_projective_sampler"])
def test_sample_predictions_and_measured_shots_agree(binding):
    registry = molecular_registry()
    program, encoding, sample = sample_program(binding)
    computed = Executor(registry).run(program)
    operator = computed[encoding][1]
    groups = pauli_measurement_groups(operator) if binding == "pauli_shot_estimator" else 1
    assert registry.implementations[binding].resource_model(shots=10)["sim_shots"] == (
        None if binding == "pauli_shot_estimator" else 10)
    prediction = registry.implementations[binding].resource_model(shots=10, operator=operator)
    assert prediction["sim_shots"] == computed["costs"][sample].sim_shots == 10 * groups
    program.budget = {"sim_shots": 10 * groups - 1}
    calls = []
    registry.implementations[binding].adapter = lambda **kwargs: calls.append(kwargs)
    if binding == "ideal_energy_projective_sampler":
        with pytest.raises(AdapterError, match="CF-E070"):
            Executor(registry).run(program)
    else:
        refused = Executor(registry).run(program)["refused"]
        assert "CF-E070" in refused[sample]
    assert calls == []


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
