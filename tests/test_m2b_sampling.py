import numpy as np

from chemflow.impls.statevector import sample_configurations


def test_configuration_frequencies_and_readout():
    state = np.zeros(16)
    state[3] = np.sqrt(0.7)
    state[12] = np.sqrt(0.3)
    clean = sample_configurations(state, 20000, 17)
    assert clean.basis == "computational"
    assert clean.source == "quantum_simulator"
    assert abs(clean.counts["3"] / clean.shots - 0.7) < 0.02
    noisy = sample_configurations(state, 20000, 17, 0.05)
    out_sector = sum(value for key, value in noisy.counts.items()
                     if int(key).bit_count() != 2)
    assert abs(out_sector / noisy.shots - (4 * 0.05 * 0.95**3 + 4 * 0.05**3 * 0.95)) < 0.02


def test_sampling_node_charges_simulator_shots():
    from chemflow import build
    from chemflow.ops.molecular import molecular_registry
    from chemflow.runtime import Executor

    with build.program("configuration-sampling") as builder:
        molecule = builder.molecule("H 0 0 0; H 0 0 0.74")
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(molecule, model)
        encoding = builder.node("encode", hamiltonian=hamiltonian)
        prep = builder.node("prepare", hamiltonian=hamiltonian, encoding=encoding)
        sampled = builder.node("sample_configurations", prep=prep, shots=100, seed=4)
    result = Executor(molecular_registry()).run(builder.to_ir())[sampled.id]
    assert result.shots == 100
    assert set(result.counts) == {"5"}
