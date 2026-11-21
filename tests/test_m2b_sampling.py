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
