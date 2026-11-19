import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, mcscf, scf

from chemflow.backends.gpu import FciKernel, cuda_available, fci_ground_state
from chemflow.impls.spin_complete import diagonalize_configurations
from chemflow.plugins.pyscf import SQDSolver


@pytest.fixture(scope="module")
def integrals():
    reference = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)).run()
    calculation = mcscf.CASCI(reference, 2, 2)
    h1, constant = calculation.get_h1eff()
    return h1, calculation.get_h2eff(), constant


@pytest.mark.parametrize("symmetry", [1, 4, 8])
@pytest.mark.parametrize("electrons", [2, (1, 1)])
def test_protocol_restored_eri_and_rdms(integrals, symmetry, electrons):
    h1, eri, constant = integrals
    eri = ao2mo.restore(symmetry, eri, 2)
    solver = SQDSolver(shots=4096, seed=3)
    energy, vector = solver.kernel(h1, eri, 2, electrons, ecore=constant,
                                   max_cycle=90, verbose=0, arbitrary_pyscf_option=True)
    exact_energy, exact = fci.direct_spin1.FCI().kernel(h1, eri, 2, electrons, ecore=constant)
    assert isinstance(vector, fci.selected_ci.SCIvector)
    assert energy == pytest.approx(exact_energy, abs=1e-9)
    for method in ("make_rdm1", "make_rdm1s", "make_rdm12", "make_rdm12s"):
        actual = getattr(solver, method)(vector, 2, electrons)
        expected = getattr(fci.direct_spin1, method)(exact, 2, electrons)
        if method in ("make_rdm12", "make_rdm12s"):
            for actual_part, expected_part in zip(actual, expected):
                np.testing.assert_allclose(actual_part, expected_part, atol=1e-8)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert solver.spin_square(vector, 2, electrons)[0] == pytest.approx(0, abs=1e-9)
    assert solver.nroots == 1 and solver.converged


def test_spin_constrained_native_wavefunction(integrals):
    h1, eri, constant = integrals
    solver = SQDSolver(shots=4096, spin=0)
    energy, vector = solver.kernel(h1, eri, 2, 2, ecore=constant)
    assert isinstance(vector, fci.selected_ci.SCIvector)
    dm1, dm2 = solver.make_rdm12(vector, 2, 2)
    reconstructed = (np.einsum("pq,qp", h1, dm1) +
                     .5 * np.einsum("pqrs,pqrs", ao2mo.restore(1, eri, 2), dm2) + constant)
    assert energy == pytest.approx(reconstructed, abs=1e-9)
    assert solver.spin_square(vector, 2, 2)[0] == pytest.approx(0, abs=1e-9)


def test_nonzero_spin_integer_electrons():
    solver = SQDSolver(shots=100, spin=1, ansatz="hf", init="zero")
    energy, vector = solver.kernel(np.diag([-1., 2.]), np.zeros((2,) * 4), 2, 1)
    assert energy == pytest.approx(-1.)
    assert solver.spin_square(vector, 2, 1)[0] == pytest.approx(.75)
    assert np.trace(solver.make_rdm1(vector, 2, 1)) == pytest.approx(1.)


@pytest.mark.parametrize("resample", ["once", "always"])
def test_sampling_policy_rediagonalizes(integrals, monkeypatch, resample):
    h1, eri, constant = integrals
    solver = SQDSolver(shots=4096, resample=resample)
    calls = []
    original = solver._sample

    def capture(store, prep):
        calls.append(prep)
        return original(store, prep)

    monkeypatch.setattr(solver, "_sample", capture)
    first, _ = solver.kernel(h1, eri, 2, 2, ecore=constant)
    second, _ = solver.kernel(h1 + np.eye(2) * .1, eri, 2, 2, ecore=constant)
    assert second == pytest.approx(first + .2, abs=1e-8)
    assert len(calls) == (1 if resample == "once" else 2)
    assert solver.last_run["samples_reused"] == (resample == "once")


def test_reused_samples_reject_different_sector(integrals):
    h1, eri, _ = integrals
    solver = SQDSolver(resample="once")
    solver.kernel(h1, eri, 2, 2)
    with pytest.raises(ValueError, match="sector"):
        solver.kernel(h1, eri, 2, 1)


@pytest.mark.parametrize("sampler", ["statevector", "aer_noisy", "gpu"])
def test_samplers_prepare_the_same_lucj_state(integrals, monkeypatch, sampler):
    if sampler == "gpu" and not cuda_available():
        pytest.skip("CUDA unavailable")
    if sampler == "aer_noisy":
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel

        monkeypatch.setattr("chemflow.backends.aer_noise.noise_model",
                            lambda *args: NoiseModel())
    h1, eri, _ = integrals
    reference = SQDSolver(shots=32768, seed=19)
    reference.kernel(h1, eri, 2, 2)
    solver = SQDSolver(sampler=sampler, shots=32768, seed=19)
    solver.kernel(h1, eri, 2, 2)
    probabilities = lambda samples: np.array([samples.counts.get(str(index), 0)
                                               for index in range(16)]) / samples.shots
    np.testing.assert_allclose(probabilities(solver._samples), probabilities(reference._samples),
                               atol=.02, rtol=0)
    if sampler == "aer_noisy":
        assert solver.last_run["device"].startswith("aer:noise(MQ02@")


def test_gpu_kernel_vector_export_in_pyscf_order(integrals):
    h1, eri, constant = integrals
    energy, _, diagnostics, vector = fci_ground_state(
        FciKernel(h1, eri, 2, (1, 1)), constant, return_wavefunction=True)
    exact_energy, exact = fci.direct_spin1.FCI().kernel(h1, eri, 2, 2, ecore=constant)
    assert energy == pytest.approx(exact_energy, abs=1e-9)
    assert diagnostics["residual"] < 1e-8
    np.testing.assert_allclose(fci.direct_spin1.make_rdm1(vector, 2, 2),
                               fci.direct_spin1.make_rdm1(exact, 2, 2), atol=1e-8)


def test_configuration_export_has_zeroes_outside_determinant_space():
    h1 = np.diag([-2., -.8, -.5])
    eri = np.zeros((3,) * 4)
    for indices in ((1, 2, 1, 2), (2, 1, 1, 2), (1, 2, 2, 1), (2, 1, 2, 1)):
        eri[indices] = .1
    energy, _, spin_squared, vector = diagonalize_configurations(
        h1, eri, 3, (2, 2), {(3, 5), (5, 3)}, 0, return_wavefunction=True)
    assert isinstance(vector, fci.selected_ci.SCIvector)
    assert vector.shape == (2, 2)
    assert vector[0, 0] == vector[1, 1] == 0
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert spin_squared == pytest.approx(0, abs=1e-10)
    dm1, dm2 = fci.selected_ci.SelectedCI().make_rdm12(vector, 3, (2, 2))
    reconstructed = np.einsum("pq,qp", h1, dm1) + .5 * np.einsum("pqrs,pqrs", eri, dm2)
    assert energy == pytest.approx(reconstructed, abs=1e-10)
