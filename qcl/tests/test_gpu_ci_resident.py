import numpy as np
import pytest
from pyscf import fci

from chemflow.backends.gpu import FciKernel, fci_ground_state
from chemflow.diagnostics import AdapterError


def test_gpu_resident_solver_rejects_numpy_kernel():
    kernel = FciKernel(np.eye(2), np.zeros((2, 2, 2, 2)), 2, (1, 1))
    with pytest.raises(ValueError, match="CuPy kernel"):
        fci_ground_state(kernel, 0, gpu_resident=True)


@pytest.mark.parametrize("gpu_id", [0, 1])
def test_gpu_fci_eigensolver_keeps_ci_vector_on_device(gpu_id):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= gpu_id:
            pytest.skip("Selected CUDA device is unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CUDA is unavailable")

    h1 = np.array([[-1.0, 0.12], [0.12, -0.3]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.6
    eri[1, 1, 1, 1] = 0.4
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.15
    with cp.cuda.Device(gpu_id):
        kernel = FciKernel(h1, eri, 2, (1, 1), xp=cp)
        energy, _spin, diagnostics, vector = fci_ground_state(
            kernel, 0, return_wavefunction=True, residual_tol=1e-9,
            gpu_resident=True)
        assert isinstance(vector, cp.ndarray)
        assert vector.device.id == gpu_id
        residual = cp.linalg.norm(kernel.sigma(vector) - energy * vector.ravel())
        assert float(residual) < 1e-9
        assert diagnostics["solver"].startswith("cupy")
    reference, _ = fci.direct_spin1.kernel(h1, eri, 2, (1, 1))
    assert energy == pytest.approx(reference, abs=1e-10)


def test_gpu_resident_solver_rejects_false_convergence(monkeypatch):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA is unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CUDA is unavailable")
    from cupyx.scipy.sparse import linalg

    monkeypatch.setattr(linalg, "eigsh", lambda *_args, **_kwargs:
                        (cp.asarray([0.0]), cp.ones((4, 1)) / 2))
    h1 = np.diag([-1.0, -0.3])
    eri = np.zeros((2, 2, 2, 2))
    with pytest.raises(AdapterError, match="residual exceeds tolerance"):
        fci_ground_state(FciKernel(h1, eri, 2, (1, 1), xp=cp), 0,
                         return_wavefunction=True, residual_tol=1e-9,
                         gpu_resident=True)
