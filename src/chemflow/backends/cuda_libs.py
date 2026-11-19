"""Make CuPy resolve NVRTC from pip wheels before GPU4PySCF loads the system CUDA toolkit."""


def prefer_wheel_cuda_libraries() -> str | None:
    """Load NVRTC through ``cuda.pathfinder`` (CuPy's own locator) and return its path.

    GPU4PySCF opens cuSOLVER with ``ctypes.util.find_library``, which consults ldconfig.
    On a host with an older system CUDA toolkit (CUDA 12.0 next to RTX 5090 / sm_120,
    for example) that loads the system cuSOLVER, which pulls in the system cuBLAS and
    NVRTC. CuPy's locator prefers already loaded libraries, so CuPy then compiles every
    kernel with the old NVRTC and launches fail with CUDA_ERROR_NO_BINARY_FOR_GPU.
    Resolving NVRTC first fixes the choice to the wheel copy: the locator searches
    wheels before the system and caches its answer. Returns None when CuPy's locator or
    a CUDA library is unavailable.
    """
    try:
        from cuda import pathfinder
    except ImportError:
        return None
    try:
        return pathfinder.load_nvidia_dynamic_lib("nvrtc").abs_path
    except Exception:  # noqa: BLE001  any locator failure leaves CuPy's default search
        return None


def import_gpu4pyscf_single_device() -> bool:
    """Import GPU4PySCF so that each calculation runs on the current CUDA device only.

    GPU4PySCF 1.4+ reads the visible device count once at import and then spreads J/K
    builds over every device, with device 0 as the primary. A calculation started under
    ``cupy.cuda.Device(k)`` for k > 0 then fails with cudaErrorIllegalAddress, and one on
    device 0 occupies GPUs that other placements own. Importing it while the device
    count reads 1 keeps every calculation on the device it was placed on. Call this
    before anything imports GPU4PySCF; returns False when it is unavailable or already
    imported in multi-device mode.
    """
    import sys

    if "gpu4pyscf" in sys.modules:
        from gpu4pyscf.__config__ import num_devices

        return num_devices == 1
    prefer_wheel_cuda_libraries()
    try:
        import cupy
    except ImportError:
        return False
    runtime = cupy.cuda.runtime
    count = runtime.getDeviceCount
    runtime.getDeviceCount = lambda: 1
    try:
        import gpu4pyscf  # noqa: F401
    except ImportError:
        return False
    finally:
        runtime.getDeviceCount = count
    return True
