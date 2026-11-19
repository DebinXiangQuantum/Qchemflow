"""The NVRTC pre-resolution is a guarded no-op without CuPy's CUDA locator."""

import builtins
import sys
import types

from chemflow.backends import cuda_libs


def test_returns_none_without_the_locator(monkeypatch):
    real_import = builtins.__import__

    def without_pathfinder(name, *args, **kwargs):
        if name == "cuda" or name.startswith("cuda."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pathfinder)
    assert cuda_libs.prefer_wheel_cuda_libraries() is None


def test_returns_the_located_path_or_none_on_failure(monkeypatch):
    located = types.SimpleNamespace(abs_path="/wheels/nvidia/cu13/lib/libnvrtc.so.13")
    pathfinder = types.SimpleNamespace(load_nvidia_dynamic_lib=lambda name: located)
    cuda = types.ModuleType("cuda")
    cuda.pathfinder = pathfinder
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", pathfinder)
    assert cuda_libs.prefer_wheel_cuda_libraries() == located.abs_path

    def missing(name):
        raise RuntimeError("not found")

    pathfinder.load_nvidia_dynamic_lib = missing
    assert cuda_libs.prefer_wheel_cuda_libraries() is None


def test_gpu4pyscf_import_sees_one_device(monkeypatch):
    seen = {}
    runtime = types.SimpleNamespace(getDeviceCount=lambda: 4)
    cupy = types.ModuleType("cupy")
    cupy.cuda = types.SimpleNamespace(runtime=runtime)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gpu4pyscf":
            seen["count"] = runtime.getDeviceCount()
            module = types.ModuleType("gpu4pyscf")
            sys.modules["gpu4pyscf"] = module
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "gpu4pyscf", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(cuda_libs, "prefer_wheel_cuda_libraries", lambda: None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert cuda_libs.import_gpu4pyscf_single_device()
    finally:
        sys.modules.pop("gpu4pyscf", None)
    assert seen["count"] == 1
    assert runtime.getDeviceCount() == 4
