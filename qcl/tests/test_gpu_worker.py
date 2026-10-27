import sys
from types import SimpleNamespace

import pytest
from qcl._frontend import _isolated_gpu_worker, _restore_worker_gpu


def test_single_gpu_worker_isolates_requested_visible_device(monkeypatch):
    runtime = SimpleNamespace(getDeviceCount=lambda: 2)
    monkeypatch.setitem(sys.modules, "cupy",
                        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    graph = {"nodes": [
        {"device": {"kind": "gpu", "id": "gpu1", "simulator_host": None}},
        {"device": {"kind": "qpu", "id": "sim1", "simulator_host": "gpu1"},
         "params": {"devices": [{"simulator_host": "gpu1"}]}},
    ]}
    worker_graph, environment, gpu_id = _isolated_gpu_worker(graph)
    assert gpu_id == "gpu1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert graph["nodes"][0]["device"]["id"] == "gpu1"
    assert worker_graph["nodes"][0]["device"]["id"] == "gpu0"
    assert worker_graph["nodes"][1]["device"]["simulator_host"] == "gpu0"
    assert worker_graph["nodes"][1]["params"]["devices"][0]["simulator_host"] == "gpu0"
    assert _restore_worker_gpu({"result": {"device": "gpu0"},
                                "profile": [{"device": "gpu0"}]}, gpu_id) == {
        "result": {"device": "gpu1"}, "profile": [{"device": "gpu1"}]}


def test_worker_rejects_an_unavailable_gpu_ordinal(monkeypatch):
    runtime = SimpleNamespace(getDeviceCount=lambda: 1)
    monkeypatch.setitem(sys.modules, "cupy",
                        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    graph = {"nodes": [{"device": {"kind": "gpu", "id": "gpu1"}}]}
    with pytest.raises(RuntimeError, match="QCL-E503: gpu1 is not a visible CUDA device"):
        _isolated_gpu_worker(graph)
