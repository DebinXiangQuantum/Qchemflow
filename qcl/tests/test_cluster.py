import sys
from types import SimpleNamespace

import pytest

import qcl


def test_cluster_loads_devices_and_explicit_billing(tmp_path):
    path = tmp_path / "cluster.toml"
    path.write_text("""
[[cpu]]
id = "cpu0"
threads = 4

[[gpu]]
id = "gpu0"
memory_gb = 24

[[qpu]]
id = "remote-free"
qubits = 8
site = "remote"

[[qpu]]
id = "cloud"
qubits = 8
site = "remote"
billing = "metered"

[[link]]
between = ["cpu0", "cloud"]
kind = "network"
bandwidth_GBps = 1
latency_us = 100
""")

    cluster = qcl.Cluster.load(path)
    assert [device.id for device in cluster.cpus] == ["cpu0"]
    assert cluster["cpu0"].threads == 4
    assert cluster["gpu0"].memory_gb == 24
    assert cluster["cloud"].qubits == 8
    assert cluster["gpu0"].with_options(precision="fp64").options["precision"] == "fp64"
    assert cluster["remote-free"].billing == "none"
    assert cluster["cloud"].billing == "metered"
    with pytest.raises(qcl.CompileError) as error:
        cluster["cpu0"].with_options(precision="fp32")
    assert error.value.diagnostics[0]["code"] == "QCL-E202"
    for device, options in (
        (cluster["cpu0"], {"threads": 0}),
        (cluster["gpu0"], {"precision": "fp16"}),
        (cluster["cloud"], {"transpile": -1}),
    ):
        with pytest.raises(qcl.CompileError) as error:
            device.with_options(**options)
        assert error.value.diagnostics[0]["code"] == "QCL-E202"


def test_cluster_rejects_unknown_link_endpoint(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("""
[[cpu]]
id = "cpu0"
threads = 2

[[link]]
between = ["cpu0", "missing"]
kind = "network"
bandwidth_GBps = 1
latency_us = 10
""")
    with pytest.raises(ValueError, match="missing"):
        qcl.Cluster.load(path)


def test_local_simulator_declares_its_backing_device():
    cluster = qcl.Cluster.local()
    host_ids = {device.id for device in (*cluster.cpus, *cluster.gpus)}
    assert cluster.qpus
    assert all(device.simulator_host in host_ids for device in cluster.qpus)
    assert all(device.billing == "none" for device in cluster.qpus)


def test_local_cluster_reports_each_gpu_capacity_without_losing_partial_devices(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 2

        @staticmethod
        def getDeviceProperties(index):
            if index == 1:
                raise RuntimeError("Properties unavailable")
            return {"totalGlobalMem": 16_000_000_000}

    monkeypatch.setitem(sys.modules, "cupy",
                        SimpleNamespace(cuda=SimpleNamespace(runtime=Runtime())))
    cluster = qcl.Cluster.local()
    assert [(gpu.id, gpu.memory_gb) for gpu in cluster.gpus] == [
        ("gpu0", 16.0), ("gpu1", None)]
    assert [qpu.simulator_host for qpu in cluster.qpus] == ["gpu0", "gpu1"]
    with pytest.raises(qcl.CompileError, match="QCL-E202"):
        cluster.gpus[0].with_options(memory_limit_gb=17)
