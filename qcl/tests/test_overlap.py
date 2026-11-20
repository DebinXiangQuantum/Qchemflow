"""Per-device lanes: overlapped execution gives the sequential result, with real overlap."""

import threading
import time

import pytest
from qcl._worker import _run_lanes

import qcl

CPU_SIMULATOR = qcl.Device("sim0", "qpu", qubits=20, simulator_host="cpu0",
                           simulator_method="statevector")
SECOND_SIMULATOR = qcl.Device("sim1", "qpu", qubits=20, simulator_host="cpu0",
                             simulator_method="statevector")


def node(identifier, device, inputs=(), guards=()):
    return {"id": identifier, "device": None if device is None else {"id": device},
            "inputs": list(inputs), "guards": list(guards)}


def test_lanes_overlap_devices_and_serialize_each_device():
    nodes = [node(0, "qpu"), node(1, "cpu"), node(2, "cpu"), node(3, "gpu", [0, 1]),
             node(4, None, [2, 3])]
    spans, lock = {}, threading.Lock()

    def run(item):
        started = time.perf_counter()
        time.sleep(0.2 if item["id"] == 0 else 0.05)
        with lock:
            spans[item["id"]] = (started, time.perf_counter(), threading.current_thread().name)

    _run_lanes(nodes, run)
    assert set(spans) == {0, 1, 2, 3, 4}
    # The QPU wait and both CPU nodes run at the same time.
    assert spans[1][0] < spans[0][1] and spans[2][0] < spans[0][1]
    # One device never runs two nodes at once.
    assert spans[1][1] <= spans[2][0] or spans[2][1] <= spans[1][0]
    # A node starts only after all of its inputs finished.
    assert spans[3][0] >= max(spans[0][1], spans[1][1])
    assert spans[4][0] >= max(spans[2][1], spans[3][1])
    assert spans[0][2].startswith("qcl-qpu") and spans[4][2].startswith("qcl-host")


def test_guards_are_dependencies():
    order = []
    nodes = [node(0, "cpu"), node(1, "gpu", guards=[{"condition": 0, "expected": True}])]

    def run(item):
        if item["id"] == 0:
            time.sleep(0.05)
        order.append(item["id"])

    _run_lanes(nodes, run)
    assert order == [0, 1]


def test_a_failure_stops_new_work_and_reports_the_first_failing_node():
    started = []
    nodes = [node(0, "cpu"), node(1, "gpu"), node(2, "cpu", [0]), node(3, "gpu", [1]),
             node(4, "qpu", [2, 3])]

    def run(item):
        started.append(item["id"])
        if item["id"] in (2, 3):
            raise ValueError(f"node {item['id']}")

    with pytest.raises(ValueError, match="node 2"):
        _run_lanes(nodes, run)
    assert 4 not in started


@qcl.workflow
def two_lanes(cpu, qpu, other):
    molecule = qcl.Molecule("H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7", "sto-3g")
    mean_field = qcl.scf.rhf(molecule).to(cpu)
    space = qcl.active_space(mean_field, qcl.Active(4, 4)).to(cpu)
    kernel = qcl.kernels.lucj(space, params=qcl.ccsd.amplitudes(space).to(cpu))
    first = qcl.sample(kernel, shots=200, seed=3).to(qpu)
    second = qcl.sample(kernel, shots=300, seed=4).to(other)
    return {key: qcl.sqd.diagonalize(space.hamiltonian, bits.to(cpu), space.sector).to(cpu)
            for key, bits in (("first", first), ("second", second))}


def test_overlapped_run_matches_sequential_and_reports_spans(tmp_path, monkeypatch):
    policy = tmp_path / "policy.toml"
    policy.write_text("[budget]\nqpu_shots = 1000\n")
    monkeypatch.setenv("QCL_POLICY_FILE", str(policy))
    cpu = qcl.Cluster.local().cpus[0]
    plan = two_lanes.compile(cpu, CPU_SIMULATOR, SECOND_SIMULATOR)
    sequential = plan.run()
    overlapped = plan.run(overlap=True)
    for key in ("first", "second"):
        assert overlapped[key].value == pytest.approx(sequential[key].value, abs=1e-10)
    profile = plan.profile()
    assert [entry["id"] for entry in profile] == sorted(entry["id"] for entry in profile)
    assert {entry["device"] for entry in profile} >= {"cpu0", "sim0", "sim1"}
    assert all(0 <= entry["start_s"] <= entry["end_s"] for entry in profile)
    with pytest.raises(TypeError, match="overlap"):
        plan.run(overlap="yes")
