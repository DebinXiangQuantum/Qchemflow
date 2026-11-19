"""The N2 dissociation application compiles to a GPU/CPU/QPU plan with explicit transfers."""

import importlib.util
from pathlib import Path

import pytest

import qcl

APP = Path(__file__).resolve().parents[1] / "apps" / "n2_dissociation.py"
CPU = qcl.Device("cpu0", "cpu", threads=8)
GPUS = [qcl.Device(f"gpu{index}", "gpu", memory_gb=32.0) for index in range(2)]
QPUS = [qcl.Device(f"sim{index}", "qpu", qubits=20, simulator_host=f"gpu{index}",
                   simulator_method="statevector") for index in range(2)]


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("n2_dissociation", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def placements(plan):
    return [(node["op"], node["device"]["id"])
            for node in plan.graph_json()["nodes"] if node["kind"] == "compute"]


def test_each_bond_length_runs_on_its_lane(app):
    plan = app.n2_curve.compile((1.1, 1.6, 2.2), CPU, list(zip(GPUS, QPUS, strict=True)),
                                strict_transfers=True)
    found = placements(plan)
    per_point = [("scf.rhf", "gpu{}"), ("active_space", "cpu0"),
                 ("ccsd.amplitudes", "cpu0"), ("sample", "sim{}"),
                 ("sqd.diagonalize", "cpu0"), ("fci.solve", "gpu{}")]
    expected = [(op, device.format(lane)) for lane in (0, 1, 0) for op, device in per_point]
    assert found == expected
    transfers = plan.explain(format="json")["transfers"]
    assert {transfer["kind"] for transfer in transfers} == {"explicit"}
    assert {(transfer["source_device"], transfer["to"]) for transfer in transfers} == {
        ("gpu0", "cpu0"), ("gpu1", "cpu0"), ("sim0", "cpu0"), ("sim1", "cpu0"),
        ("cpu0", "gpu0"), ("cpu0", "gpu1")}
    assert plan.estimate()["shots"] == 3 * 4000


def test_lanes_pair_every_gpu_with_a_qpu(app):
    cluster = qcl.Cluster(cpus=(CPU,), gpus=tuple(GPUS), qpus=tuple(QPUS[:1]))
    assert [(gpu.id, qpu.id) for gpu, qpu in app.lanes_from(cluster)] == [
        ("gpu0", "sim0"), ("gpu1", "sim0")]
    with pytest.raises(SystemExit):
        app.lanes_from(qcl.Cluster(cpus=(CPU,), gpus=(), qpus=tuple(QPUS)))


def test_methods_are_not_mixed_inside_the_graph(app):
    @qcl.workflow
    def mixed(molecule, cpu, gpu, qpu):
        space = qcl.active_space(qcl.scf.rhf(molecule).to(cpu), qcl.Active(2, 2)).to(cpu)
        bits = qcl.sample(qcl.kernels.hf(space), shots=100).to(qpu)
        sqd = qcl.sqd.diagonalize(space.hamiltonian, bits.to(cpu), space.sector).to(cpu)
        casci = qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy
        return sqd - casci

    with pytest.raises(qcl.CompileError, match="QCL-E103"):
        mixed.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"), CPU, GPUS[0], QPUS[0])
