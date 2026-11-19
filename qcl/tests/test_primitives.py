"""The primitive table, the worker implementations and the checker must agree."""

import json

import pytest
from qcl._impls import IMPLEMENTATIONS
from qcl._qcl import GraphBuilder

import qcl

SOURCE = "test_primitives.py:1:1-1:2"
DEVICES = {
    "cpu": {"id": "cpu0", "kind": "cpu"},
    "gpu": {"id": "gpu0", "kind": "gpu", "memory_gb": 32.0},
    "qpu": {"id": "qpu0", "kind": "qpu", "qubits": 20, "site": "local",
            "simulator_method": "statevector", "simulator_host": "cpu0"},
}
# Placements that the table allows but a conditional rule in qcl-check can still refuse,
# depending on inputs: GPU SCF (closed shell, no ERI override), GPU densities (FCI state on
# the same GPU) and QPU programs (kernel width, simulator and site checks).
CONDITIONAL = {("rdm1", "gpu"), ("rdm2", "gpu")} | {
    (op, "gpu") for op in ("scf.rhf", "scf.hf", "scf.rohf", "scf.uhf", "dft.rks", "dft.uks")
} | {(op, "qpu") for op in ("sample", "estimate.quantum", "vqe.sampled")}


def e203(op: str, kind: str) -> list[dict]:
    graph = GraphBuilder()
    node = graph.add_node(op, [], "{}", SOURCE)
    graph.place(node, json.dumps(DEVICES[kind]), SOURCE)
    return [item for item in json.loads(graph.diagnostics_json())
            if item["code"] == "QCL-E203"]


def test_every_primitive_has_an_implementation_and_vice_versa():
    table = qcl.primitives()
    assert sorted(set(table) - set(IMPLEMENTATIONS)) == []
    assert sorted(set(IMPLEMENTATIONS) - set(table)) == []


def test_table_entries_are_well_formed():
    for op, spec in qcl.primitives().items():
        assert spec["kind"] in ("host", "compute", "projection", "copy"), op
        if spec["kind"] == "compute":
            assert spec["devices"], op
            assert set(spec["devices"]) <= {"cpu", "gpu", "qpu"}, op
        else:
            assert "devices" not in spec, op


@pytest.mark.parametrize("op", sorted(op for op, spec in qcl.primitives().items()
                                      if spec["kind"] == "compute"))
def test_checker_follows_the_table(op):
    devices = qcl.primitives()[op]["devices"]
    for kind in DEVICES:
        found = e203(op, kind)
        if kind not in devices:
            assert found, f"{op} on {kind} should be refused"
        elif (op, kind) not in CONDITIONAL:
            assert not found, f"{op} on {kind} should be accepted: {found}"


def test_unavailable_wording_comes_from_the_table():
    [diagnostic] = e203("eigh", "qpu")
    assert diagnostic["message"] == "eigh has no qpu implementation"
    assert diagnostic["hint"] == "Available device kinds: cpu, gpu"
    [diagnostic] = e203("sample", "cpu")
    assert diagnostic["message"] == "sample requires a QPU or QPU simulator"
    [diagnostic] = e203("scf.rhf", "qpu")
    assert diagnostic["hint"] == "HF/KS without an ERI override supports cpu and gpu"


def test_native_ci_engine_stays_cpu_only():
    graph = GraphBuilder()
    node = graph.add_node("fci.solve", [], json.dumps({"engine": "native"}), SOURCE)
    graph.place(node, json.dumps(DEVICES["gpu"]), SOURCE)
    [diagnostic] = [item for item in json.loads(graph.diagnostics_json())
                    if item["code"] == "QCL-E203"]
    assert diagnostic["hint"] == "Native CI is available on CPU only"


def test_add_node_takes_kind_and_type_from_the_table():
    graph = GraphBuilder()
    flag = graph.add_node("boolean.constant", [], json.dumps({"value": True}), SOURCE)
    tensor = graph.add_node("ci.vector", [], "{}", SOURCE)
    chosen = graph.add_node("cond", [flag, tensor, tensor], "{}", SOURCE)
    nodes = json.loads(graph.graph_json())["nodes"]
    assert (nodes[flag]["kind"], nodes[flag]["value_type"]) == ("host", "boolean")
    assert nodes[chosen]["value_type"] == "tensor"
    assert graph.primitive_mismatches() == []
    with pytest.raises(ValueError, match="not declared in primitives.toml"):
        graph.add_node("no.such.op", [], "{}", SOURCE)
    with pytest.raises(TypeError, match="takes its output type from input 1"):
        graph.add_node("cond", [flag], "{}", SOURCE)


def test_compiled_application_graphs_match_the_table():
    @qcl.workflow
    def energy(molecule, active, cpu):
        mean_field = qcl.scf.hf(molecule).to(cpu)
        space = qcl.active_space(mean_field, active).to(cpu)
        return qcl.fci.solve(space.hamiltonian, space.sector).to(cpu).energy

    plan = energy.compile(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g"),
                          qcl.Active(2, 2), qcl.Device("cpu0", "cpu"))
    ops = [node["op"] for node in plan.graph_json()["nodes"]]
    assert {"molecule", "scf.hf", "active_space", "fci.solve", "fci.energy"} <= set(ops)
