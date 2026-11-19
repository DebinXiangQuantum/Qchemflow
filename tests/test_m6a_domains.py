import itertools
import json
import sys

import numpy as np
import pytest

from chemflow import build
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.check import check
from chemflow.diagnostics import AdapterError, CheckError, Diagnostic
from chemflow.domains import get_pack, load_registry
from chemflow.domains.benchmarks import _task, run_benchmark
from chemflow.domains.materials import lattice_arrays
from chemflow.ir.schema import Node
from chemflow.ops.molecular import molecular_registry
from chemflow.relations import TargetRelation, combine_difference, combine_linear
from chemflow.runtime import Executor
from chemflow.units import Q


def freeze(spec):
    store = TaskStore("m6a-tests")
    return store.freeze(spec, "test"), store


def test_pack_isolation_and_check_plugin():
    molecular = load_registry(["molecular"])
    materials = load_registry(["materials"])
    assert molecular.implementations.keys() == molecular_registry().implementations.keys()
    assert "lattice_model" not in molecular.implementations
    assert "reaction_energy" not in materials.implementations
    molecular.checks.append(lambda program, registry: [Diagnostic(code="CF-TEST", message="plugin")])
    with build.program("empty") as builder:
        pass
    assert {issue.code for issue in check(builder.to_ir(), molecular)} == {"CF-TEST"}
    assert not check(builder.to_ir(), load_registry(["molecular"]))
    with pytest.raises(ValueError):
        load_registry(["unknown"])


def test_discriminated_targets_and_legacy():
    spec = _task("materials", "h6_ewf_full_bath").model_dump(mode="json")
    del spec["target"]["kind"]
    assert TaskSpec.model_validate(spec).target.kind == "molecular_ground_energy"
    spec["target"]["kind"] = "unknown"
    with pytest.raises(ValueError):
        TaskSpec.model_validate(spec)


def test_linear_relation_table():
    for left, right in itertools.product(TargetRelation, repeat=2):
        assert combine_linear([(1, left), (-1, right)]) == combine_difference(left, right)
    exact, upper = TargetRelation.equivalent, TargetRelation.variational_upper_bound
    assert combine_linear([(-3, exact), (2, upper), (1, upper)]) == upper
    assert combine_linear([(0, upper)]) == TargetRelation.approximate


def test_reaction_charge_and_model_policy_validation():
    spec = _task("reactions", "h2_atomization").model_dump(mode="json")
    spec["target"]["species"][1]["molecule"]["charge"] = 1
    with pytest.raises(ValueError, match="CF-E012"):
        TaskSpec.model_validate(spec)
    record, _ = freeze(_task("reactions", "h2_atomization"))
    plan, _ = get_pack("reactions").base_plan(record)
    hamiltonians = [node for node in plan.nodes if node.op == "hamiltonian"]
    hamiltonians[0].params["active"]["orbital_source"] = "natural"
    assert "CF-E011" in {issue.code for issue in check(plan, load_registry(["reactions"]))}


@pytest.mark.parametrize("imaginary", [0, 1])
def test_minimum_frequencies_when_present(imaginary):
    record, _ = freeze(_task("reactions", "h2_atomization"))
    plan, _ = get_pack("reactions").base_plan(record)
    molecule = next(node for node in plan.nodes if node.op == "molecule")
    model = next(node for node in plan.nodes if node.op == "electronic_model")
    plan.nodes.append(Node(id="frequencies", op="frequencies",
                           inputs={"molecule": molecule.id, "model": model.id}))
    registry = load_registry(["reactions"])
    registry.implementations["frequencies"].adapter = lambda **kwargs: {
        "n_imaginary": imaginary, "max_gradient": 0.01, "freq_wavenumber": [100.]}
    if imaginary:
        with pytest.raises(AdapterError, match="CF-E021"):
            Executor(registry).run(plan)
    else:
        Executor(registry).run(plan)


def test_square_lattice_integrals_and_invalid_geometry():
    h1, eri = lattice_arrays(geometry="square", shape=[2, 3], t=2, U=5, unit="eV")
    assert h1.shape == (6, 6)
    assert np.count_nonzero(h1) == 14
    assert h1[0, 3] == pytest.approx(-Q(2, "eV").to("hartree").value)
    assert h1[2, 3] == 0
    assert eri[4, 4, 4, 4] == pytest.approx(Q(5, "eV").to("hartree").value)
    assert np.count_nonzero(eri) == 6
    for params in ({"geometry": "square", "shape": [2, 2], "n_sites": 3},
                   {"geometry": "ring", "n_sites": 0}, {"geometry": "triangle"}):
        with pytest.raises(ValueError):
            lattice_arrays(**params)


def test_lattice_non_half_filling_and_candidates():
    spec = _task("materials", "hubbard_dimer").model_dump(mode="json")
    spec["target"]["sector"] = {"n_orb": 2, "n_alpha": 1, "n_beta": 0}
    spec["target"]["target_spin"] = 0.5
    record, store = freeze(TaskSpec.model_validate(spec))
    plan, _ = get_pack("materials").base_plan(record)
    result = Executor(load_registry(["materials"])).run_plan(plan, record, store)
    assert result[plan.references[0]].value.to("eV").value == pytest.approx(-1, abs=1e-9)
    expanded, identifiers = Orchestrator._candidate_plan(plan, [
        {"template": "sci", "select_cutoff": 1e-3}])
    assert identifiers and not check(expanded, load_registry(["materials"]))


def test_domain_active_arguments_and_reaction_candidates():
    for pack, name in (("materials", "hubbard_dimer"), ("reactions", "h2_atomization")):
        record, store = freeze(_task(pack, name))
        with pytest.raises(ValueError, match="[Aa]ctive"):
            get_pack(pack).base_plan(record, {"n_orb": 100, "n_elec": 2})
        if pack == "reactions":
            plan, _ = get_pack(pack).base_plan(record)
            expanded, identifiers = Orchestrator._candidate_plan(plan, [
                {"template": "sci", "select_cutoff": 1e-3}])
            result = Executor(load_registry([pack])).run_plan(expanded, record, store)
            assert result[next(iter(identifiers.values()))].kind == "reaction_energy"


@pytest.mark.parametrize("pack,name", [("reactions", "h2_atomization"),
                                       ("materials", "hubbard_dimer")])
def test_orchestrator_domain_dispatch(tmp_path, pack, name):
    class Designer:
        def decide(self, observation):
            return {"candidates": [{"template": "sci", "select_cutoff": 1e-3}]}

    store = TaskStore("orchestrator-test")
    orchestrator = Orchestrator(store, lambda spec: store.freeze(spec, "test"), tmp_path,
                                max_rounds=1, policies={"designer": Designer()})
    result = orchestrator.run(_task(pack, name))
    assert result.results and (tmp_path / "report.md").exists()
    if pack == "materials":
        assert "Hubbard site basis" in result.report


@pytest.mark.parametrize("solver", ["fci", "ccsd", "sqd"])
def test_partial_embedding_reports_approximate_finite_error(solver):
    with build.program("partial-bath") as builder:
        mol = builder.molecule("H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6")
        model = builder.electronic_model(basis="sto-3g")
        ham = builder.hamiltonian(mol, model, active={"n_orb": 4, "n_elec": 4})
        sector = builder.sector(n_orb=4, n_alpha=2, n_beta=2)
        reference = builder.ground_energy(ham, sector, s=0)
        builder.refine(reference, "fci")
        builder.bind(reference, "fci_pyscf")
        embedded = builder.ground_energy(ham, sector, s=0)
        builder.refine(embedded, "ewf", bath="dmet", solver=solver, shots=128)
        builder.bind(embedded, "ewf_embedding")
    result = Executor(load_registry(["materials"])).run(builder.to_ir())
    estimate = result[embedded.id]
    error = abs(estimate.value.value - result[reference.id].value.value)
    assert np.isfinite(error) and error > 1e-6
    assert estimate.relation == "approximate"
    assert estimate.diagnostics["n_fragments"] == 4
    if solver == "sqd":
        assert estimate.cost.sim_shots == 512
        builder.ir.budget = {"sim_shots": 0}
        executor = Executor(load_registry(["materials"]))
        with pytest.raises(CheckError, match="CF-E070"):
            executor.run(builder.to_ir())
        assert not executor.cache and not executor.store.memory


@pytest.mark.parametrize("overrides", [{"lattice": [[1, 0, 0]]}, {"pseudo": None}])
def test_periodic_invalid_systems(overrides):
    params = {"lattice": np.eye(3).tolist(), "atoms": "He 0 0 0", "basis": "gth-szv",
              "pseudo": "gth-pade", **overrides}
    with build.program("invalid-pbc") as builder:
        builder.node("periodic_system", **params)
    with pytest.raises(CheckError, match="CF-E080"):
        Executor(load_registry(["materials"])).run(builder.to_ir())


def test_benchmark_cli_and_independent_oracle(tmp_path, monkeypatch, capsys):
    from chemflow.cli import main

    monkeypatch.setattr(sys, "argv", ["chemflow", "bench", "list"])
    main()
    assert "materials hubbard_dimer" in capsys.readouterr().out
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(sys, "argv", ["chemflow", "bench", "run", "materials", "hubbard_dimer",
                                     "--json", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["passed"] and report["task_hash"]
    assert "Analytic" in report["oracle"]["source"]
    with pytest.raises(ValueError):
        run_benchmark("materials", "unknown")


def test_mcp_packs_argument(monkeypatch):
    from chemflow import mcp

    calls = []

    class Server:
        def run(self, transport):
            calls.append(transport)

    def create_server(*args, **kwargs):
        calls.append(kwargs["packs"])
        return Server()

    monkeypatch.setattr(mcp, "create_server", create_server)
    monkeypatch.setattr(sys, "argv", ["chemflow-mcp", "--profile", "full", "--task", "task.json",
                                     "--public-key", "key.pem", "--packs", "reactions,materials"])
    mcp.main()
    assert calls == ["reactions,materials", "stdio"]
