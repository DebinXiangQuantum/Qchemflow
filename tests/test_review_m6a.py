"""Reviewer regression tests for M6a (written by Claude; implementers must not edit).

Domain-pack core + reactions + materials. Oracles are computed here, independently of the
ChemFlow code path (direct PySCF calls or a small exact diagonalisation written in this file).
Contract (M6a brief): `chemflow.domains.{pack_names, load_registry, get_pack}`,
`chemflow.domains.benchmarks.{list_benchmarks, run_benchmark}`,
`chemflow.relations.combine_linear`, task target kinds `reaction_energy` and
`lattice_ground_energy`, operators `reaction_energy`, `geometry_optimize`, `frequencies`,
`lattice_model`, `periodic_system`, `periodic_scf`, refinement `ewf` (binding `ewf_embedding`).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from chemflow import build
from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.ir.schema import Node
from chemflow.relations import TargetRelation, combine_difference
from chemflow.runtime import Executor

GATE = [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}]
H2 = {"name": "H2", "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"}, "coefficient": -1,
      "active": {"n_elec": 2, "n_orb": 2}, "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
      "target_spin": 0}
H_ATOM = {"name": "H", "molecule": {"atoms": "H 0 0 0", "spin": 1}, "coefficient": 2,
          "active": {"n_elec": 1, "n_orb": 1}, "sector": {"n_alpha": 1, "n_beta": 0, "n_orb": 1},
          "target_spin": 0.5}
ATOMIZATION = {"question": "H2 atomization energy", "target": {
    "kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
    "species": [H2, H_ATOM]}, "gates": GATE, "budget": {"wall_s": 600},
    "allowed_effects": ["pure"], "decision": "d"}
RING6 = {"question": "6-site Hubbard ring", "target": {
    "kind": "lattice_ground_energy",
    "lattice": {"model": "hubbard", "geometry": "ring", "n_sites": 6, "t": 1.0, "U": 4.0,
                "unit": "eV"},
    "sector": {"n_alpha": 3, "n_beta": 3, "n_orb": 6}, "target_spin": 0},
    "gates": GATE, "budget": {"wall_s": 600}, "allowed_effects": ["pure"], "decision": "d"}


def _freeze(spec: dict):
    store = TaskStore("reviewer")
    return store.freeze(TaskSpec.model_validate(spec), "reviewer"), store


def _codes(error) -> set[str]:
    return {item.code for item in error.diagnostics}


# --- core -----------------------------------------------------------------------------------

def test_packs_are_named_opt_in_and_the_molecular_core_is_unchanged() -> None:
    from chemflow.domains import load_registry, pack_names
    from chemflow.ops.molecular import molecular_registry

    assert {"molecular", "reactions", "materials"} <= set(pack_names())
    core = load_registry(["molecular"])
    assert not any("reaction_energy" in key for key in core.operators)
    assert set(core.implementations) == set(molecular_registry().implementations)
    extended = load_registry(["molecular", "reactions", "materials"])
    assert any("reaction_energy" in key for key in extended.operators)
    assert any("lattice_model" in key for key in extended.operators)
    with pytest.raises((KeyError, ValueError)):
        load_registry(["molecular", "no-such-pack"])


def test_legacy_molecular_tasks_still_validate_as_the_default_kind() -> None:
    spec = TaskSpec.model_validate({
        "question": "q", "target": {"molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
                                    "electronic_model": {"basis": "sto-3g"},
                                    "active": {"n_elec": 2, "n_orb": 2},
                                    "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
                                    "target_spin": 0},
        "gates": GATE, "decision": "d"})
    assert spec.target.kind == "molecular_ground_energy"


def test_linear_relation_algebra_matches_pairwise_differences() -> None:
    from chemflow.relations import combine_linear

    for left, right in itertools.product(TargetRelation, repeat=2):
        assert combine_linear([(1.0, left), (-1.0, right)]) == combine_difference(left, right)
    exact, upper = TargetRelation.equivalent, TargetRelation.variational_upper_bound
    assert combine_linear([(-1, exact), (2, exact)]) == exact
    assert combine_linear([(-1, exact), (2, upper)]) == upper  # bounds only on products
    assert combine_linear([(-1, upper), (2, exact)]) == TargetRelation.approximate
    assert combine_linear([(-1, TargetRelation.model_change), (2, exact)]) == \
        TargetRelation.model_change


# --- reactions ------------------------------------------------------------------------------

def _pyscf_atomization() -> float:
    from pyscf import fci, gto, scf

    h2 = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)).run()
    e_h2 = fci.FCI(h2).kernel()[0]
    e_h = scf.UHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)).run().e_tot
    return 2 * e_h - e_h2


def test_reaction_energy_task_end_to_end_against_independent_pyscf() -> None:
    from chemflow.domains import get_pack, load_registry

    registry = load_registry(["molecular", "reactions"])
    record, store = _freeze(ATOMIZATION)
    plan, controls = get_pack("reactions").base_plan(record)
    result = Executor(registry).run_plan(plan, record, store)
    reference = result[plan.references[0]]
    assert reference.kind == "reaction_energy" and reference.relation == "equivalent"
    assert reference.value.to("hartree").value == pytest.approx(_pyscf_atomization(), abs=1e-8)
    assert controls and all(result[node].kind == "reaction_energy" for node in controls)
    hf = next(node for node in controls if "hf" in {item.refinement for item in plan.nodes
                                                    if item.id in _ancestors(plan, node)})
    assert result[hf].relation == "approximate"  # reactant HF bound becomes a lower bound
    assert result["gates"] and all(gate.passed for gate in result["gates"]
                                   if gate.node == plan.references[0])


def _ancestors(plan, node_id: str) -> set[str]:
    nodes = {node.id: node for node in plan.nodes}
    seen, pending = set(), [node_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(nodes[current].inputs.values())
    return seen


def test_unbalanced_reactions_are_refused_at_task_and_plan_level() -> None:
    from chemflow.check import check
    from chemflow.domains import get_pack, load_registry

    unbalanced = {**ATOMIZATION, "target": {**ATOMIZATION["target"], "species": [
        H2, {**H_ATOM, "coefficient": 1}]}}
    with pytest.raises(ValueError):
        TaskSpec.model_validate(unbalanced)
    record, _ = _freeze(ATOMIZATION)
    plan, _ = get_pack("reactions").base_plan(record)
    node = next(item for item in plan.nodes if item.id == plan.references[0])
    coefficients = node.params["coefficients"]
    port = next(key for key, value in coefficients.items() if value == 2)
    coefficients[port] = 1
    assert "CF-E012" in {item.code for item in
                         check(plan, load_registry(["molecular", "reactions"]))}


def test_species_from_different_models_cannot_be_combined() -> None:
    from chemflow.domains import get_pack, load_registry

    record, _ = _freeze(ATOMIZATION)
    plan, _ = get_pack("reactions").base_plan(record)
    molecules = {node.id: node for node in plan.nodes if node.op == "molecule"}
    atom = next(node_id for node_id, node in molecules.items()
                if node.params["atoms"].strip() == "H 0 0 0")
    plan.nodes.append(Node(id="other_model", op="electronic_model", params={"basis": "6-31g"}))
    for node in plan.nodes:
        if node.op == "hamiltonian" and node.inputs.get("molecule") == atom:
            node.inputs["model"] = "other_model"
    with pytest.raises((CheckError, AdapterError)) as info:
        Executor(load_registry(["molecular", "reactions"])).run(plan)
    assert "CF-E011" in _codes(info.value)


BARRIER = {"question": "H + H2 exchange barrier", "target": {
    "kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
    "species": [
        {**H2, "coefficient": -1},
        {**H_ATOM, "coefficient": -1},
        {"name": "H3_TS", "role": "ts", "coefficient": 1,
         "molecule": {"atoms": "H 0 0 -0.93; H 0 0 0; H 0 0 0.93", "spin": 1},
         "active": {"n_elec": 3, "n_orb": 3}, "sector": {"n_alpha": 2, "n_beta": 1, "n_orb": 3},
         "target_spin": 0.5}]},
    "gates": GATE, "budget": {"wall_s": 1200}, "allowed_effects": ["pure"], "decision": "d"}


def test_barrier_uses_a_verified_transition_state_and_refuses_an_unverified_one() -> None:
    from pyscf import fci, gto, scf

    from chemflow.domains import get_pack, load_registry

    registry = load_registry(["molecular", "reactions"])
    record, store = _freeze(BARRIER)
    plan, _ = get_pack("reactions").base_plan(record)
    assert any(node.op == "geometry_optimize" for node in plan.nodes)
    assert any(node.op == "frequencies" for node in plan.nodes)
    result = Executor(registry).run_plan(plan, record, store)
    frequencies = next(result[node.id] for node in plan.nodes if node.op == "frequencies")
    assert frequencies["n_imaginary"] == 1 and frequencies["max_gradient"] < 1e-4
    optimized = next(result[node.id] for node in plan.nodes if node.op == "geometry_optimize")
    ts = scf.UHF(gto.M(atom=optimized["atoms"], basis="sto-3g", spin=1,
                       unit=optimized.get("unit", "angstrom"), verbose=0)).run()
    e_ts = fci.FCI(ts).kernel()[0]
    e_h2 = fci.FCI(scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                                  verbose=0)).run()).kernel()[0]
    e_h = scf.UHF(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)).run().e_tot
    reference = result[plan.references[0]].value.to("hartree").value
    assert reference == pytest.approx(e_ts - e_h2 - e_h, abs=1e-7) and reference > 0
    # Remove the optimisation: the TS species now sits on the unverified guess geometry.
    optimizer = next(node for node in plan.nodes if node.op == "geometry_optimize")
    guess = optimizer.inputs["molecule"]
    for node in plan.nodes:
        for port, source in list(node.inputs.items()):
            if source == optimizer.id:
                node.inputs[port] = guess
    plan.nodes.remove(optimizer)
    plan.gates.clear()
    with pytest.raises((AdapterError, CheckError)) as info:
        Executor(registry).run(plan)
    assert "CF-E021" in _codes(info.value)


# --- materials ------------------------------------------------------------------------------

def _hubbard_ed(n_sites: int, t: float, u: float, n_up: int, n_down: int, ring: bool) -> float:
    """Independent exact diagonalisation (bit strings, explicit fermionic signs)."""
    bonds = [(i, i + 1) for i in range(n_sites - 1)] + ([(n_sites - 1, 0)] if ring else [])
    ups = [sum(1 << i for i in c) for c in itertools.combinations(range(n_sites), n_up)]
    downs = [sum(1 << i for i in c) for c in itertools.combinations(range(n_sites), n_down)]
    basis = [(a, b) for a in ups for b in downs]
    index = {state: k for k, state in enumerate(basis)}
    h = np.zeros((len(basis), len(basis)))

    def hop(bits: int, i: int, j: int):
        if not bits >> j & 1 or bits >> i & 1:
            return None
        low, high = min(i, j), max(i, j)
        between = (bits & (((1 << high) - 1) ^ ((1 << (low + 1)) - 1))).bit_count()
        return bits ^ (1 << i) ^ (1 << j), (-1) ** between

    for k, (a, b) in enumerate(basis):
        h[k, k] = u * (a & b).bit_count()
        for i, j in bonds:
            for p, q in ((i, j), (j, i)):
                moved = hop(a, p, q)
                if moved:
                    h[index[(moved[0], b)], k] += -t * moved[1]
                moved = hop(b, p, q)
                if moved:
                    h[index[(a, moved[0])], k] += -t * moved[1]
    return float(np.linalg.eigvalsh(h)[0])


def _lattice_energy(n_sites: int, geometry: str, t: float, u: float, n_up: int, n_down: int):
    from chemflow.domains import load_registry

    with build.program("lattice") as builder:
        lattice = builder.node("lattice_model", model="hubbard", geometry=geometry,
                               n_sites=n_sites, t=t, U=u, unit="eV")
        sector = builder.sector(n_alpha=n_up, n_beta=n_down, n_orb=n_sites)
        energy = builder.ground_energy(lattice, sector)
        builder.refine(energy, "fci")
        builder.bind(energy, "fci_pyscf")
    result = Executor(load_registry(["molecular", "materials"])).run(builder.to_ir())
    return result[energy.id]


def test_hubbard_dimer_matches_the_analytic_ground_state() -> None:
    estimate = _lattice_energy(2, "chain", 1.0, 4.0, 1, 1)
    assert estimate.value.to("eV").value == pytest.approx((4 - np.sqrt(16 + 16)) / 2, abs=1e-9)
    assert estimate.relation == "equivalent"


@pytest.mark.parametrize("geometry,n_sites,n_up,n_down", [("ring", 6, 3, 3), ("chain", 5, 3, 2)])
def test_hubbard_lattices_match_an_independent_exact_diagonalisation(
        geometry, n_sites, n_up, n_down) -> None:
    estimate = _lattice_energy(n_sites, geometry, 1.0, 4.0, n_up, n_down)
    oracle = _hubbard_ed(n_sites, 1.0, 4.0, n_up, n_down, ring=geometry == "ring")
    assert estimate.value.to("eV").value == pytest.approx(oracle, abs=1e-8)


def test_lattice_task_identity_and_gates() -> None:
    from chemflow.domains import get_pack, load_registry

    registry = load_registry(["molecular", "materials"])
    record, store = _freeze(RING6)
    plan, controls = get_pack("materials").base_plan(record)
    result = Executor(registry).run_plan(plan, record, store)
    reference = result[plan.references[0]]
    oracle = _hubbard_ed(6, 1.0, 4.0, 3, 3, ring=True)
    assert reference.value.to("eV").value == pytest.approx(oracle, abs=1e-8)
    assert controls and all(gate.passed for gate in result["gates"]
                            if gate.node == plan.references[0])
    lattice = next(node for node in plan.nodes if node.op == "lattice_model")
    lattice.params["U"] = 5.0
    with pytest.raises(AdapterError) as info:
        Executor(registry).run_plan(plan, record, store)
    assert "CF-E120" in _codes(info.value)


def test_lattice_tasks_work_through_the_agent_tools() -> None:
    from chemflow.agent.tools import make_tools
    from chemflow.domains import load_registry

    registry = load_registry(["molecular", "materials"])
    record, store = _freeze(RING6)
    tools = make_tools(registry, Executor(registry), record, store)
    plan_id = tools.call("modeler", "plan.new", {"identifier": "ring"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": plan_id,
                                                "active": {"n_elec": 6, "n_orb": 6}})["data"]
    out = tools.call("designer", "run", {"plan_id": base["plan_id"], "scale": "screen"})["data"]
    assert out["gates"] and any(gate["passed"] for gate in out["gates"])


def test_complete_bath_embedding_reproduces_full_fci() -> None:
    from pyscf import fci, gto, scf

    from chemflow.domains import load_registry

    atoms = "; ".join(f"H 0 0 {1.0 * index}" for index in range(6))
    with build.program("ewf") as builder:
        model = builder.electronic_model(basis="sto-3g")
        hamiltonian = builder.hamiltonian(builder.molecule(atoms), model,
                                          active={"n_elec": 6, "n_orb": 6})
        sector = builder.sector(n_alpha=3, n_beta=3, n_orb=6)
        embedded = builder.ground_energy(hamiltonian, sector, s=0)
        builder.refine(embedded, "ewf", fragments="atoms", bath="full", solver="fci")
        builder.bind(embedded, "ewf_embedding")
    estimate = Executor(load_registry(["molecular", "materials"])).run(builder.to_ir())[
        embedded.id]
    full = fci.FCI(scf.RHF(gto.M(atom=atoms, basis="sto-3g", verbose=0)).run()).kernel()[0]
    assert estimate.value.to("hartree").value == pytest.approx(full, abs=1e-6)
    assert estimate.relation == "approximate"
    assert estimate.diagnostics["n_fragments"] == 6


def test_kpoint_and_supercell_periodic_hf_agree_per_cell() -> None:
    from chemflow.domains import load_registry

    lattice = [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]]
    values = {}
    for label, params in (("kmesh", {"kmesh": [1, 1, 2]}), ("supercell", {"supercell": [1, 1, 2]})):
        with build.program("pbc") as builder:
            system = builder.node("periodic_system", lattice=lattice, unit="angstrom",
                                  atoms="C 0 0 0; C 0.8917 0.8917 0.8917",
                                  basis="gth-szv", pseudo="gth-pade")
            energy = builder.node("periodic_scf", system=system, method="rhf",
                                  exxdiv="ewald", **params)
        result = Executor(load_registry(["molecular", "materials"])).run(builder.to_ir())
        values[label] = result[energy.id].value.to("hartree").value
    assert values["kmesh"] == pytest.approx(values["supercell"], abs=1e-7)
    assert values["kmesh"] == pytest.approx(-10.5309051677, abs=1e-6)  # PySCF KRHF, reviewer


def test_benchmarks_are_listed_and_pass_against_their_oracles() -> None:
    from chemflow.domains.benchmarks import list_benchmarks, run_benchmark

    names = set(list_benchmarks())
    assert {("reactions", "h2_atomization"), ("reactions", "h3_exchange_barrier"),
            ("materials", "hubbard_dimer"), ("materials", "hubbard_ring6"),
            ("materials", "h6_ewf_full_bath"), ("materials", "diamond_kpoint_supercell")} <= names
    report = run_benchmark("materials", "hubbard_dimer")
    assert report["passed"] is True and report["oracle"]["source"]
    assert "chemflow" not in report["oracle"]["source"].lower()
