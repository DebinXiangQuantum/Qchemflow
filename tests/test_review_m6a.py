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

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.relations import TargetRelation, combine_difference

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
