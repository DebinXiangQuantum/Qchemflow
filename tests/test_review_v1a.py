"""Reviewer tests for V1a, relation certificates (written by Claude; implementers must not edit).

User request (2026-10-14): the agent must check that a designed algorithm is right
symbolically, not only numerically. Several correct inputs do not make an algorithm correct.
Plan: docs/research/07-verified-design.md. The user chose a small independent Python kernel
now, with Lean proofs of the lemma table later (V1d).

Today a result's relation (equivalent, variational upper bound, ...) is a label declared per
refinement and propagated by an algebra. V1a replaces the labels with proofs.

Contract (`chemflow.verify`):
- `LEMMAS` is the lemma table. Each entry has `statement`, `side_conditions` (names) and
  `reference`, all non-empty. It includes at least:
  - exact_diagonalization
  - rayleigh_ritz
  - rayleigh_ritz_full
  - variational_principle
  - sector_decomposition
  - positive_linear_combination
  - linear_combination_exact
  - lehmann_completeness
  - stochastic_estimate
  - no_ordering_law
- `AXIOMS` is the implementation-axiom table: binding contracts, such as "sci_pyscf returns
  the lowest eigenvalue of H projected on its determinant set". Each entry names its
  `binding` and has a `statement`. V1b will falsify them.
- `prove(program, node_id, registry=..., computed=...)` returns a certificate: a
  JSON-serialisable mapping `{"version", "node", "relation", "program_hash", "steps": [...]}`.
  Each step is `{"id", "lemma", "node", "uses", "side_conditions": {name: value},
  "conclusion": {"node", "relation"}}`. `lemma` names a LEMMAS or AXIOMS entry. The last
  step concludes the certificate's relation for its node.
- `check_certificate(certificate, program, registry=..., computed=...)` is the kernel. It
  returns an object with `valid` and `reasons`. It rejects:
  - an unknown lemma or axiom;
  - a program hash that differs;
  - a step whose conclusion does not follow from the lemma and the steps it uses;
  - a side condition that it recomputes from the program and computed artifacts and finds
    false or different from the recorded value.

  The kernel module imports nothing from the prover, and certificates survive a JSON round
  trip.
- Results carry their proof: candidate and control items gain `relation_status`
  ("proved" or "declared") and `relation_certificate` (the SHA-256 of the canonical
  certificate JSON). The certificate is written to `<run dir>/certificates/<id>.json`.
- The critic never escalates a candidate whose relation_status is "declared".
"""
from __future__ import annotations

import ast
import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.runtime import Executor

H4 = {"question": "H4 chain", "target": {
    "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"},
    "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 4, "n_orb": 4},
    "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 100000}, "allowed_effects": ["pure", "simulator"],
    "decision": "d"}


def _executed(spec_dict, active, candidates=()):
    from chemflow.domains import load_registry, pack_for_target
    from chemflow.ops.molecular import molecular_registry

    spec = TaskSpec.model_validate(spec_dict)
    store = TaskStore("reviewer")
    record = store.freeze(spec, "reviewer")
    registry = (molecular_registry() if spec.target.kind == "molecular_ground_energy" else
                load_registry(["molecular", pack_for_target(spec.target.kind).name]))
    executor = Executor(registry)
    tools = make_tools(registry, executor, record, store)
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty, "active": active})["data"]
    plan_id, ids = base["plan_id"], {}
    if candidates:
        built = tools.call("designer", "plan.candidates", {"plan_id": plan_id,
                                                          "candidates": list(candidates)})["data"]
        plan_id, ids = built["plan_id"], built["ids"]
    plan = tools.plans.get(plan_id)
    computed = executor.run_plan(plan, record, store)
    methods = {node.id: node.refinement for node in plan.nodes}
    return plan, computed, registry, base, ids, methods


@pytest.fixture(scope="module")
def h4():
    return _executed(H4, {"n_elec": 4, "n_orb": 4},
                     [{"template": "sci", "select_cutoff": 1e-3},
                      {"template": "sqd", "shots": 2048}])


def _final(certificate):
    return certificate["steps"][-1]


def _lemmas(certificate):
    return {step["lemma"] for step in certificate["steps"]}


def test_v1a_the_lemma_and_axiom_tables_are_complete() -> None:
    from chemflow.verify import AXIOMS, LEMMAS

    required = {"exact_diagonalization", "rayleigh_ritz", "rayleigh_ritz_full",
                "variational_principle", "sector_decomposition", "positive_linear_combination",
                "linear_combination_exact", "lehmann_completeness", "stochastic_estimate",
                "no_ordering_law"}
    assert required <= set(LEMMAS)
    for name, lemma in LEMMAS.items():
        assert lemma["statement"].strip() and lemma["reference"].strip(), name
        assert isinstance(lemma["side_conditions"], (list, tuple)), name
    assert AXIOMS and all(entry["binding"] and entry["statement"].strip()
                          for entry in AXIOMS.values())


def test_v1a_every_ground_energy_method_gets_a_checked_proof(h4) -> None:
    from chemflow.verify import check_certificate, prove

    plan, computed, registry, base, ids, methods = h4
    expected = {"fci": "equivalent", "hf": "variational_upper_bound",
                "ccsd_t": "approximate", "sci": "variational_upper_bound",
                "sqd": "variational_upper_bound"}
    seen = set()
    for node_id in [*base["controls"], *plan.references, *ids.values()]:
        method = methods[node_id]
        if method not in expected:
            continue
        certificate = json.loads(json.dumps(prove(plan, node_id, registry=registry,
                                                  computed=computed)))
        assert certificate["relation"] == expected[method], method
        assert _final(certificate)["conclusion"] == {"node": node_id,
                                                      "relation": expected[method]}
        verdict = check_certificate(certificate, plan, registry=registry, computed=computed)
        assert verdict.valid, (method, verdict.reasons)
        lemmas = _lemmas(certificate)
        if method in ("sci", "sqd"):
            assert {"rayleigh_ritz", "sector_decomposition"} <= lemmas, method
        if method == "fci":
            assert lemmas & {"exact_diagonalization", "rayleigh_ritz_full"}
        if method == "hf":
            assert "variational_principle" in lemmas
        if method == "ccsd_t":
            assert "no_ordering_law" in lemmas
        seen.add(method)
    assert seen == set(expected)


def test_v1a_the_kernel_rejects_forged_certificates(h4) -> None:
    from chemflow.verify import check_certificate, prove

    plan, computed, registry, _, ids, methods = h4
    sqd = next(node for node in ids.values() if methods[node] == "sqd")
    honest = prove(plan, sqd, registry=registry, computed=computed)
    honest = json.loads(json.dumps(honest))

    def rejected(certificate):
        return not check_certificate(certificate, plan, registry=registry,
                                     computed=computed).valid

    # A sampled subspace is not the full sector: "exact" cannot be concluded
    exact = copy.deepcopy(honest)
    exact["relation"] = "equivalent"
    _final(exact)["lemma"] = "rayleigh_ritz_full"
    _final(exact)["conclusion"]["relation"] = "equivalent"
    assert rejected(exact)
    # Unknown lemma
    unknown = copy.deepcopy(honest)
    _final(unknown)["lemma"] = "trust_me"
    assert rejected(unknown)
    # A recorded side condition that the kernel recomputes differently
    tampered = copy.deepcopy(honest)
    step = next(step for step in tampered["steps"] if step["side_conditions"])
    name = next(iter(step["side_conditions"]))
    value = step["side_conditions"][name]
    step["side_conditions"][name] = (not value) if isinstance(value, bool) else "forged"
    assert rejected(tampered)
    # A certificate for another program
    other = copy.deepcopy(honest)
    other["program_hash"] = "0" * 64
    assert rejected(other)
    assert not rejected(honest)


def test_v1a_mixed_sign_reactions_are_not_bounds() -> None:
    from chemflow.eval.runner import load_suite
    from chemflow.verify import check_certificate, prove

    task = next(item for item in load_suite("core").tasks if item.id == "h2-atomization")
    plan, computed, registry, base, _, _ = _executed(task.spec.model_dump(mode="json"),
                                                     {"n_elec": 3, "n_orb": 3})
    nodes = {node.id: node for node in plan.nodes}
    reactions = [node_id for node_id in [*base["controls"], *plan.references]
                 if nodes[node_id].op == "reaction_energy"]

    def species_method(node_id):
        return {nodes[source].refinement for source in nodes[node_id].inputs.values()}

    hf = next(node_id for node_id in reactions if species_method(node_id) == {"hf"})
    fci = next(node_id for node_id in reactions if species_method(node_id) == {"fci"})
    exact = prove(plan, fci, registry=registry, computed=computed)
    assert exact["relation"] == "equivalent"
    assert "linear_combination_exact" in _lemmas(exact)
    bound = prove(plan, hf, registry=registry, computed=computed)
    assert bound["relation"] == "approximate"  # -1 x H2 + 2 x H: upper bounds of mixed sign
    forged = json.loads(json.dumps(bound))
    forged["relation"] = "variational_upper_bound"
    _final(forged)["lemma"] = "positive_linear_combination"
    _final(forged)["conclusion"]["relation"] = "variational_upper_bound"
    assert not check_certificate(forged, plan, registry=registry, computed=computed).valid


def test_v1a_the_kernel_is_independent_of_the_prover() -> None:
    from chemflow.verify import kernel

    tree = ast.parse(Path(kernel.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert not any("prover" in name for name in imported), imported


def test_v1a_the_critic_never_escalates_a_declared_relation() -> None:
    from chemflow.agent.critic import review

    candidate = {"id": "c", "method": "sci", "relation": "variational_upper_bound",
                 "relation_status": "declared", "evidence": True, "gate_passed": True,
                 "error": 1e-6, "cost": {"wall_s": 0.0}, "selection_work": 1}
    control = {"id": "n6", "method": "ccsd_t", "relation": "approximate",
               "relation_status": "proved", "evidence": True, "gate_passed": False,
               "error": 1e-2, "cost": {"wall_s": 1.0}, "selection_work": 10 ** 6}
    assert review([candidate], [control]).decision != "escalate"
    proved = candidate | {"relation_status": "proved"}
    assert review([proved], [control]).decision == "escalate"


def test_v1a_runs_record_their_certificates(tmp_path) -> None:
    from chemflow.eval import runner

    out = tmp_path / "out"
    runner.run(Namespace(suite="core", suite_file=None, tasks="h2-ground", policy="scripted",
                         model=None, roles="designer", jobs=1, out=out))
    entries = [json.loads(line) for line in
               (out / "h2-ground" / "journal.jsonl").read_text().splitlines()]
    critique = next(entry for entry in entries if entry["state"] == "CRITIQUE")
    items = critique["observation"]["candidates"] + critique["observation"]["controls"]
    assert items
    for item in items:
        assert item["relation_status"] == "proved", item["id"]
        path = out / "h2-ground" / "certificates" / f"{item['relation_certificate']}.json"
        certificate = json.loads(path.read_text())
        assert certificate["node"] == item["id"] and certificate["relation"] == item["relation"]
