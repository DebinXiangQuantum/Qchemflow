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

import json
from argparse import Namespace

H4 = {"question": "H4 chain", "target": {
    "operator": "ground_energy", "molecule": {"atoms": "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"},
    "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 4, "n_orb": 4},
    "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 100000}, "allowed_effects": ["pure", "simulator"],
    "decision": "d"}


def _final(certificate):
    return certificate["steps"][-1]


def _lemmas(certificate):
    return {step["lemma"] for step in certificate["steps"]}


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
