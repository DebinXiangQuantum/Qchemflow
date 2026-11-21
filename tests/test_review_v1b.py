"""Reviewer tests for V1b, falsification (written by Claude; implementers must not edit).

User request (2026-10-14): "some times several input case is right do not means the algorithm
is totally right." V1a proves a result's relation from mathematical lemmas plus
implementation axioms: binding contracts such as "sci_pyscf returns the lowest eigenvalue of
H projected on its determinant set". Lemmas are mathematics; axioms are code, and code can be
wrong in part of the domain only. V1b searches for counterexamples.

Contract:
- `algorithm.falsify {plan_id, node_id}` is a designer/critic tool. Its schema has exactly
  those two properties.
- It searches the node's design family for counterexamples, against an independent exact
  reference, and is charged to the task budget.
  - The domain is the task's `validity` axes (V1c) when declared, otherwise the family's
    documented default generators.
  - Generators: domain endpoints and other edge cases (symmetric or near-degenerate points
    where the family has them), plus seeded random draws.
  - Each failure is shrunk to the counterexample closest to the task point, within the
    axis `resolution`.
- Properties checked:
  - the node's certified relation: equivalent means |E − E_exact| ≤ 1e-8; a variational
    upper bound means E ≥ E_exact − 1e-9;
  - the implementation axioms cited by its certificate, where an independent check exists.
  The accuracy bar is not a falsification property: V1c maps it.
- Returns `{"verdict": "clean" | "falsified", "instances": [...],
  "generators": [...], "counterexamples": [{"instance", "property", "observed",
  "expected"}], "refuted_axioms": [...], "report_id"}`.
- A falsified relation marks results of that design `relation_status: "refuted"`, and the
  critic never escalates a refuted candidate.
"""
from __future__ import annotations


def _sci(tools, base):
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    return built["plan_id"], next(iter(built["ids"].values()))


def test_v1b_the_critic_never_escalates_a_refuted_relation() -> None:
    from chemflow.agent.critic import review

    candidate = {"id": "c", "method": "sci", "relation": "variational_upper_bound",
                 "relation_status": "refuted", "evidence": True, "gate_passed": True,
                 "error": 1e-6, "cost": {"wall_s": 0.0}, "selection_work": 1}
    control = {"id": "n6", "method": "ccsd_t", "relation": "approximate",
               "relation_status": "proved", "evidence": True, "gate_passed": False,
               "error": 1e-2, "cost": {"wall_s": 1.0}, "selection_work": 10 ** 6}
    assert review([candidate], [control]).decision != "escalate"
