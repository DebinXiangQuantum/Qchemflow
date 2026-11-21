"""Reviewer regression tests for R125 (written by Claude; implementers must not edit).

Found in the final-code core benchmark (2026-10-14, 201651f). glm-5.3 was correct on 11 of 11
tasks, but its decision was outside the task's `decision_in` on 3. One was h2-atomization,
where GLM's SCI reaction candidate was escalated over CCSD(T). The critique showed why: the
candidate's cost was all zeros. GLM had screen-run its plan during its own turn, so the
orchestrator's screening run hit the executor cache. A cache hit records a zero CostRecord,
which is right for consumption, and `_affordable` compared those zeros. No classical control
was "no more expensive" than a free candidate.

Two problems lie underneath:
- The scripted critic (R105) decides on a deterministic basis: `selection_work` plus shot
  counts. The LLM design path calls `review` on measured costs. Those are noisy at these sizes,
  and zero for cached nodes. The same plan can therefore get different decisions depending on
  the policy and on timing.
- `selection_work` is defined only for ground_energy nodes, so a composite result is never
  dominated. Composite results include reaction energies built from species ground energies.

Contract:
- Every critic path decides on the same deterministic basis as the scripted critic: the
  scripted loop, the LLM design path, and the verdict kept when the critic is an LLM. Measured
  and cached costs stay in records and reports, but never decide.
- The `selection_work` of a node that is not itself a solver node is the sum over the
  ground_energy solver nodes in its dependency closure. It is None if any of them has no
  estimate, or if there are none.
"""
from __future__ import annotations

from test_review_r114_118 import _eval


def _critique(entries):
    return next(entry for entry in entries if entry["state"] == "CRITIQUE")


def test_r125_composite_results_sum_their_solver_work(tmp_path, monkeypatch) -> None:
    _, _, journal, _ = _eval(tmp_path, monkeypatch, "h2-atomization", policy="scripted")
    controls = {item["method"]: item for item in
                _critique(journal("h2-atomization"))["observation"]["controls"]}
    # H2 (2 orbitals) plus H (1 orbital), with the ground_energy power laws n^3 and n^7
    assert controls["hf"]["selection_work"] == 2 ** 3 + 1 ** 3
    assert controls["ccsd_t"]["selection_work"] == 2 ** 7 + 1 ** 7
