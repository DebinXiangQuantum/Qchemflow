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

from test_review_r114_118 import _base, _eval, _results, _turn

from chemflow.agent.policies import ScriptedPolicy


def _critique(entries):
    return next(entry for entry in entries if entry["state"] == "CRITIQUE")


def test_r125_composite_results_sum_their_solver_work(tmp_path, monkeypatch) -> None:
    _, _, journal, _ = _eval(tmp_path, monkeypatch, "h2-atomization", policy="scripted")
    controls = {item["method"]: item for item in
                _critique(journal("h2-atomization"))["observation"]["controls"]}
    # H2 (2 orbitals) plus H (1 orbital), with the ground_energy power laws n^3 and n^7
    assert controls["hf"]["selection_work"] == 2 ** 3 + 1 ** 3
    assert controls["ccsd_t"]["selection_work"] == 2 ** 7 + 1 ** 7


def test_r125_the_llm_design_path_decides_like_the_scripted_critic(tmp_path,
                                                                   monkeypatch) -> None:
    from chemflow.agent import report

    original = report._selection_work

    def expensive_sci(node_id, plan):  # make the candidate dominated on the deterministic basis
        nodes = {node.id: node for node in plan.nodes} if plan is not None else {}
        if node_id in nodes and nodes[node_id].refinement == "sci":
            return 10 ** 12
        return original(node_id, plan)

    monkeypatch.setattr(report, "_selection_work", expensive_sci)

    def script(call, messages):
        if call == 1:
            return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                {"template": "sci", "select_cutoff": 0.01, "shots": None}]})])
        built = [data for name, data in _results(messages) if name == "plan.candidates"][-1]
        if call == 2:  # screen-run it first, so the orchestrator's screening hits the cache
            return _turn([("run", {"plan_id": built["plan_id"], "scale": "screen"})])
        return _turn([("design.submit", {"plan_id": built["plan_id"]})])

    # Revised after R127: h2-ground's signed relation gate (variational upper bound) now fails
    # every approximate control, so no control can dominate there. h2-atomization requires
    # only "approximate", so CCSD/CCSD(T) controls can dominate the composite SCI candidate.
    task = "h2-atomization"
    _, rows, journal, _ = _eval(tmp_path, monkeypatch, task, script)
    critique = _critique(journal(task))
    expected = ScriptedPolicy("critic").decide(critique["observation"])
    assert expected["decision"] == "stop"  # dominated by a classical control
    assert critique["decision"]["decision"] == expected["decision"]
    assert critique["decision"]["candidate"] == expected["candidate"]
    assert rows[task]["decision"] == "stop" and rows[task]["correct"]
