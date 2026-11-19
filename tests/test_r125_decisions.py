from copy import deepcopy

import pytest
from test_review_r114_118 import _base, _eval, _results, _turn

from chemflow.agent import report
from chemflow.agent.critic import Critic, decision_items, review
from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.policies import ScriptedPolicy
from chemflow.agent.tools import ToolRegistry
from chemflow.ir.schema import Node, Program


@pytest.mark.parametrize("candidate_time", [0, 100])
@pytest.mark.parametrize("work,expected", [(100, "stop"), (1, "escalate"), (None, "escalate")])
def test_every_critic_uses_work_without_mutating_measured_costs(
        tmp_path, candidate_time, work, expected):
    candidate = {"id": "candidate", "gate_passed": True, "evidence": True,
                 "relation": "variational_upper_bound", "error": 0, "selection_work": work,
                 "cost": {"wall_s": candidate_time, "sim_shots": 0, "qpu_shots": 0}}
    control = {**candidate, "id": "control", "selection_work": 10,
               "cost": {"wall_s": 1, "sim_shots": 0, "qpu_shots": 0}}
    observation = {"candidates": [candidate], "controls": [control]}
    original = deepcopy(observation)
    scripted = ScriptedPolicy("critic").decide(observation)
    llm = LLMPolicy("critic", FakeLLMClient([_turn(content="escalate")]), ToolRegistry(),
                    journal=Journal(tmp_path / "critic.jsonl"))
    assert scripted["decision"] == expected
    assert vars(review(decision_items([candidate]), decision_items([control]))) == scripted
    assert vars(Critic().review([candidate], [control])) == scripted
    assert llm.decide(observation)["verdict"] == scripted
    assert observation == original


def test_cached_and_uncached_designs_have_identical_decisions(tmp_path, monkeypatch):
    original = report._selection_work

    def expensive_candidate(node_id, plan):
        node = next(node for node in plan.nodes if node.id == node_id)
        return 10 ** 12 if node.refinement == "sci" else original(node_id, plan)

    monkeypatch.setattr(report, "_selection_work", expensive_candidate)
    decisions, costs = [], []
    for cached in (False, True):
        def script(call, messages, cached=cached):
            if call == 1:
                return _turn([("plan.candidates", {"plan_id": _base(messages), "candidates": [
                    {"template": "sci", "select_cutoff": 0.01}]})])
            built = [data for name, data in _results(messages) if name == "plan.candidates"][-1]
            if cached and call == 2:
                return _turn([("run", {"plan_id": built["plan_id"], "scale": "screen"})])
            return _turn([("design.submit", {"plan_id": built["plan_id"]})])

        _, rows, journal, _ = _eval(tmp_path / str(cached), monkeypatch, "h2-ground", script)
        critique = next(entry for entry in journal("h2-ground") if entry["state"] == "CRITIQUE")
        decisions.append(critique["decision"])
        costs.append(critique["observation"]["candidates"][0]["cost"]["wall_s"])
        assert rows["h2-ground"]["correct"]
        assert rows["h2-ground"]["relation_ok"]
    assert costs[0] > 0 and costs[1] == 0
    assert decisions[0] == decisions[1]
    assert decisions[0]["decision"] == "escalate"


def composite_plan():
    return Program(name="shared-solvers", nodes=[
        Node(id="sector", op="sector", params={"n_orb": 2, "n_alpha": 1, "n_beta": 1}),
        Node(id="hf", op="ground_energy", refinement="hf", inputs={"sector": "sector"}),
        Node(id="sci", op="ground_energy", refinement="sci", inputs={"sector": "sector"}),
        Node(id="unused", op="ground_energy", refinement="unsupported"),
        Node(id="inner", op="reaction_energy", inputs={"first": "hf", "second": "sci"}),
        Node(id="outer", op="reaction_energy", inputs={"first": "inner", "shared": "hf"}),
    ])


def test_composite_work_counts_each_solver_once_and_uses_module_lookup(monkeypatch):
    plan = composite_plan()
    assert report._selection_work("outer", plan) == 8 + 16
    original = report._selection_work
    monkeypatch.setattr(report, "_selection_work",
                        lambda node_id, plan: 100 if node_id == "sci" else original(node_id, plan))
    assert report._selection_work("outer", plan) == 8 + 100


@pytest.mark.parametrize("missing", ["refinement", "sector", "size"])
def test_composite_work_is_unknown_if_any_solver_work_is_unknown(missing):
    plan = composite_plan()
    if missing == "refinement":
        plan.nodes[2].refinement = "unsupported"
    elif missing == "sector":
        plan.nodes[2].inputs = {}
    else:
        plan.nodes[0].params["n_orb"] = None
    assert report._selection_work("outer", plan) is None


def test_work_without_solvers_or_plan_is_unknown():
    assert report._selection_work("sector", composite_plan()) is None
    assert report._selection_work("outer", None) is None
