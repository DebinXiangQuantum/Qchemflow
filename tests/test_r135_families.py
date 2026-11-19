from types import SimpleNamespace

import pytest
import test_review_v1c as v1c

from chemflow.agent.families import FamilySearch
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.ir.schema import Node, Program
from chemflow.runtime.budget import BudgetExceeded


def _point(value, failing=False):
    return {"value": value, "result": 0., "exact": 0., "error": .1 if failing else 0.,
            "certified_bound": None, "relation": "equivalent", "instance_hash": str(value),
            "certificate": {"steps": []},
            "failures": [{"property": "equivalent", "observed": .1}] if failing else []}


def _search(monkeypatch, action, evaluate):
    tools, base = v1c._tools(v1c._h4())
    monkeypatch.setattr(FamilySearch, "_evaluate", evaluate)
    return tools.call("critic", f"algorithm.{action}", {
        "plan_id": base["plan_id"], "node_id": v1c._method(tools, base["plan_id"], "fci")
    })["data"]


@pytest.mark.parametrize("action", ["falsify", "validity"])
def test_unavailable_task_point_does_not_hide_other_instances(monkeypatch, action):
    def evaluate(self, candidate, reference, axis, value, required_relation=None):
        if value == 1.:
            raise AdapterError("CF-E012", None, "geometry reference refused", "test")
        return _point(value)

    report = _search(monkeypatch, action, evaluate)
    assert report["unevaluable"] == [{"instance": {"geometry_scale": 1.},
                                       "reason": "CF-E012: geometry reference refused"}]
    if action == "falsify":
        assert report["verdict"] == "clean"
        assert report["instances"]
    else:
        assert report["task_point"]["status"] == "unknown"


@pytest.mark.parametrize("action", ["falsify", "validity"])
def test_all_unevaluable_is_structured_error(monkeypatch, action):
    def evaluate(*args, **kwargs):
        raise ValueError("exact geometry reference unavailable")

    with pytest.raises(CheckError, match="No family instance could be evaluated"):
        _search(monkeypatch, action, evaluate)


@pytest.mark.parametrize("action", ["falsify", "validity"])
@pytest.mark.parametrize("error", [BudgetExceeded("CF-E070: exhausted"),
                                   AdapterError("CF-E070", None, "exhausted", "test")])
def test_budget_failure_stops_search(monkeypatch, action, error):
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        raise error

    with pytest.raises((BudgetExceeded, CheckError), match="CF-E070"):
        _search(monkeypatch, action, evaluate)
    assert len(calls) == 1


def test_falsification_does_not_shrink_through_unknown(monkeypatch):
    def evaluate(self, candidate, reference, axis, value, required_relation=None):
        if 1.2 < value < 1.8:
            raise RuntimeError("SCF did not converge")
        return _point(value, failing=value >= 1.8)

    report = _search(monkeypatch, "falsify", evaluate)
    assert report["verdict"] == "falsified"
    assert report["unevaluable"]
    assert all("shrink_bracket" not in example for example in report["counterexamples"])


def test_validity_unknown_gap_is_not_interpolated(monkeypatch):
    def evaluate(self, candidate, reference, axis, value, required_relation=None):
        if 1.2 < value < 1.8:
            raise ValueError("missing reference")
        return _point(value, failing=value >= 1.8)

    report = _search(monkeypatch, "validity", evaluate)
    assert v1c._region(report["axes"][0], 1.5)["status"] == "unknown"
    assert v1c._region(report["axes"][0], 2.)["status"] == "invalid"


@pytest.mark.parametrize("exact_fails", [False, True])
def test_executor_identity_refusal_preserves_reason(exact_fails):
    search = object.__new__(FamilySearch)
    success = {"energy": object()}
    refused = {"refused": {"energy": "CF-E012: identity refused"}}
    search.executor = SimpleNamespace(run=lambda *args, **kwargs:
                                      success if exact_fails else refused)
    search.reference_executor = SimpleNamespace(run=lambda *args, **kwargs: refused)
    search.ledger = None
    plan = Program(name="refusal", nodes=[Node(id="energy", op="ground_energy")],
                   targets=["energy"])
    point = search.evaluate(plan, plan, "geometry_scale", 1.)
    assert point["status"] == "unknown"
    assert "identity refused" in point["reason"]


@pytest.mark.parametrize("run", [{"refused": {"energy": "CF-E070: exhausted"}},
                                 {"budget_overruns": ["wall_s"]}])
def test_returned_budget_refusal_stops_before_reference(run):
    search = object.__new__(FamilySearch)
    search.executor = SimpleNamespace(run=lambda *args, **kwargs: run)
    search.reference_executor = None
    search.ledger = None
    plan = Program(name="budget", nodes=[Node(id="energy", op="ground_energy")],
                   targets=["energy"])
    with pytest.raises(CheckError, match="CF-E070"):
        search.evaluate(plan, plan, "geometry_scale", 1.)
