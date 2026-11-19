import json

import pytest

from chemflow.diagnostics import AdapterError
from chemflow.ir.schema import Node, Program, Scan
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import ExecutionResult, Executor
from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetExceeded, BudgetLedger
from chemflow.types import CostRecord
from chemflow.units import Q


@pytest.mark.parametrize("dimension", RESOURCE_DIMENSIONS)
def test_measured_overrun_persists_and_reloads(tmp_path, dimension):
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger({dimension: 1}, path=path)
    assert ledger.record(CostRecord(**{dimension: 2})) == [dimension]
    assert json.loads(path.read_text()) == {dimension: 2}
    reopened = BudgetLedger({dimension: 1}, path=path)
    assert reopened.spent == {dimension: 2}
    reopened.reserve(CostRecord(**{dimension: 0}))
    with pytest.raises(BudgetExceeded):
        reopened.reserve(CostRecord(**{dimension: None}))
    assert reopened.record(CostRecord(**{dimension: 0})) == []


@pytest.mark.parametrize("dimension", RESOURCE_DIMENSIONS)
@pytest.mark.parametrize("limit,spent", [(0, 0), (1, 1), (1, 2)])
@pytest.mark.parametrize("prediction", [0, 1, None])
def test_exhausted_dimension_reservation_and_forecast(dimension, limit, spent, prediction):
    ledger = BudgetLedger({dimension: limit})
    ledger.spent[dimension] = spent
    predicted = CostRecord(**{dimension: prediction})
    registry = molecular_registry()
    calls = []
    registry.implementations["molecule"].resource_model = lambda **kwargs: predicted.model_dump()
    registry.implementations["molecule"].adapter = lambda **kwargs: (
        calls.append(True) or ExecutionResult("completed", cost={dimension: 0}))
    program = Program(name="exhausted", nodes=[Node(
        id="node", op="molecule", params={"atoms": "H 0 0 0"})])
    if prediction == 0:
        ledger.reserve(predicted)
        computed = Executor(registry).run(program, ledger=ledger)
        assert computed["node"] == "completed"
        assert computed["refused"] == {} and computed["budget_overruns"] == []
        assert calls == [True]
    else:
        with pytest.raises(BudgetExceeded, match="CF-E070"):
            ledger.reserve(predicted)
        with pytest.raises(AdapterError, match="CF-E070"):
            Executor(registry).run(program, ledger=ledger)
        assert calls == []


@pytest.mark.parametrize("dimension", RESOURCE_DIMENSIONS)
def test_remaining_budget_accepts_unknown_and_checks_known(dimension):
    ledger = BudgetLedger({dimension: 3})
    ledger.record(CostRecord(**{dimension: 1}))
    for prediction in (None, 0, 2):
        ledger.reserve(CostRecord(**{dimension: prediction}))
    with pytest.raises(BudgetExceeded):
        ledger.reserve(CostRecord(**{dimension: 3}))


@pytest.mark.parametrize("dimension", ["money", "qpu_jobs", "currency"])
def test_removed_dimensions_are_refused(dimension):
    with pytest.raises(ValueError, match="D-R2"):
        BudgetLedger({dimension: 20})


def test_scan_retains_completed_points_and_refuses_later_work():
    registry = molecular_registry()
    calls = []
    registry.implementations["molecule"].resource_model = lambda **kwargs: {"qpu_s": 1}
    registry.implementations["molecule"].adapter = lambda **kwargs: (
        calls.append(True) or ExecutionResult("completed", cost={"qpu_s": 3}))
    program = Program(name="scan", nodes=[Node(id="node", op="molecule",
        params={"atoms": "H 0 0 {distance}"})], budget={"qpu_s": 2},
        scans=[Scan(name="distance", values=Q([0.74, 0.75], "angstrom"))])
    result = Executor(registry).run(program)
    assert calls == [True]
    assert result["scan"][0]["node"] == "completed"
    assert "CF-E070" in result["scan"][1]["refused"]["node"]
    assert result["budget_overruns"] == [
        {"node": "node", "dimension": "qpu_s", "limit": 2, "spent": 3}]
    assert "1:node" in result["refused"]


@pytest.mark.parametrize("prediction", [1, None])
def test_runtime_exhaustion_refuses_positive_and_unknown_but_not_zero(prediction):
    registry = molecular_registry()
    calls = []
    predictions = {"H 0 0 0": 0.2, "H 0 0 1": prediction, "H 0 0 2": 0}
    registry.implementations["molecule"].resource_model = (
        lambda atoms, **kwargs: {"qpu_s": predictions[atoms]})
    registry.implementations["molecule"].adapter = lambda atoms, **kwargs: (
        calls.append(atoms) or ExecutionResult(atoms, cost={"qpu_s": 3 if atoms == "H 0 0 0" else 0}))
    program = Program(name="runtime", budget={"qpu_s": 2}, nodes=[
        Node(id=identifier, op="molecule", params={"atoms": atoms})
        for identifier, atoms in zip(("paid", "refused", "free"), predictions)])
    result = Executor(registry).run(program)
    assert calls == ["H 0 0 0", "H 0 0 2"]
    assert "CF-E070" in result["refused"]["refused"]
    assert result["paid"] == "H 0 0 0" and result["free"] == "H 0 0 2"
