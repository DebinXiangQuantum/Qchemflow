import json

import pytest

from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetExceeded, BudgetLedger
from chemflow.types import CostRecord


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
