import pytest

from chemflow.runtime.budget import BudgetExceeded, BudgetLedger, resource_defaults
from chemflow.types import CostRecord

NO_QPU = {"wall_s": 1800, "sim_shots": 200000, "qpu_s": 0, "qpu_shots": 0}


@pytest.mark.parametrize("kind", ["cpu_numeric", "quantum_simulator"])
def test_known_zero_defaults_do_not_exempt_consumed_resources(kind):
    ledger = BudgetLedger({"wall_s": 0, "qpu_s": 0, "qpu_shots": 0})
    with pytest.raises(BudgetExceeded, match="wall_s"):
        ledger.reserve(CostRecord(**resource_defaults({}, kind)))
    ledger = BudgetLedger({"qpu_s": 0, "qpu_shots": 0})
    ledger.reserve(CostRecord(**resource_defaults({}, kind)))
    for amount in (None, 1):
        with pytest.raises(BudgetExceeded, match="qpu_s"):
            ledger.reserve(CostRecord(**resource_defaults({"qpu_s": amount}, "qpu")))
