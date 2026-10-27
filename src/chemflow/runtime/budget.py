import json
import os
import tempfile
from pathlib import Path

from chemflow.types import CostRecord

RESOURCE_DIMENSIONS = ("wall_s", "cpu_core_s", "gpu_s", "sim_shots", "qpu_s",
                       "qpu_shots")


def resource_defaults(resources, kind):
    zeros = {
        "cpu_numeric": ("gpu_s", "sim_shots", "qpu_s", "qpu_shots"),
        "gpu_numeric": ("sim_shots", "qpu_s", "qpu_shots"),
        "quantum_simulator": ("qpu_s", "qpu_shots"),
        "qpu": ("gpu_s", "sim_shots"),
    }
    return {**dict.fromkeys(zeros.get(kind, ()), 0), **resources}


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, limits: dict[str, float], path: Path | None = None):
        if set(limits) & {"money", "qpu_jobs", "currency", "budget_currency"}:
            raise ValueError("D-R2: budgets contain device resources, not money or job counts")
        self.limits = limits
        self.spent: dict[str, float | None] = {key: 0 for key in limits}
        self.path = path
        self.reload()

    def reload(self) -> None:
        if self.path is not None and self.path.exists():
            self.spent = json.loads(self.path.read_text())

    def reserve(self, cost: CostRecord) -> None:
        for dimension, limit in self.limits.items():
            if not hasattr(cost, dimension):
                raise ValueError(f"Unknown budget dimension {dimension}")
            amount = getattr(cost, dimension)
            spent = self.spent[dimension]
            if amount == 0:
                continue
            if spent is None or spent >= limit or (amount is not None and spent + amount > limit):
                raise BudgetExceeded(f"CF-E070: {dimension} budget exceeded")

    def record(self, cost: CostRecord) -> list[str]:
        overruns = []
        for dimension, limit in self.limits.items():
            amount = getattr(cost, dimension)
            if self.spent[dimension] is None:
                if amount:
                    overruns.append(dimension)
                continue
            if amount is not None:
                self.spent[dimension] += amount
                if amount > 0 and self.spent[dimension] > limit:
                    overruns.append(dimension)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".ledger-")
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(self.spent, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return overruns
