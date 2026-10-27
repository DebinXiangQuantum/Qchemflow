from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    value: Any
    device: str = "local"
    cost: dict = field(default_factory=dict)
    job_ids: tuple[str, ...] = ()
    software_versions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
