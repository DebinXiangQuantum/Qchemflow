from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemflow.ids import canonical_json


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, state: str, role: str, observation: dict,
               decision: Any, reason: str = "", calls: list | None = None,
               outputs: list | None = None) -> None:
        entry = {"state": state, "role": role, "observation": observation,
                 "decision": decision, "reason": reason, "tool_calls": calls or [],
                 "calls": calls or [],
                 "outputs": outputs or []}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(entry) + "\n")


def replay(journal: Journal | str | Path, policies: dict) -> list[dict]:
    path = journal.path if isinstance(journal, Journal) else Path(journal)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    for entry in entries:
        policy = policies.get(entry["role"])
        if policy is not None:
            actual = policy.decide(entry["observation"])
            assert json.loads(canonical_json(actual)) == entry["decision"], entry["state"]
    return entries
