"""Decision replay of hash-pinned historical screening summaries, without computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from chemflow.agent.critic import CriticDecision, decision_items, review
from chemflow.agent.memory import ResultsDB

REGISTRY = Path(__file__).with_name("data") / "screens.json"


@dataclass(frozen=True)
class ReplayResult:
    decision: CriticDecision
    recorded_rejection: bool
    candidates: list[dict]
    controls: list[dict]

    @property
    def matches_record(self) -> bool:
        return (self.decision.decision != "escalate") == self.recorded_rejection


def load_screen(name: str, root: str | Path) -> dict:
    entry = json.loads(REGISTRY.read_text())[name]
    path = Path(root) / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Recorded screen absent: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Recorded screen hash mismatch: {path}")
    return json.loads(raw)


def replay_screen(screen: dict, threshold_mha: float = 1.6,
                  results: ResultsDB | None = None) -> ReplayResult:
    """Map group minima (the best of each three runs) and SCI controls into critic records.

    HSQD_SCALING_20261008.md lines 28-42: same fixed-Sz sector; saved-state
    original-H expectations are variational upper bounds, as are SCI checked
    expectations. A candidate > SCI upper bound + 1.6 mHa is *rejected*,
    not assigned an error vs an exact energy. Group minima suffice to reject
    all 27 runs; they do not reconstruct individual energies. Lines 44-50:
    CCSD/CCSD(T) are approximate and never enter this inequality. Timing is
    the reported method-stage screen cost, not end-to-end or target-scale cost.
    """
    baselines = screen["baselines"]
    controls = [{"id": f"sci_m{orb}", "value": baseline["checked_expectation"],
                 "unit": "hartree", "relation": "variational_upper_bound",
                 "evidence": baseline["valid"], "gate_passed": True,
                 "cost": {"wall_s": baseline["total_s"]},
                 "size": baseline["product_dimension"]}
                for orb, baseline in baselines.items()]
    candidates = []
    for group in screen["quality_groups"]:
        control = baselines[str(group["m"])]
        gap_mha = (group["energy_min"] - control["checked_expectation"]) * 1000
        candidates.append({"id": f"hsqd_m{group['m']}_{group['variant']}",
                           "value": group["energy_min"], "unit": "hartree",
                           "relation": "variational_upper_bound", "evidence": True,
                           "gate_passed": gap_mha <= threshold_mha,
                           "gap_vs_sci_mha": gap_mha,
                           "cost": {"wall_s": group["warm_stage_s"]},
                           "size": group["dimension_min"]})
    if results is not None:
        results.seed_replay("hsqd_scaling_20261008", "candidate", candidates)
        results.seed_replay("hsqd_scaling_20261008", "control", controls)
        candidates = results.replay_records("hsqd_scaling_20261008", "candidate")
        controls = results.replay_records("hsqd_scaling_20261008", "control")
    decision = review(decision_items(candidates), decision_items(controls))
    return ReplayResult(decision, screen["all_rejected_at_1p6_mha"], candidates, controls)
