from __future__ import annotations

from dataclasses import dataclass

METHOD_ORDER = {method: rank for rank, method in enumerate(("hf", "mp2", "ccsd", "ccsd_t", "fci"))}


def _strongest(controls: list[dict]) -> dict:
    errors = [item.get("error") if item.get("error") is not None else float("inf")
              for item in controls]
    minimum = min(errors)
    tied = [item for item, error in zip(controls, errors, strict=True)
            if error <= minimum + 1e-9]
    return min(tied, key=lambda item: (METHOD_ORDER.get(item.get("method"), len(METHOD_ORDER)),
                                       item["id"]))


@dataclass(frozen=True)
class CriticDecision:
    decision: str
    reason: str
    candidate: str | None = None


def _affordable(control: dict, candidate: dict) -> bool:
    costs = control.get("decision_cost", control.get("cost", {}))
    candidate_costs = candidate.get("decision_cost", candidate.get("cost", {}))
    return all(value is not None and costs.get(key) is not None and
               costs[key] <= value for key, value in candidate_costs.items()
               if key != "diagnostics")


def decision_items(items: list[dict]) -> list[dict]:
    """Use deterministic selection costs without changing recorded resource consumption."""
    return [{**item, "decision_cost": {
        "solver_work": item["selection_work"],
        **{key: item.get("cost", {}).get(key) for key in ("sim_shots", "qpu_shots")}}}
            if "selection_work" in item else item for item in items]


def review(candidates: list[dict], controls: list[dict]) -> CriticDecision:
    """Classical dominance has priority over quantum escalation."""
    if not candidates:
        passing = [control for control in controls if control.get("gate_passed")]
        classical = [control for control in passing if control.get("relation") != "equivalent"]
        if not passing:
            return CriticDecision("revise", "No control passed all gates")
        strongest = _strongest(classical or passing)
        return CriticDecision("stop", f"use classical: {strongest['id']} is the lowest-error "
                              "gate-passing control (method ladder then id breaks ties)",
                              strongest["id"])
    dominated = []
    for candidate in candidates:
        if candidate.get("deployable") is False:
            continue
        if candidate.get("relation_status") in ("declared", "refuted"):
            continue
        if "validity_status" in candidate and candidate["validity_status"] not in (
                "certified", "sampled_valid", "inside_domain"):
            continue
        if not candidate.get("gate_passed") or not candidate.get("evidence") or candidate.get(
                "relation") in (None, "unknown", "unrelated"):
            continue
        classical = [control for control in controls if control.get("relation") != "equivalent"
                     and _affordable(control, candidate)
                     and control.get("gate_passed") and control.get("evidence")]
        if classical:
            dominated.extend(classical)
        else:
            return CriticDecision("escalate", "Candidate meets gates with supported relation "
                                  "and evidence", candidate["id"])
    if dominated:
        strongest = _strongest(dominated)
        return CriticDecision("stop", f"use classical: {strongest['id']} meets gates at "
                              "no greater decision cost", strongest["id"])
    if candidates and any(not item.get("gate_passed") for item in candidates):
        return CriticDecision("revise", "No candidate passed all gates")
    if any(control.get("gate_passed") for control in controls):
        return review([], controls)
    return CriticDecision("stop", "No supported candidate within budget")


class Critic:
    def review(self, candidates: list[dict], controls: list[dict]) -> CriticDecision:
        return review(decision_items(candidates), decision_items(controls))
