from __future__ import annotations

from typing import Protocol

from chemflow.agent.llm import FakeLLMClient
from chemflow.agent.task import TaskSpec

__all__ = ["FakeLLMClient", "Policy", "ScriptedPolicy"]


class Policy(Protocol):
    def decide(self, observation: dict) -> dict | list[dict]: ...


class ScriptedPolicy:
    def __init__(self, role: str, max_orbitals: int = 10):
        self.role = role
        self.max_orbitals = max_orbitals

    def decide(self, observation: dict) -> dict:
        if self.role == "planner":
            return TaskSpec.model_validate(observation["task"]).model_dump(mode="json")
        if self.role == "modeler":
            diagnostics = observation.get("diagnostics", {})
            if diagnostics.get("full_molecular_space"):
                return {"active": None, "diagnostics": diagnostics}
            suggested = diagnostics.get("suggested_active", {})
            if not suggested or suggested["n_orb"] > self.max_orbitals:
                raise ValueError("Modeler has no feasible capped active-space recommendation")
            return {"active": suggested, "diagnostics": diagnostics,
                    "capped": len(diagnostics.get("selected_orbitals", [])) > self.max_orbitals}
        if self.role == "designer":
            cards = observation.get("cards", [])
            target = observation.get("target", "ground_energy")
            selected = [card for card in cards if target in card["target"]]
            candidates = []
            attempted = observation.get("attempted", [])
            for card in selected:
                if card["template"] == "sci":
                    cutoffs = (1e-2, 1e-3) if not attempted else (1e-4,)
                    candidates.extend({"template": "sci", "select_cutoff": cutoff}
                                      for cutoff in cutoffs)
                elif card["template"] in ("sqd", "pairing_sqd"):
                    shots_options = (1024, 2048) if not attempted else (4096, 8192)
                    candidates.extend({"template": "sqd", "shots": shots}
                                      for shots in shots_options)
            return {"candidates": [candidate for candidate in candidates
                                   if candidate not in attempted]}
        if self.role == "critic":
            from chemflow.agent.critic import decision_items, review

            return vars(review(decision_items(observation.get("candidates", [])),
                               decision_items(observation.get("controls", []))))
        if self.role == "analyst":
            return {"decision": observation.get("decision"),
                    "reason": observation.get("reason")}
        return {"status": "no_action"}
