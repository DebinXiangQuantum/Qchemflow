import json

import pytest
from test_m3a_task import spec
from test_r106_loop import turn

from chemflow.agent.journal import Journal
from chemflow.agent.llm import FakeLLMClient, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore


class InspectingDesigner(FakeLLMClient):
    def __init__(self, recover):
        super().__init__([])
        self.recover = recover

    def complete(self, messages, tools):
        self.messages.append(list(messages))
        base = json.loads(messages[1]["content"])["observation"]["base_plan_id"]
        current = len(self.messages)
        if current == 1:
            return turn("missing" if self.recover else "plan.render", "inspect",
                        arguments={"plan_id": base})
        if current == 2:
            return turn("cards.search", "cards")
        if current == 3:
            return turn("plan.candidates", "candidate", arguments={
                "plan_id": base, "candidates": [{"template": "sci", "select_cutoff": 0.01}]})
        if current == 4:
            return turn("cards.search", "post-proposal-inspection")
        return turn(content="Candidate ready for screening")


@pytest.mark.parametrize("recover", [False, True])
def test_inspect_then_propose_on_third_turn_is_screened_and_critiqued(tmp_path, recover):
    store = TaskStore("offline-test")
    designer = InspectingDesigner(recover)
    critic = FakeLLMClient([turn(content="Challenge the screened candidate")])
    journal = Journal(tmp_path / "policy.jsonl")
    policies = {"designer": LLMPolicy("designer", designer, None, journal=journal),
                "critic": LLMPolicy("critic", critic, None, journal=journal)}
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=1, policies=policies)
    result = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    records = [json.loads(line) for line in result.journal.read_text().splitlines()]
    assert len(designer.messages) == 5
    assert not any(entry["state"] == "DESIGN_REFUSED" for entry in records)
    screen = next(entry for entry in records if entry["state"] == "SCREEN")
    observation = json.loads(critic.messages[0][1]["content"])["observation"]
    assert len(observation["candidates"]) == 1
    assert observation["controls"]
    critique = next(entry for entry in records if entry["state"] == "CRITIQUE")
    assert observation["candidates"] == critique["observation"]["candidates"]
    assert screen["outputs"]
    assert any(entry["state"] == "LLM_TOOL" and
               entry["decision"].get("name") == "plan.candidates" for entry in records)
