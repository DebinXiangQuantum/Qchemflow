from chemflow.agent.journal import Journal, replay
from chemflow.agent.policies import ScriptedPolicy


def test_replay(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    policy = ScriptedPolicy("critic")
    observation = {"candidates": [], "controls": []}
    journal.append(state="CRITIQUE", role="critic", observation=observation,
                   decision=policy.decide(observation), reason="empty")
    decision = replay(journal, {"critic": policy})[0]["decision"]
    assert decision["decision"] == "revise"
    assert decision["candidate"] is None
