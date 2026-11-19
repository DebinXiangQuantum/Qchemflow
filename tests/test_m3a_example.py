from test_m3a_task import spec

from chemflow.agent.journal import replay
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskStore


def test_scripted_end_to_end(tmp_path):
    store = TaskStore("scripted-test")
    agent = Orchestrator(store, lambda task: store.freeze(task, "test approver"),
                         tmp_path, max_rounds=1)
    run = agent.run(spec(), diagnostics={"mp2_noons": [2, 0],
                                         "suggested_active": {"n_elec": 2, "n_orb": 2}})
    assert run.decision.decision in ("stop", "escalate")
    assert (tmp_path / "report.md").exists()
    assert replay(run.journal, agent.policies)
