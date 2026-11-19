from test_m3a_task import spec

from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.report import write_report
from chemflow.agent.task import TaskStore


def test_backend_chain_and_orbital_label(tmp_path):
    store = TaskStore("key")
    agent = Orchestrator(store, lambda task: store.freeze(task, "human"), tmp_path,
                         max_rounds=1)
    run = agent.run(spec(), diagnostics={"suggested_active": {"n_elec": 2, "n_orb": 2}})
    assert "canonical HF frontier window" in run.report
    assert all("backend_kind" in row for row in run.results)
    assert any("quantum_simulator→cpu_numeric" in row["backend_kind"]
               for row in run.results)
    assert "Target cost interval" in run.report
    assert "target-scale dominance" in write_report("q", "stop", "r", [])
