import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chemflow.eval.runner import load_suite, run


def evaluate(out, tasks, jobs):
    completed = subprocess.run(
        [str(Path(sys.executable).with_name("chemflow")), "eval", "run", "--suite", "full",
         "--policy", "scripted", "--tasks", ",".join(tasks), "--jobs", str(jobs),
         "--out", str(out)], check=False, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    return json.loads((out / "results.json").read_text())


def test_jobs_determinism_order_and_runtime_refusals(tmp_path):
    identifiers = ["h2-equilibrium", "impurity-weak", "select-h4-1p5",
                   "refuse-wall-s", "refuse-cpu-core-s", "refuse-sim-shots", "refuse-effects"]
    sequential = evaluate(tmp_path / "sequential", identifiers[::-1], 1)
    parallel = evaluate(tmp_path / "parallel", identifiers[::-1], 2)
    assert [row["id"] for row in parallel["rows"]] == identifiers
    for first, second in zip(sequential["rows"], parallel["rows"], strict=True):
        assert {key: value for key, value in first.items() if key != "wall_s"} == {
            key: value for key, value in second.items() if key != "wall_s"}
        assert first["correct"]
        folder = tmp_path / "parallel" / first["id"]
        assert (folder / "agent.log").is_file()
        if first["category"] == "refusal":
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["refusal"]["nothing_consumed"]
            assert summary["refusal"]["spent_at_refusal"] == summary["spent"]
            assert summary["spent"]["sim_shots"] == 0
            if first["id"] != "refuse-sim-shots":
                assert all(value == 0 for value in summary["spent"].values())


@pytest.mark.parametrize("jobs", [0, -1, 1.5, True])
def test_invalid_jobs_leave_no_output(tmp_path, jobs):
    out = tmp_path / "out"
    args = argparse.Namespace(suite="full", suite_file=None, tasks="h2-equilibrium", jobs=jobs,
                              policy="scripted", model=None, roles="designer", out=out)
    with pytest.raises(ValueError, match="positive integer"):
        run(args)
    assert not out.exists()


def test_all_selection_tasks_have_deployable_passing_candidates(tmp_path):
    tasks = [task.id for task in load_suite("full").tasks if task.category == "selection"]
    result = evaluate(tmp_path / "out", tasks, 2)
    assert all(row["correct"] for row in result["rows"])
    for identifier in tasks:
        summary = json.loads((tmp_path / "out" / identifier / "summary.json").read_text())
        candidates = [item for item in summary["results"]
                      if item["node"].startswith("candidate")]
        verdicts = [item["gate_verdict"] for item in candidates]
        assert "PASS" in verdicts
        assert all(item["deployable"] for item in candidates)
        assert all("deployment" in item for item in candidates)


def test_measured_timings_do_not_change_candidate_dominance():
    from chemflow.agent.policies import ScriptedPolicy

    candidate = {"id": "candidate", "gate_passed": True, "evidence": True,
                 "relation": "variational_upper_bound", "error": 0,
                 "selection_work": 16}
    control = {**candidate, "id": "control", "relation": "approximate", "selection_work": 64}
    critic = ScriptedPolicy("critic")
    for candidate_time, control_time in ((0.001, 1), (1, 0.001)):
        candidate["cost"] = {"wall_s": candidate_time, "sim_shots": 0, "qpu_shots": 0}
        control["cost"] = {"wall_s": control_time, "sim_shots": 0, "qpu_shots": 0}
        verdict = critic.decide({"candidates": [candidate], "controls": [control]})
        assert verdict["decision"] == "escalate" and verdict["candidate"] == "candidate"
    control["selection_work"] = 8
    verdict = critic.decide({"candidates": [candidate], "controls": [control]})
    assert verdict["decision"] == "stop" and verdict["candidate"] == "control"
