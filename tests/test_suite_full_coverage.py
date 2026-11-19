from collections import Counter
from copy import deepcopy

import pytest
from pydantic import ValidationError

from chemflow.agent.task import TaskSpec
from chemflow.eval.runner import EvaluationTask, Suite, load_suite


def test_full_schema_categories_difficulties_and_physical_variation():
    suite = load_suite("full")
    counts = Counter(task.category for task in suite.tasks)
    minimum = {"molecular": 12, "reaction": 6, "thermo": 5, "lattice": 6,
               "spectral": 4, "interaction": 4, "selection": 6, "refusal": 4}
    assert len(suite.tasks) >= 50
    assert all(counts[category] >= count for category, count in minimum.items())
    assert all(task.difficulty in {"easy", "medium", "hard"} for task in suite.tasks)
    assert sum(task.difficulty == "hard" for task in suite.tasks) >= 10
    assert len({task.id for task in suite.tasks}) == len(suite.tasks)
    molecular = [task.spec.target for task in suite.tasks if task.category == "molecular"]
    assert {target.electronic_model["basis"] for target in molecular} == {"sto-3g", "6-31g"}
    assert {target.molecule.get("spin", 0) for target in molecular} >= {0, 1, 2}
    assert max(target.active["n_orb"] for target in molecular) == 10
    assert any(target.active["n_orb"] > 6 for target in molecular)
    for task in suite.tasks:
        TaskSpec.model_validate(task.spec.model_dump())
        assert task.reference_source.strip()
        assert task.accuracy_threshold is not None
        assert "tolerance_hartree" not in task.expected.model_dump()


def test_difficulty_is_optional_for_core_but_restricted_when_present():
    task = load_suite("core").tasks[0]
    assert task.difficulty is None
    data = task.model_dump()
    for value in ("unknown", 1, ""):
        with pytest.raises(ValidationError):
            EvaluationTask.model_validate({**data, "difficulty": value})
    assert EvaluationTask.model_validate({**data, "difficulty": "hard"}).difficulty == "hard"
    suite = load_suite("full").model_dump()
    suite["tasks"][0].pop("difficulty")
    with pytest.raises(ValidationError, match="requires a difficulty"):
        Suite.model_validate(suite)


def test_refusals_are_valid_runtime_constraints_not_bad_targets():
    tasks = [task for task in load_suite("full").tasks if task.expected.outcome == "refused"]
    assert len(tasks) >= 4
    for task in tasks:
        spec = deepcopy(task.spec.model_dump())
        assert all(value >= 0 for value in spec["budget"].values())
        spec["budget"] = {"wall_s": 600, "sim_shots": 100000}
        spec["allowed_effects"] = ["pure", "simulator"]
        TaskSpec.model_validate(spec)
    assert any(task.spec.budget.get("wall_s") == 0 for task in tasks)
    assert any(task.spec.budget.get("cpu_core_s") == 0 for task in tasks)
    assert any(0 < task.spec.budget.get("sim_shots", 100000) < 3072 for task in tasks)
    assert any(not task.spec.allowed_effects for task in tasks)
