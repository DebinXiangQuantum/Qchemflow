
import pytest
from pydantic import ValidationError

from chemflow.agent.task import PairingTarget, TaskSpec
from chemflow.eval.runner import EvaluationTask, Expected, score_task


def task_spec():
    return TaskSpec.model_validate({
        "question": "Pair correlations in a Hubbard dimer",
        "target": {"kind": "pairing_correlation",
                   "lattice": {"model": "hubbard", "geometry": "chain", "n_sites": 2,
                               "t": 1., "U": 4., "unit": "hartree"},
                   "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2},
                   "target_spin": 0, "site_pair": [0, 1], "quantity": "pair_s_wave"},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": 1e-6,
                   "unit": "dimensionless"},
                  {"kind": "relation_at_least", "relation": "approximate"}],
        "budget": {"wall_s": 600, "sim_shots": 100000},
        "allowed_effects": ["pure", "simulator"], "decision": "Choose a supported method"})


@pytest.mark.parametrize("patch", [{"site_pair": [-1, 0]}, {"site_pair": [True, 1]},
                                   {"site_pair": [0, 2]}, {"quantity": "d_wave"},
                                   {"extra": 1}])
def test_pairing_target_validation(patch):
    with pytest.raises(ValidationError):
        PairingTarget.model_validate({**task_spec().target.model_dump(), **patch})


def test_dimensionless_scoring_uses_the_task_bar():
    task = EvaluationTask(id="dimer-pair", category="lattice", spec=task_spec(),
                          expected=Expected(value=.125, unit="dimensionless"),
                          reference_source="Analytic test value")
    for value, correct in ((.125, True), (.1250005, True), (.125002, False)):
        result = {"node": "answer", "value": f"{value} dimensionless",
                  "relation": "approximate", "gate_verdict": "PASS"}
        row = score_task(task, {"candidate": "answer", "results": [result]}, [])
        assert row["correct"] is correct
    for expected in ({"value": 1}, {"unit": "dimensionless"},
                     {"value": 1, "unit": "dimensionless", "value_hartree": 1}):
        with pytest.raises(ValidationError):
            Expected(**expected)
