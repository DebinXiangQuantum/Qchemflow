from __future__ import annotations

from itertools import permutations

import pytest
from pydantic import ValidationError
from test_critic_answer import control
from test_g5_evaluation import synthetic

from chemflow.agent.critic import review
from chemflow.agent.report import result_method
from chemflow.eval.runner import EvaluationTask, load_suite, score_task
from chemflow.ir.schema import Node, Program
from chemflow.units import Q


@pytest.mark.parametrize("gates", [
    [], [{"kind": "relation_at_least", "relation": "equivalent"}],
    [{"kind": "l1_error_vs_exact", "threshold": 0.1, "unit": "dimensionless"}],
    *[[{"kind": "abs_error_vs_exact", "threshold": value, "unit": "hartree"}]
      for value in (-1, float("nan"), float("inf"))],
    [{"kind": "abs_error_vs_exact", "threshold": 1, "unit": "angstrom"}],
])
def test_answer_requires_valid_target_accuracy_gate(gates):
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"] = gates
    with pytest.raises(ValidationError):
        EvaluationTask.model_validate(data)


def test_gate_units_multiple_gates_and_reference_use_same_bar():
    task, summary, journal = synthetic()
    data = task.model_dump(mode="json")
    data["spec"]["gates"] = [
        {"kind": "abs_error_vs_exact", "threshold": 0.01, "unit": "hartree"},
        {"kind": "abs_error_vs_exact", "threshold": Q(0.0016, "hartree").to("eV").value,
         "unit": "eV"}]
    task = EvaluationTask.model_validate(data)
    assert task.accuracy_threshold == pytest.approx(0.0016)
    for offset, expected in ((0.0015, True), (0.0017, False)):
        for headline in summary["results"][1:]:
            headline["value"] = f"{task.expected.value_hartree + offset} hartree"
        row = score_task(task, summary, journal)
        assert row["correct"] is expected and row["reference_correct"] is expected


def test_zero_accuracy_and_refusal_without_gate():
    data = load_suite("core").tasks[0].model_dump(mode="json")
    data["spec"]["gates"][0]["threshold"] = 0
    assert EvaluationTask.model_validate(data).accuracy_threshold == 0
    data["spec"]["gates"] = []
    data["expected"] = {"outcome": "refused", "refusal_code": "CF-E070"}
    assert EvaluationTask.model_validate(data).accuracy_threshold is None


def test_spectral_scalar_uses_declared_dimensionless_l1_bar():
    task = next(item for item in load_suite("core").tasks if item.category == "spectral")
    summary = {"candidate": "answer", "results": [{"node": "answer"},
               {"node": "reference", "screening_reference": True}]}
    for offset, correct in ((0.09, True), (0.11, False)):
        values = {node: task.expected.value_hartree + offset for node in ("answer", "reference")}
        row = score_task(task, summary, [], spectral_values=values)
        assert row["correct"] is correct and row["reference_correct"] is correct
    data = task.model_dump(mode="json")
    data["spec"]["gates"] = [{"kind": "abs_error_vs_exact", "threshold": 0.1, "unit": "hartree"}]
    with pytest.raises(ValidationError):
        EvaluationTask.model_validate(data)


@pytest.mark.parametrize("with_candidate", [False, True])
def test_error_band_method_ladder_and_id_are_order_independent(with_candidate):
    controls = [control("z-ccsd", method="ccsd", error=1e-7 + 1e-12),
                control("a-triples", method="ccsd_t", error=1e-7),
                control("a-ccsd", method="ccsd", error=1e-7 + 1e-12)]
    candidates = [control("candidate", cost={"wall_s": 100})] if with_candidate else []
    for ordering in permutations(controls):
        assert review(candidates, list(ordering)).candidate == "a-ccsd"
    controls.append(control("mp2", method="mp2", error=1e-7 + 2e-9))
    assert review(candidates, controls).candidate == "a-ccsd"
    controls.append(control("hf", method="hf", error=1e-7 + 0.5e-9))
    assert review(candidates, controls).candidate == "hf"


def test_control_order_ignores_measured_costs_and_unknown_methods_sort_last():
    for slow, fast in ((100, 0.001), (0.001, 100), (None, None)):
        controls = [control("ccsd", method="ccsd", cost={"wall_s": slow}),
                    control("triples", method="ccsd_t", cost={"wall_s": fast}),
                    control("a-unknown", cost={"wall_s": 0})]
        assert review([], controls).candidate == "ccsd"


def test_method_metadata_for_primitive_and_composite_controls():
    plan = Program(name="methods", nodes=[
        Node(id="energy", op="ground_energy", refinement="ccsd"),
        Node(id="thermo", op="thermochemistry"),
        Node(id="reaction", op="reaction_energy", inputs={"species0": "energy", "thermo0": "thermo"}),
        Node(id="component", op="component_energy", params={"method": "ccsd_t"}),
        Node(id="interaction", op="interaction_energy", inputs={"component0": "component"}),
    ])
    assert result_method("energy", plan) == "ccsd"
    assert result_method("reaction", plan) == "ccsd"
    assert result_method("interaction", plan) == "ccsd_t"
    assert result_method("thermo", plan) is None
