import pytest
import test_review_a1a as a1a
import test_review_a1b as a1b

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.design import (
    MOVES,
    DesignStore,
    apply_move,
    design_loop,
    select_move,
)


@pytest.mark.parametrize("move", MOVES)
def test_moves_preserve_identity_and_gates(move):
    program, response = a1a._enriched_plan()
    a1a._accept(program, response)
    original = program.model_dump()
    changed = apply_move(program, response, move)
    assert changed is not None
    assert program.model_dump() == original
    assert changed.gates == program.gates
    before = {node.id: node for node in program.nodes}
    for node in changed.nodes:
        if node.op in ("sector", "impurity_model", "spectral_function"):
            assert node == before[node.id]
    assert changed.hash != program.hash


@pytest.mark.parametrize("flags,expected", [
    (["incomplete_ground", "ground_energy_error"], "more_shots"),
    (["ground_energy_error"], "enriched_ground"),
    (["incomplete_response"], "doubles_closure"),
    (["sum_rule_error"], "doubles_closure"),
    (["truncated_resolution"], "more_ritz_roots"),
    (["accuracy_failure"], "full_sector"),
    (["relation_failure"], "full_sector"),
    (["invalid_candidate", "accuracy_failure"], None),
])
def test_rule_table(flags, expected):
    program, response = a1a._enriched_plan()
    move, changed, rationale = select_move(program, response, {"flags": flags}, MOVES)
    assert move == expected
    assert (changed is not None) == (expected is not None)
    assert bool(rationale) == (expected is not None)


def test_shots_cap_and_full_exclusion():
    program, response = a1a._enriched_plan()
    changed = apply_move(program, response, "more_shots")
    assert apply_move(changed, response, "more_shots") is None
    assert next(node.params["shots"] for node in changed.nodes
                if node.op == "sample_configurations") == 40000
    assert select_move(program, response, {"flags": ["accuracy_failure"]},
                       [move for move in MOVES if "full" not in move])[0] is None


def run_loop(program, response, **kwargs):
    registry, store = a1b._registry(), DesignStore()
    result = design_loop(program, response, target=a1b.TARGET, registry=registry,
                         store=store, instances=3, seed=5, **kwargs)
    return result, registry, store


def test_cutoff_does_not_apply_unverified_move_or_register():
    program, response = a1a._enriched_plan()
    result, registry, store = run_loop(a1a._accept(program, response), response,
                                      max_rounds=1, register_as="cutoff")
    assert result.outcome == "exhausted" and result.design is None
    assert len(result.trajectory) == 1 and result.trajectory[0]["move"] is None
    assert store.list() == [] and not registry.designs
    entry = result.trajectory[0]
    measured = store.get(entry["execution_report_id"])
    stable = store.get(entry["report_id"])
    assert measured.resources["wall_s"] > 0 and stable.resources["wall_s"] is None
    assert measured.instances == stable.instances
    assert any(abs(item["diagnostics"]["ground_energy_error"]) > 0.1
               for item in stable.instances)


def test_task_bound_loop_keeps_bar_and_registers_only_passing_evidence():
    program, response = a1a._enriched_plan()
    task_store = TaskStore("a1d")
    record = task_store.freeze(TaskSpec.model_validate({
        "question": "AIM", "target": a1b.TARGET, "gates": [a1a.ACCURACY],
        "budget": {"wall_s": 900}, "allowed_effects": ["pure", "simulator"],
        "decision": "d"}), "a1d")
    result, registry, store = run_loop(program, response, max_rounds=5, task=record,
                                      register_as="task_loop")
    assert result.outcome == "passed"
    assert result.design["status"] == "experimental"
    assert len(registry.designs) == len(store.list()) == 1
    assert result.design["evidence"] == result.trajectory[-1]["report_id"]
    for entry in result.trajectory:
        report = store.get(entry["report_id"])
        assert report.accuracy["source"] == "task"
        assert report.accuracy["threshold"] == 0.1
        assert report.accuracy["task_hash"] == record.content_hash
        assert not report.template["gates"]
        for instance in entry["resources"]["subspaces"]:
            for sector in instance["sectors"].values():
                assert sector["fraction"] == sector["size"] / sector["full_size"]


def test_empty_moves_and_invalid_round_limits():
    program, response = a1a._enriched_plan()
    a1a._accept(program, response)
    result, _, _ = run_loop(program, response, max_rounds=5, moves=[])
    assert result.outcome == "exhausted" and len(result.trajectory) == 1
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_rounds"):
            run_loop(program, response, max_rounds=limit)


def test_full_fallback_is_verified_and_removes_sampling_cost():
    program, response = a1a._enriched_plan()
    result, _, store = run_loop(a1a._accept(program, response), response,
                                max_rounds=2, moves=["full_sector"], register_as="full")
    assert result.outcome == "passed"
    assert [entry["move"] for entry in result.trajectory] == ["full_sector", None]
    final = store.get(result.trajectory[-1]["report_id"])
    assert final.inferred_relation == "equivalent" and final.max_metric < 1e-8
    assert final.resources["sim_shots"] == 0
    assert all(sector["fraction"] == 1 for instance in final.resources["subspaces"]
               for sector in instance["sectors"].values())


def test_doubles_closure_has_exact_two_layer_support():
    from chemflow.domains.algorithms import extend_subspace
    from chemflow.types import Sector, Subspace

    source = Subspace(sector=Sector(n_alpha=2, n_beta=2, n_orb=5),
                      determinants=(0b00110 | (0b00011 << 5),))
    target = Sector(n_alpha=3, n_beta=2, n_orb=5)
    image = source.determinants[0] | 1
    result = extend_subspace(None, source, target, "create", 0, "alpha", closure="doubles")
    expected = {state for state in range(1 << 10)
                if (state & 31).bit_count() == 3 and (state >> 5).bit_count() == 2
                and (state ^ image).bit_count() <= 4}
    assert set(result.determinants) == expected


def test_loop_charges_the_supplied_ledger_across_rounds():
    from chemflow.runtime.budget import BudgetLedger

    program, response = a1a._enriched_plan()
    ledger = BudgetLedger({"sim_shots": 200000})
    result, _, _ = run_loop(a1a._accept(program, response), response,
                             max_rounds=2, ledger=ledger)
    assert ledger.spent["sim_shots"] == sum(entry["resources"]["sim_shots"]
                                            for entry in result.trajectory)
    assert ledger.spent["sim_shots"] == 180000
