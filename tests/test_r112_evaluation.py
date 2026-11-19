import pytest

from chemflow.eval.runner import aggregate_rows, load_suite, score_task, write_report


@pytest.mark.parametrize("valid_count,invalid_count", [(1, 1), (0, 2), (2, 0), (0, 0)])
def test_report_excludes_failed_models_even_with_correct_fallback(
        tmp_path, valid_count, invalid_count):
    task = load_suite("core").tasks[0]
    summary = {"decision": "stop", "candidate": "control", "results": [
        {"node": "control", "value": f"{task.expected.value_hartree} hartree",
         "gate_verdict": "PASS", "relation": "equivalent", "screening_reference": True}]}
    clean = score_task(task, summary, [{"state": "LLM_ERROR"}, {"state": "LLM_CALL"}])
    failed = score_task(task, summary, [{"state": "LLM_FAILED", "role": "critic"}])
    assert clean["valid"] and not clean["llm_failed"] and clean["llm_calls"] == 2
    assert not failed["valid"] and failed["llm_failed"] and failed["correct"]
    rows = [clean] * valid_count + [failed] * invalid_count
    aggregate = aggregate_rows(rows)
    assert aggregate["invalid"] == invalid_count
    assert aggregate["correct_valid"] == valid_count
    write_report({"metadata": {"suite": "core", "policy": "glm", "model": "offline",
                               "roles": "designer,critic", "git_commit": "offline"},
                  "rows": rows, "aggregate": aggregate}, tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert f"Policy accuracy: {valid_count}/{valid_count} valid rows correct" in report
    assert f"{invalid_count} invalid rows excluded" in report
    assert "| id | valid | llm_failed |" in report
    if invalid_count:
        assert f"| {task.id} | False | True |" in report
    assert "not policy decisions" in report
