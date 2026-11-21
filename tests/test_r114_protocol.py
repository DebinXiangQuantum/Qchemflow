import json

from test_g5_evaluation import synthetic

from chemflow.eval.runner import aggregate_rows, score_task, write_report


def turn(name=None, arguments=None):
    return {"content": "", "tool_calls": [] if name is None else [
        {"id": "call", "name": name, "arguments": arguments or {}}]}


def records(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def test_design_and_relation_totals_are_distinct_and_reported(tmp_path):
    task, summary, _ = synthetic()
    screened = score_task(task, summary, [{"state": "SCREEN", "outputs": ["chosen"]}])
    declined = score_task(task, summary, [{"state": "DESIGN_DECLINED"}])
    refused = score_task(task, summary, [{"state": "DESIGN_REFUSED"}])
    unanswered = score_task(task, {**summary, "candidate": None}, [])
    rows = [screened, declined, refused, unanswered]
    assert [row["design"] for row in rows] == ["screened", "declined", "refused", None]
    assert [row["answer_from_candidate"] for row in rows] == [True, False, False, False]
    assert unanswered["relation_ok"] is None
    aggregate = aggregate_rows(rows)
    expected = {"relation_ok": 3, "relation_rated": 3, "design_screened": 1,
                "design_declined": 1, "design_refused": 1, "answer_from_candidate": 1}
    assert {key: aggregate[key] for key in expected} == expected
    write_report({"metadata": {"suite": "core", "policy": "scripted", "model": None,
                               "roles": "designer", "git_commit": "offline"},
                  "rows": rows, "aggregate": aggregate}, tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert all(f"| {key} | {value} |" in report for key, value in expected.items())
