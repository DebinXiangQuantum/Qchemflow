import pytest
from test_v1c_domains import specification, toolset

from chemflow.design import DesignStore
from chemflow.ids import content_hash


@pytest.mark.parametrize("declared", [False, True])
def test_clean_search_records_instances_and_persists(tmp_path, declared):
    spec = specification()
    if not declared:
        spec.pop("validity")
    tools, base, _ = toolset(spec, tmp_path)
    node = tools.plans.get(base["plan_id"]).references[0]
    report = tools.call("critic", "algorithm.falsify", {
        "plan_id": base["plan_id"], "node_id": node})["data"]
    assert report["verdict"] == "clean"
    assert len(report["instances"]) >= 12
    assert report["seed"] == 1729
    assert not report["counterexamples"]
    assert tools.family_search.ledger.spent["wall_s"] > 0
    assert content_hash({key: value for key, value in report.items()
                         if key != "report_id"}) == report["report_id"]
    stored = DesignStore(tmp_path / "designs.sqlite").get(report["report_id"])
    assert stored.instances == report["instances"]
    values = [item["hubbard_u"] for item in report["instances"]]
    assert min(values) == 0
    assert max(values) == (4 if declared else 8)


def test_budget_refusal_is_not_a_clean_search():
    spec = specification()
    spec["budget"]["wall_s"] = 0
    tools, base, _ = toolset(spec)
    node = tools.plans.get(base["plan_id"]).references[0]
    with pytest.raises(Exception, match="budget|refused"):
        tools.call("critic", "algorithm.falsify", {"plan_id": base["plan_id"], "node_id": node})
    assert not list(tools.designs._connection.execute("SELECT * FROM verification_reports"))


def test_refutation_overlays_previous_results(monkeypatch):
    from chemflow.ops import molecular

    spec = specification("molecular")
    spec.pop("validity")
    tools, base, executor = toolset(spec)
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"],
        "candidates": [{"template": "sci", "select_cutoff": 1e-4}]})["data"]
    node = next(iter(built["ids"].values()))
    run = tools.call("designer", "run", {"plan_id": built["plan_id"], "scale": "screen"})["data"]
    original = molecular.sci_energy

    def broken(*args, **kwargs):
        energy, diagnostics, extra = original(*args, **kwargs)
        return energy - .1, diagnostics, extra

    monkeypatch.setattr(molecular, "sci_energy", broken)
    executor.cache.clear()
    report = tools.call("critic", "algorithm.falsify", {
        "plan_id": built["plan_id"], "node_id": node})["data"]
    assert report["verdict"] == "falsified"
    previous = tools.call("critic", "results.get", {"identifier": run["results"][node]})["data"]
    assert previous["relation_status"] == "refuted"
    assert previous["falsification_report"] == report["report_id"]
    assert any(item["property"] == "projected_lowest_eigenvalue"
               for item in report["counterexamples"])
    proof = tools.call("critic", "algorithm.prove", {
        "plan_id": built["plan_id"], "node_id": node})["data"]
    assert proof["relation_status"] == "refuted"
