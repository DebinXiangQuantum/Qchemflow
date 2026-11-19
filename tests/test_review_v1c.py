"""Reviewer tests for V1c, validity domains of approximations (written by Claude; implementers
must not edit).

User request (2026-10-14): "Some time the algorithm is approximated, THEN the agent should give
the range that the approximation is enough, and what range the algorithm is invalid." The user
chose that the domain is declared in the signed task, like accuracy bars (A1c): the agent maps
inside it and never chooses it. Plan: docs/research/07-verified-design.md, V1c.

The canonical case was measured by the reviewer with PySCF, linear H4 in STO-3G:
- CCSD(T) − FCI is about −0.04 mHa at 1.0 Å spacing, −1.05 at 1.4, −3.8 at 1.6, −20 at 2.0
  and −35 at 2.5.
- At the 1.6 mHa bar, CCSD(T) is valid near equilibrium and invalid beyond about 1.45 Å.
  It overshoots below FCI there, so it is not variational.
- The T1 diagnostic stays at or below 0.02 throughout, so an a-priori diagnostic alone would
  miss the failure.

Contract:
- `TaskSpec.validity` is a tuple of axes: `{axis, range: [lo, hi], points, resolution}`.
  - Supported axes: `geometry_scale` for molecular ground-energy targets (all coordinates
    scaled about the first atom; 1.0 is the signed geometry), `hubbard_u` for lattice
    targets, and `impurity_u` for impurity spectral targets.
  - The range must contain the task point.
  - `points` (3–33) are evenly spaced.
  - `resolution` is the bisection tolerance in axis units.
- `algorithm.validity {plan_id, node_id}` is a designer/critic tool. Its schema has exactly
  those two properties: the domain, bar and reference come from the signed task. A task
  without `validity` is refused with a structured error.
  - It maps the node's dependency closure over each axis against an independent exact
    reference, at the task's accuracy bar.
  - It is charged to the task budget.
  - It returns `{"node", "method", "bar", "axes": [...], "task_point": {...}}`. Each axis has:
    - `points`: `[{"value", "result", "exact", "error", "status", "certified_bound"}]`;
    - `regions`: `[{"lo", "hi", "status"}]`, partitioning the range;
    - `counterexamples`: `[{"value", "error"}]`.
  - Statuses:
    - `certified`: the error certificate is no larger than the bar;
    - `sampled_valid`: |error| is no larger than the bar at the samples bounding the region;
    - `invalid`: a counterexample with |error| above the bar;
    - `unknown`.
  - A valid→invalid transition between adjacent points is bisected until the unknown gap is
    no wider than `resolution`.
  - The map states any interpolation assumption in `assumption`.
- `certified_bound`, for variational results with a state vector (SCI, SQD, Ritz, VQE), is
  Weinstein's σ = sqrt(⟨H²⟩ − ⟨H⟩²). It is recorded with the ground-state-identification
  assumption. It is null for results without a state, such as CCSD(T).
- Candidate items gain `validity_status`, the status at the task point, whenever the task
  declares validity. The critic never escalates a candidate whose status is not certified
  or sampled_valid.
- report.md gains a "Validity" section, stating for the answer the valid and invalid ranges
  with the bar.
"""
from __future__ import annotations

import pytest

from chemflow.agent.task import TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor

BAR = 0.0016


def _h4(validity=True):
    return {"question": "H4 chain energy", "target": {
        "operator": "ground_energy",
        "molecule": {"atoms": "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0"},
        "electronic_model": {"basis": "sto-3g"}, "active": {"n_elec": 4, "n_orb": 4},
        "sector": {"n_alpha": 2, "n_beta": 2, "n_orb": 4}, "target_spin": 0},
        "gates": [{"kind": "abs_error_vs_exact", "threshold": BAR, "unit": "hartree"}],
        "budget": {"wall_s": 1800, "sim_shots": 200000},
        "allowed_effects": ["pure", "simulator"], "decision": "d",
        **({"validity": [{"axis": "geometry_scale", "range": [0.7, 2.5], "points": 7,
                          "resolution": 0.1}]} if validity else {})}


def _tools(spec):
    store = TaskStore("reviewer")
    record = store.freeze(TaskSpec.model_validate(spec), "reviewer")
    registry = molecular_registry()
    tools = make_tools(registry, Executor(registry), record, store)
    empty = tools.call("designer", "plan.new", {"identifier": "p"})["data"]["plan_id"]
    base = tools.call("modeler", "plan.base", {"plan_id": empty,
                                               "active": {"n_elec": 4, "n_orb": 4}})["data"]
    return tools, base


def _method(tools, plan_id, method):
    plan = tools.plans.get(plan_id)
    return next(node.id for node in plan.nodes if node.refinement == method)


def _pyscf_error(spacing):
    from pyscf import cc, fci, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {index * spacing}" for index in range(4)),
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    coupled = cc.CCSD(mf).run()
    return coupled.e_tot + coupled.ccsd_t() - fci.FCI(mf).kernel()[0]


def _region(axis, value):
    return next(region for region in axis["regions"] if region["lo"] <= value <= region["hi"])


@pytest.fixture(scope="module")
def ccsd_t_map():
    tools, base = _tools(_h4())
    node = _method(tools, base["plan_id"], "ccsd_t")
    return tools.call("critic", "algorithm.validity", {"plan_id": base["plan_id"],
                                                       "node_id": node})["data"]


def test_v1c_ccsd_t_on_a_stretching_h4_chain_is_valid_then_invalid(ccsd_t_map) -> None:
    axis = ccsd_t_map["axes"][0]
    assert axis["axis"] == "geometry_scale" and axis["range"] == [0.7, 2.5]
    assert _region(axis, 1.0)["status"] in ("sampled_valid", "certified")
    assert _region(axis, 0.8)["status"] in ("sampled_valid", "certified")
    assert _region(axis, 2.5)["status"] == "invalid"
    assert ccsd_t_map["task_point"]["status"] in ("sampled_valid", "certified")
    valid_hi = max(region["hi"] for region in axis["regions"]
                   if region["status"] in ("sampled_valid", "certified") and region["lo"] < 1.5)
    invalid_lo = min(region["lo"] for region in axis["regions"] if region["status"] == "invalid")
    assert 1.3 <= valid_hi <= invalid_lo <= 1.7, (valid_hi, invalid_lo)
    assert invalid_lo - valid_hi <= 0.1 + 1e-9
    assert all(point["certified_bound"] is None for point in axis["points"])  # no state vector
    assert axis["counterexamples"]


def test_v1c_counterexamples_are_real(ccsd_t_map) -> None:
    for example in ccsd_t_map["axes"][0]["counterexamples"][:2]:
        independent = _pyscf_error(example["value"] * 1.0)  # scale 1.0 is 1.0 angstrom spacing
        assert abs(independent) > BAR
        assert abs(abs(example["error"]) - abs(independent)) < 1e-5


def test_v1c_variational_results_carry_a_sound_certificate() -> None:
    tools, base = _tools(_h4())
    built = tools.call("designer", "plan.candidates", {"plan_id": base["plan_id"], "candidates": [
        {"template": "sci", "select_cutoff": 1e-4}]})["data"]
    node = next(iter(built["ids"].values()))
    mapped = tools.call("critic", "algorithm.validity", {"plan_id": built["plan_id"],
                                                         "node_id": node})["data"]
    points = mapped["axes"][0]["points"]
    assert points and all(point["certified_bound"] is not None for point in points)
    for point in points:
        assert point["certified_bound"] + 1e-10 >= abs(point["error"]), point
    assert mapped["task_point"]["status"] in ("certified", "sampled_valid")


def test_v1c_the_domain_comes_from_the_signed_task() -> None:
    signed, _ = _tools(_h4())  # revised after R134: the tool is offered only with a domain
    schema = next(item for item in signed.schemas("critic") if item["name"] == "algorithm.validity")
    assert set(schema["parameters"]["properties"]) == {"plan_id", "node_id"}
    tools, base = _tools(_h4(validity=False))
    node = _method(tools, base["plan_id"], "ccsd_t")
    with pytest.raises(Exception) as caught:
        tools.call("critic", "algorithm.validity", {"plan_id": base["plan_id"], "node_id": node})
    assert "validity" in str(caught.value).lower()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(_h4() | {"validity": [{"axis": "geometry_scale",
                                                       "range": [1.2, 2.5], "points": 5,
                                                       "resolution": 0.1}]})  # excludes 1.0


def test_v1c_the_critic_escalates_only_inside_the_valid_region() -> None:
    from chemflow.agent.critic import review

    candidate = {"id": "c", "method": "sqd", "relation": "variational_upper_bound",
                 "relation_status": "proved", "evidence": True, "gate_passed": True,
                 "error": 1e-6, "cost": {"wall_s": 0.0}, "selection_work": 1}
    control = {"id": "n6", "method": "ccsd_t", "relation": "approximate",
               "relation_status": "proved", "evidence": True, "gate_passed": False,
               "error": 1e-2, "cost": {"wall_s": 1.0}, "selection_work": 10 ** 6}
    assert review([candidate | {"validity_status": "invalid"}], [control]).decision != "escalate"
    assert review([candidate | {"validity_status": "unknown"}], [control]).decision != "escalate"
    assert review([candidate | {"validity_status": "sampled_valid"}],
                  [control]).decision == "escalate"


def test_v1c_reports_state_the_valid_and_invalid_ranges(tmp_path) -> None:
    import json
    from argparse import Namespace

    import yaml
    from pyscf import fci, gto, scf

    from chemflow.eval import runner

    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", verbose=0)
    exact = fci.FCI(scf.RHF(mol).run()).kernel()[0]
    suite = {"name": "validity", "tasks": [{
        "id": "h4-validity", "category": "molecular", "spec": _h4(),
        "expected": {"value_hartree": float(exact)},
        "reference_source": "Reviewer PySCF RHF then FCI, linear H4 1.0 angstrom, STO-3G"}]}
    path = tmp_path / "validity.yaml"
    path.write_text(yaml.safe_dump(suite))
    out = tmp_path / "out"
    results = runner.run(Namespace(suite=None, suite_file=path, tasks=None, policy="scripted",
                                   model=None, roles="designer", jobs=1, out=out))
    assert results["rows"][0]["correct"]
    report = (out / "h4-validity" / "report.md").read_text()
    assert "Validity" in report and "geometry_scale" in report
    entries = [json.loads(line) for line in
               (out / "h4-validity" / "journal.jsonl").read_text().splitlines()]
    critique = next(entry for entry in entries if entry["state"] == "CRITIQUE")
    assert all("validity_status" in item for item in critique["observation"]["candidates"])
