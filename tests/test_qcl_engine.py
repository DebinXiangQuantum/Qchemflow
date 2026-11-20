"""The QCL engine answers signed tasks: screens, critic decisions and refusals."""

import argparse
import json

import pytest

from chemflow.agent import cli as agent_cli
from chemflow.agent import screen
from chemflow.eval.runner import load_suite
from chemflow.eval.runner import run as run_suite
from chemflow.signing import TaskSigner, generate_keypair

pytest.importorskip("qcl")


def _evaluate(tmp_path, suite, tasks):
    args = argparse.Namespace(suite=suite, suite_file=None, tasks=",".join(tasks),
                              policy="scripted", model=None, roles="designer", jobs=1,
                              out=tmp_path / suite, max_turns=8, engine="qcl")
    return {row["id"]: row for row in run_suite(args)["rows"]}


def test_engine_selection_defaults_to_qcl_for_scripted_policies():
    assert agent_cli.engine(argparse.Namespace(policy="scripted")) == "qcl"
    assert agent_cli.engine(argparse.Namespace(policy="anthropic", engine="auto")) == "legacy"
    assert agent_cli.engine(argparse.Namespace(policy="anthropic", engine="qcl")) == "qcl"
    assert agent_cli.engine(argparse.Namespace(policy="scripted", engine="legacy")) == "legacy"


def test_ground_energy_screen_escalates_a_passing_candidate(tmp_path):
    rows = _evaluate(tmp_path, "core", ["h2-ground", "h2-atomization", "h2-budget-refusal"])
    ground = rows["h2-ground"]
    assert ground["correct"] and ground["decision"] == "escalate"
    assert ground["relation"] == "variational_upper_bound"
    summary = json.loads((tmp_path / "core" / "h2-ground" / "summary.json").read_text())
    methods = {item["node"]: item for item in summary["results"]}
    assert summary["engine"] == "qcl" and methods[summary["candidate"]]["method"] == "sci"
    assert summary["candidate"].startswith("candidate")
    assert methods["fci"]["screening_reference"] and methods["fci"]["relation"] == "equivalent"
    assert methods["ccsd"]["gate_verdict"].startswith("FAIL (Relation")
    # Classical controls are answered by the classical ladder; reactions stop on one.
    assert rows["h2-atomization"]["correct"] and rows["h2-atomization"]["decision"] == "stop"
    refusal = rows["h2-budget-refusal"]
    assert refusal["correct"] and refusal["refusal_code"] == "CF-E070"


def test_refusals_consume_nothing(tmp_path):
    rows = _evaluate(tmp_path, "full", ["refuse-effects", "refuse-sim-shots"])
    assert rows["refuse-effects"]["refusal_code"] == "CF-E120"
    assert rows["refuse-sim-shots"]["refusal_code"] == "CF-E070"
    assert all(row["correct"] for row in rows.values())


def test_application_targets_run_on_qcl(tmp_path):
    rows = _evaluate(tmp_path, "apps", ["frontier_orbitals", "bond_dissociation",
                                        "lattice_superconductivity"])
    assert all(row["correct"] and row["decision_ok"] for row in rows.values())
    assert rows["bond_dissociation"]["relation"] == "approximate"


def test_dry_run_compiles_one_screen_plan(tmp_path):
    task = next(item for item in load_suite("core").tasks if item.id == "h2-ground")
    private, public = generate_keypair(b"p" * 32)
    record = TaskSigner(private, b"p" * 32).freeze(task.spec, "test")
    (tmp_path / "task.json").write_text(record.model_dump_json())
    (tmp_path / "public.pem").write_bytes(public)
    agent_cli.run(argparse.Namespace(task=tmp_path / "task.json",
                                     public_key=tmp_path / "public.pem", out=tmp_path / "out",
                                     policy="scripted", model=None, roles="designer",
                                     rounds=2, max_turns=8, engine="qcl", dry_run=True))
    review = json.loads((tmp_path / "out" / "qcl-review.json").read_text())
    names = [item["name"] for item in review["outputs"]]
    assert names[:4] == list(screen.LADDER) and names[-1] == "fci"
    assert review["relations"]["fci"] == "equivalent"
    assert review["relations"]["candidate2"] == "variational_upper_bound"
    assert review["outputs"][6] == {"name": "candidate2", "role": "candidate",
                                    "method": "sqd", "params": {"shots": 1024}}
    assert review["sim_shots"] == 1024 + 2048


def test_automatic_active_space_from_an_application_draft(tmp_path):
    from chemflow.agent.task import TaskSpec
    from chemflow.apps import draft

    spec = draft("ground_state_energy",
                 {"molecule": {"atoms": "H 0 0 0; H 0 0 1.5; H 0 0 3.0; H 0 0 4.5"}})[0]
    assert spec["target"]["active"] == "auto"
    private, public = generate_keypair(b"p" * 32)
    record = TaskSigner(private, b"p" * 32).freeze(TaskSpec.model_validate(spec), "test")
    (tmp_path / "task.json").write_text(record.model_dump_json())
    (tmp_path / "public.pem").write_bytes(public)
    agent_cli.run(argparse.Namespace(task=tmp_path / "task.json",
                                     public_key=tmp_path / "public.pem", out=tmp_path / "out",
                                     policy="scripted", model=None, roles="designer",
                                     rounds=2, max_turns=8))
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["engine"] == "qcl" and summary["decision"] in ("stop", "escalate")
    journal = [json.loads(line) for line in
               (tmp_path / "out" / "journal.jsonl").read_text().splitlines()]
    chosen = next(entry for entry in journal if entry["state"] == "MODEL")
    active = chosen["decision"]["active"]
    assert 2 <= active["n_orb"] <= 10 and active["n_elec"] % 2 == 0
    fci = next(item for item in summary["results"] if item["node"] == "fci")
    assert fci["relation"] == "equivalent"


def _turn(name=None, arguments=None):
    return {"content": "", "tool_calls": [] if name is None else [
        {"id": f"call-{name}", "name": name, "arguments": arguments or {}}]}


def _signed(tmp_path, spec):
    private, public = generate_keypair(b"p" * 32)
    record = TaskSigner(private, b"p" * 32).freeze(spec, "test")
    (tmp_path / "public.pem").write_bytes(public)
    return record


def test_llm_designer_proposes_candidates_that_qcl_screens(tmp_path):
    import qcl
    from chemflow.agent.llm import FakeLLMClient, LLMPolicy
    from chemflow.agent.qcl_run import run as run_screens
    from chemflow.agent.qcl_tools import QclToolRegistry

    task = next(item for item in load_suite("core").tasks if item.id == "h4-selection-1p5")
    record = _signed(tmp_path, task.spec)
    candidates = [{"template": "sci", "select_cutoff": 1e-4}]
    designer = FakeLLMClient([_turn("methods.list"),
                              _turn("screen.review", {"candidates": candidates}),
                              _turn("design.submit", {"candidates": candidates})])
    analyst = FakeLLMClient([_turn("results.query"),
                             _turn("report.write", {"text": "Exact reference "
                                                    "-1.99615032532 hartree."}), _turn()])
    policies = {"designer": LLMPolicy("designer", designer, QclToolRegistry()),
                "analyst": LLMPolicy("analyst", analyst, QclToolRegistry())}
    summary = run_screens(argparse.Namespace(out=tmp_path / "out"), record,
                          qcl.Cluster.local(), policies=policies, policy="fake")
    assert summary["decision"] == "escalate" and summary["candidate"] == "candidate0"
    chosen = next(item for item in summary["results"] if item["node"] == "candidate0")
    assert chosen["method"] == "sci" and chosen["params"] == {"select_cutoff": 1e-4}
    assert summary["roles"] == ["analyst", "designer"]
    reviewed = designer.messages[2][-1]["content"]
    assert "variational_upper_bound" in reviewed and "plan_hash" in reviewed
    report = (tmp_path / "out" / "report.md").read_text()
    assert "## Analyst" in report and "-1.99615032532 hartree" in report
    states = [json.loads(line)["state"] for line in
              (tmp_path / "out" / "journal.jsonl").read_text().splitlines()]
    assert states.count("SCREEN") == 2 and "PROPOSE" in states


def test_llm_designer_cannot_submit_invented_methods(tmp_path):
    import qcl
    from chemflow.agent.llm import FakeLLMClient, LLMPolicy
    from chemflow.agent.qcl_run import run as run_screens
    from chemflow.agent.qcl_tools import QclToolRegistry

    task = next(item for item in load_suite("core").tasks if item.id == "h2-ground")
    record = _signed(tmp_path, task.spec)
    designer = FakeLLMClient([
        _turn("design.submit", {"candidates": [{"template": "vqe", "shots": 10}]}),
        _turn("design.submit", {"candidates": [{"template": "sci", "select_cutoff": 2.0}]}),
        _turn("design.decline", {"reason": "Controls suffice for H2"})])
    summary = run_screens(argparse.Namespace(out=tmp_path / "out"), record,
                          qcl.Cluster.local(),
                          policies={"designer": LLMPolicy("designer", designer,
                                                          QclToolRegistry())},
                          policy="fake")
    # Both invalid submissions were rejected by the tool; the decline keeps the controls,
    # and the reference answers because no classical control passes the relation gate.
    assert summary["decision"] == "stop" and summary["candidate"] == "fci"
    states = [json.loads(line)["state"] for line in
              (tmp_path / "out" / "journal.jsonl").read_text().splitlines()]
    assert "DESIGN_DECLINED" in states and states.count("SCREEN") == 1


def test_report_lint_rejects_unrecorded_numbers():
    from chemflow.agent.report_lint import ReportLintError, lint_numbers

    headlines = [{"value": "-1.13728383449 hartree"}, {"values_hartree": [0.2041203]}]
    lint_numbers("FCI gives -1.1373 hartree and 0.204 hartree.", headlines)
    with pytest.raises(ReportLintError, match="not a recorded"):
        lint_numbers("CCSD gives -1.2000 hartree.", headlines)
    with pytest.raises(ReportLintError, match="advantage"):
        lint_numbers("This shows quantum advantage.", headlines)


def test_chemflow_errors_survive_process_boundaries():
    import pickle

    from chemflow.diagnostics import AdapterError, EffectDenied

    error = pickle.loads(pickle.dumps(AdapterError("CF-E080", "n1", "Tools missing", "Install",
                                                   nothing_consumed=True)))
    assert error.diagnostics[0].code == "CF-E080" and error.nothing_consumed
    assert str(error) == "CF-E080: Tools missing"
    denied = pickle.loads(pickle.dumps(EffectDenied("n2", "qpu")))
    assert isinstance(denied, EffectDenied) and denied.diagnostics[0].node == "n2"
