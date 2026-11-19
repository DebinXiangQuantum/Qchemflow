from __future__ import annotations

import json
import sqlite3
import sys
from copy import deepcopy

import pytest

from chemflow.agent import cli as agent_cli
from chemflow.agent.task import TaskSpec
from chemflow.cli import main
from chemflow.signing import TaskSigner, generate_keypair

SPEC = {
    "question": "Screen H2", "target": {
        "molecule": {"atoms": "H 0 0 0; H 0 0 0.74"},
        "electronic_model": {"basis": "sto-3g"},
        "active": {"n_elec": 2, "n_orb": 2},
        "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "target_spin": 0},
    "gates": [{"kind": "abs_error_vs_exact", "threshold": 0.0016, "unit": "hartree"}],
    "budget": {"wall_s": 600, "sim_shots": 100000},
    "allowed_effects": ["pure", "simulator"], "decision": "Choose a method"}


def signed(tmp_path, spec=None):
    private, public = generate_keypair(b"human")
    record = TaskSigner(private, b"human").freeze(TaskSpec.model_validate(spec or SPEC), "human")
    task, key = tmp_path / "task.json", tmp_path / "public.pem"
    task.write_text(record.model_dump_json())
    key.write_bytes(public)
    return record, task, key


def invoke(monkeypatch, task, key, out, *extra):
    monkeypatch.setattr(sys, "argv", ["chemflow", "agent", "run", "--task", str(task),
                        "--public-key", str(key), "--out", str(out), *extra])
    main()


def test_budget_accumulates_and_second_run_refuses(tmp_path, monkeypatch, capsys):
    spec = deepcopy(SPEC)
    spec["budget"]["sim_shots"] = 4000
    record, task, key = signed(tmp_path, spec)
    out = tmp_path / "run"
    invoke(monkeypatch, task, key, out)
    ledger = out / f"{record.content_hash}.ledger.json"
    first = json.loads(ledger.read_text())
    assert 0 < first["sim_shots"] <= 4000
    summary = json.loads((out / "summary.json").read_text())
    assert summary["spent"] == first and summary["limits"] == spec["budget"]
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, task, key, out)
    assert error.value.code != 0
    assert "CF-E070" in capsys.readouterr().err
    assert json.loads(ledger.read_text())["sim_shots"] >= first["sim_shots"]


@pytest.mark.parametrize("extra,environment,message", [
    (["--policy", "openai"], {}, "--model"),
    (["--policy", "openai", "--model", "fake"], {}, "CHEMFLOW_LIVE_LLM"),
    (["--policy", "openai", "--model", "fake"], {"CHEMFLOW_LIVE_LLM": "1"}, "OPENAI_API_KEY"),
    (["--policy", "anthropic"], {"CHEMFLOW_LIVE_LLM": "1"}, "ANTHROPIC_API_KEY"),
    (["--roles", "designer,unknown"], {}, "--roles"),
    (["--rounds", "0"], {}, "--rounds"),
])
def test_preflight_refuses_without_output(tmp_path, monkeypatch, capsys, extra, environment, message):
    _, task, key = signed(tmp_path)
    for name in ("CHEMFLOW_LIVE_LLM", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "run"
    with pytest.raises(SystemExit):
        invoke(monkeypatch, task, key, out, *extra)
    assert message in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_live_policy_uses_injected_client(tmp_path, monkeypatch, provider):
    record, task, key = signed(tmp_path)
    calls = []

    class Client:
        def complete(self, messages, tools):
            if messages[-1]["role"] == "tool":
                assert json.loads(messages[-1]["content"])["result"]["data"]["plan_id"]
                return {"content": "SCI plan ready", "tool_calls": []}
            context = json.loads(messages[-1]["content"])
            assert "plan.candidates" in {tool["name"] for tool in tools}
            return {"content": "Screen SCI", "tool_calls": [{"name": "plan.candidates",
                "arguments": {"plan_id": context["observation"]["base_plan_id"],
                              "candidates": [{"template": "sci", "select_cutoff": 0.01}]}}]}

    def factory(provider, role, model):
        calls.append((provider, role, model))
        return Client()

    monkeypatch.setattr(agent_cli, "client_factory", factory)
    monkeypatch.setenv("CHEMFLOW_LIVE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    out = tmp_path / "run"
    invoke(monkeypatch, task, key, out, "--policy", provider, "--model", "fake-model")
    assert calls == [(provider, "designer", "fake-model")]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["record_hash"] == record.hash and summary["results"]
    assert summary["policy"] == provider and summary["roles"] == ["designer"]
    journal = [json.loads(line) for line in (out / "journal.jsonl").read_text().splitlines()]
    assert any(entry["state"] == "LLM_TOOL" for entry in journal)
    assert not any(entry["state"] == "DESIGN_REFUSED" for entry in journal)


def test_reaction_energy_end_to_end(tmp_path, monkeypatch):
    spec = deepcopy(SPEC)
    species = {"name": "reactant", "molecule": spec["target"]["molecule"], "coefficient": -1,
               "active": spec["target"]["active"], "sector": spec["target"]["sector"],
               "target_spin": 0}
    spec["target"] = {"kind": "reaction_energy", "electronic_model": {"basis": "sto-3g"},
                      "species": [species, {**species, "name": "product", "coefficient": 1,
                          "molecule": {"atoms": "H 0 0 0; H 0 0 1.0"}}]}
    record, task, key = signed(tmp_path, spec)
    out = tmp_path / "run"
    invoke(monkeypatch, task, key, out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["record_hash"] == record.hash and summary["results"]
    assert (out / "report.md").read_text().strip()
    with sqlite3.connect(out / "results.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM results WHERE operator='reaction_energy'"
                                  ).fetchone()[0] >= 2
    assert (out / "results.sqlite").stat().st_size > 0


def test_unsigned_task_creates_nothing(tmp_path, monkeypatch, capsys):
    _, task, key = signed(tmp_path)
    task.write_text(json.dumps(SPEC))
    out = tmp_path / "run"
    with pytest.raises(SystemExit):
        invoke(monkeypatch, task, key, out)
    assert "CF-E120" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("target", [
    {"kind": "lattice_ground_energy", "lattice": {"model": "hubbard", "geometry": "ring",
        "n_sites": 6, "t": 1.0, "U": 4.0, "unit": "hartree"},
     "sector": {"n_alpha": 3, "n_beta": 3, "n_orb": 6}, "target_spin": 0},
    {"kind": "spectral_function", "model": {"impurity_model": {
        "U": 2.0, "eps_bath": [0.0], "t_bath": [0.5], "unit": "hartree"}},
     "sector": {"n_alpha": 1, "n_beta": 1, "n_orb": 2}, "orbital": 0, "spin": "alpha",
     "omega": {"start": -4.0, "stop": 4.0, "n": 101, "unit": "hartree"}, "eta": 0.1},
    {"kind": "interaction_energy", "complex": {
        "atoms": "H 0 0 0; H 0 0 0.74; H 0 0 3.0; H 0 0 3.74"},
     "fragments": [[0, 1], [2, 3]], "electronic_model": {"basis": "sto-3g"}, "counterpoise": True},
])
def test_other_task_kinds(tmp_path, monkeypatch, target):
    spec = deepcopy(SPEC)
    spec["target"] = target
    if target["kind"] == "spectral_function":
        spec["gates"] = [{"kind": "l1_error_vs_exact", "threshold": 0.1, "unit": "dimensionless"}]
    record, task, key = signed(tmp_path, spec)
    out = tmp_path / "run"
    invoke(monkeypatch, task, key, out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["record_hash"] == record.hash and summary["results"]
    assert (out / "report.md").read_text().strip()
    assert any(result.get("screening_reference") for result in summary["results"])
