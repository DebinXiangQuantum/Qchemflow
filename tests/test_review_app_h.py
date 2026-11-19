"""Reviewer tests for APP batch H: the application layer (written by Claude; implementers must
not edit).

User request (2026-10-15): the 13 applications must be supported, and "our agent can call
these tools to build various application". Batches A–G added the operators and target
kinds. H turns them into applications that a user or the planner agent can pick.

Contract:
- **Catalogue.** `chemflow.apps.catalog()` returns one entry per application:
  `{id, name_en, name_zh, description, target_kinds, inputs, parameters, example}`.
  - The ids and Chinese names are exactly those in `APPS` below.
  - `inputs` declares file or text kinds (xyz, pdb, sdf, smiles, lattice); `parameters`
    declares defaults.
- **Drafting.** `chemflow.apps.draft(app_id, inputs=None, parameters=None, example=False)`
  returns a non-empty list of TaskSpec dicts, unsigned drafts for human approval.
  - Each validates as `TaskSpec`, has a target kind in the app's `target_kinds`, and
    declares gates: an absolute-error bar and a relation_at_least gate.
  - Each declares a budget, plus `validity` where the app is an approximation map (PES).
  - `example=True` uses small bundled example inputs.
  - An app needing several tasks returns several, e.g. one per solvent.
  - Apps whose science needs a dedicated target kind get one, e.g. `metabolic_sites`
    wrapping site_reactivity as empirical evidence.
- **CLI.** `chemflow app list | describe <id> | draft <id> [--example] [--input path ...]
  [--param key=value ...] --out <dir>`. `draft` writes one task JSON per drafted task.
- **Eval suite `apps`.** One small task per application, with an independent reference
  (computed by `scripts/eval_reference_values.py`, never by ChemFlow's pipeline). The
  scripted policy answers every task correctly. The protein–ligand task may be the only
  row needing `chemflow-tools-env`; when that is absent it must fail as a structured
  CF-E080 error row, not crash.
- **Agent.** The planner role gets the tool `app.draft`, which returns drafts only; it can
  never sign. The planner's system prompt names the catalogue ids, so an LLM agent can
  start from an application.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHEMFLOW = str(Path(sys.executable).parent / "chemflow")
APPS = {
    "ground_state_energy": "分子基态能量",
    "stability_ranking": "多分子稳定性排序",
    "conformer_ranking": "多构象能量排序",
    "pes_scan": "势能面扫描",
    "metabolic_sites": "代谢位点预测",
    "bond_dissociation": "键解离能预测",
    "solvation_scan": "溶剂化自由能扫描",
    "frontier_orbitals": "前线轨道能量预测",
    "aqueous_solubility": "水溶度快速评估",
    "reaction_barrier": "反应能垒估算",
    "dimer_binding_scan": "二聚体结合能(刚性扫描)",
    "protein_ligand_binding": "蛋白质-配体结合能",
    "lattice_superconductivity": "超导材料模拟(格点模型)",
}


def test_h1_the_catalogue_has_the_thirteen_applications() -> None:
    from chemflow.apps import catalog

    entries = {entry["id"]: entry for entry in catalog()}
    assert set(entries) == set(APPS)
    for app_id, entry in entries.items():
        assert entry["name_zh"] == APPS[app_id], app_id
        assert entry["target_kinds"] and entry["inputs"] and entry["description"].strip()


@pytest.mark.parametrize("app_id", sorted(APPS))
def test_h2_every_application_drafts_valid_tasks(app_id) -> None:
    from chemflow.agent.task import TaskSpec
    from chemflow.apps import catalog, draft

    entry = next(item for item in catalog() if item["id"] == app_id)
    drafts = draft(app_id, example=True)
    assert drafts, app_id
    for task in drafts:
        spec = TaskSpec.model_validate(task)
        assert spec.target.kind in entry["target_kinds"], (app_id, spec.target.kind)
        kinds = {gate.kind for gate in spec.gates}
        assert "relation_at_least" in kinds and kinds & {"abs_error_vs_exact",
                                                         "l1_error_vs_exact"}, app_id
        assert spec.budget, app_id
    if app_id == "solvation_scan":
        assert len(drafts) >= 2  # one task per solvent
    if app_id == "pes_scan":
        assert any(TaskSpec.model_validate(task).validity for task in drafts)


def test_h3_the_cli_lists_and_drafts(tmp_path) -> None:
    listed = subprocess.run([CHEMFLOW, "app", "list"], capture_output=True, text=True,
                            check=False, timeout=300)
    assert listed.returncode == 0, listed.stderr
    for app_id, name in APPS.items():
        assert app_id in listed.stdout and name in listed.stdout
    out = tmp_path / "drafts"
    drafted = subprocess.run([CHEMFLOW, "app", "draft", "frontier_orbitals", "--example",
                              "--out", str(out)], capture_output=True, text=True,
                             check=False, timeout=300)
    assert drafted.returncode == 0, drafted.stderr
    from chemflow.agent.task import TaskSpec

    files = sorted(out.glob("*.json"))
    assert files
    for path in files:
        TaskSpec.model_validate(json.loads(path.read_text()))


def test_h4_the_apps_suite_is_answered_by_the_scripted_policy(tmp_path) -> None:
    completed = subprocess.run([CHEMFLOW, "eval", "run", "--suite", "apps", "--policy",
                                "scripted", "--jobs", "3", "--out", str(tmp_path / "out")],
                               capture_output=True, text=True, check=False, timeout=5400)
    assert completed.returncode == 0, completed.stderr[-3000:]
    rows = json.loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert len(rows) >= len(APPS)
    tools = (Path(__file__).resolve().parents[1].parent / "chemflow-tools-env" / "bin" /
             "python").exists()
    for row in rows:
        if row["category"] == "binding" and not tools:
            assert row["decision"] in ("error", "failed", "refused"), row
            continue
        assert row["correct"], (row["id"], row.get("error"), row["decision"])


def test_h5_the_planner_can_draft_from_the_catalogue() -> None:
    import test_review_m4a as m4a

    from chemflow.agent.llm import LLMPolicy

    tools = m4a._tools()
    planner = {schema["name"] for schema in tools.schemas("planner")}
    assert "app.draft" in planner
    drafted = tools.call("planner", "app.draft", {"app_id": "frontier_orbitals",
                                                  "example": True})["data"]
    assert drafted and all("signature" not in json.dumps(item) for item in drafted)

    class Capture:
        model = "capture"

        def __init__(self):
            self.messages = []

        def complete(self, messages, schemas):
            self.messages.append(messages)
            return {"tool_calls": [], "content": "done", "stop_reason": "stop", "usage": {}}

    client = Capture()
    LLMPolicy("planner", client, tools).decide({"task": {}})
    system = client.messages[0][0]["content"]
    assert all(app_id in system for app_id in APPS)
