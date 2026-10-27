"""Run signed evaluations through the public agent entry point and score artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import secrets
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemflow.agent import cli as agent_cli
from chemflow.agent.llm import DEFAULT_MAX_TURNS
from chemflow.agent.llm_errors import is_quota
from chemflow.agent.task import TaskSpec
from chemflow.relations import TargetRelation, satisfies
from chemflow.signing import TaskSigner, generate_keypair
from chemflow.units import Q

RESOURCES = ("wall_s", "sim_shots", "qpu_s", "qpu_shots")
INTEGRITY = re.compile(r"\bCF-E(?:060|120|130)\b")
ERROR_LIMIT = 2000
logger = logging.getLogger(__name__)


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_hartree: float | None = Field(default=None, allow_inf_nan=False)
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    values_hartree: list[Annotated[float, Field(allow_inf_nan=False)]] | None = Field(
        default=None, min_length=1)
    ranking: list[str] | None = None
    site_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] | None = None
    decision_in: list[Literal["stop", "escalate", "revise", "refused"]] | None = Field(
        default=None, min_length=1)
    outcome: Literal["answer", "refused"] = "answer"
    refusal_code: str | None = Field(default=None, pattern=r"^CF-E\d{3}$")

    @model_validator(mode="after")
    def consistent_outcome(self):
        if (self.value is None) != (self.unit is None):
            raise ValueError("expected.value requires expected.unit")
        if self.value is not None:
            Q(self.value, self.unit)
            if self.value_hartree is not None:
                raise ValueError("Choose expected.value or expected.value_hartree")
        if self.outcome == "refused":
            if self.refusal_code is None:
                raise ValueError("A refusal outcome requires refusal_code")
        elif (sum(value is not None for value in
                  (self.value_hartree, self.value, self.values_hartree, self.ranking)) != 1 or
              self.refusal_code):
            raise ValueError("An answer requires a value and no refusal_code")
        if self.values_hartree is not None and (not self.values_hartree or
                not all(math.isfinite(value) for value in self.values_hartree)):
            raise ValueError("values_hartree must be finite and nonempty")
        if self.ranking is not None and (not self.ranking or
                                        len(set(self.ranking)) != len(self.ranking)):
            raise ValueError("ranking must be nonempty and unique")
        return self


class EvaluationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    category: Literal["molecular", "reaction", "lattice", "spectral", "interaction", "thermo",
                      "selection", "refusal", "property", "bond", "ranking", "path", "binding"]
    spec: TaskSpec
    expected: Expected
    reference_source: str = Field(min_length=1, pattern=r"\S")
    difficulty: Literal["easy", "medium", "hard"] | None = None

    @property
    def accuracy_threshold(self) -> float | None:
        spectral = self.spec.target.kind == "spectral_function"
        kind = "l1_error_vs_exact" if spectral else "abs_error_vs_exact"
        unit = self.expected.unit or ("dimensionless" if spectral else "hartree")
        thresholds = [Q(gate.threshold, gate.unit).to(unit).value
                      for gate in self.spec.gates if gate.kind == kind]
        if any(not math.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("Task accuracy thresholds must be finite and nonnegative")
        return min(thresholds) if thresholds else None

    @model_validator(mode="after")
    def declared_accuracy_gate(self):
        if self.expected.site_scores is not None and self.spec.target.kind != "metabolic_sites":
            raise ValueError("Site-score expectations require a metabolic_sites target")
        if self.accuracy_threshold is None and self.expected.outcome != "refused":
            raise ValueError("An answer task requires a target-compatible accuracy gate")
        return self


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    tasks: list[EvaluationTask] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Suite task ids must be unique")
        if self.name == "full" and any(task.difficulty is None for task in self.tasks):
            raise ValueError("Every full-suite task requires a difficulty")
        return self


def list_suites() -> list[str]:
    return sorted(path.name.removesuffix(".yaml") for path in files(
        "chemflow.eval.suites").iterdir() if path.name.endswith(".yaml"))


def load_suite(name: str | None = None, path: Path | None = None) -> Suite:
    if (name is None) == (path is None):
        raise ValueError("Supply exactly one of --suite or --suite-file")
    if path is None:
        if name not in list_suites():
            raise ValueError(f"Unknown suite: {name}")
        path = files("chemflow.eval.suites").joinpath(f"{name}.yaml")
    return Suite.model_validate(yaml.safe_load(path.read_text()))


def select_tasks(suite: Suite, selection: str | None) -> list[EvaluationTask]:
    if selection is None:
        return suite.tasks
    identifiers = [item.strip() for item in selection.split(",")]
    unknown = set(identifiers) - {task.id for task in suite.tasks}
    if unknown or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Unknown or duplicate task ids: {', '.join(sorted(unknown)) or selection}")
    return [task for task in suite.tasks if task.id in identifiers]


def _headline_value(task: EvaluationTask, headline: dict | None,
                    spectral_values: dict[str, float]) -> float | list[float] | None:
    if headline is None:
        return None
    if task.expected.values_hartree is not None:
        values = headline.get("values_hartree")
        return (values if isinstance(values, list) and
                all(type(value) in (float, int) and math.isfinite(value) for value in values)
                else None)
    if task.spec.target.kind == "spectral_function":
        value = spectral_values.get(headline.get("node"))
    else:
        try:
            number, unit = headline["value"].split()
            value = Q(float(number), unit).to(task.expected.unit or "hartree").value
        except (KeyError, ValueError, TypeError):
            return None
    return value if value is not None and math.isfinite(value) else None


def score_task(task: EvaluationTask, summary: dict | None, journal: list[dict],
               *, spectral_values: dict[str, float] | None = None) -> dict:
    summary = summary or {}
    headlines = summary.get("results", [])
    reference = next((item for item in headlines if item.get("screening_reference")), None)
    answer_node = summary.get("candidate")
    answers = [item for item in headlines if answer_node is not None and
               item.get("node") == answer_node]
    answer = answers[0] if len(answers) == 1 else None
    value = _headline_value(task, answer, spectral_values or {})
    reference_value = _headline_value(task, reference, spectral_values or {})
    threshold = task.accuracy_threshold

    def matches(actual):
        expected = (task.expected.value if task.expected.value is not None else
                    task.expected.value_hartree)
        return (actual is not None and expected is not None and
                threshold is not None and
                abs(actual - expected) <= threshold)

    decision = summary.get("decision", "failed")
    refusal = summary.get("refusal") or {}
    snapshot = refusal.get("spent_at_refusal")
    unspent = (isinstance(snapshot, dict) and bool(snapshot) and
               all(amount is not None and math.isfinite(amount) and amount >= 0
                   for amount in snapshot.values()) and snapshot == summary.get("spent"))
    refused = decision == "refused" and bool(refusal.get("code"))
    correct = (refused and refusal.get("code") == task.expected.refusal_code and
               refusal.get("nothing_consumed") is True and unspent
               if task.expected.outcome == "refused" else matches(value))
    completed = refused if task.expected.outcome == "refused" else value is not None
    reference_correct = matches(reference_value)
    if task.expected.site_scores is not None:
        def sites_match(headline):
            sites = (headline or {}).get("sites", [])
            if not isinstance(sites, list) or any(not isinstance(site, dict) or
                    type(site.get("atom")) is not int or "score" not in site for site in sites):
                return False
            actual = {str(site["atom"]): site["score"] for site in sites}
            expected = task.expected.site_scores
            return (actual.keys() == expected.keys() and len(actual) == len(sites) and
                    threshold is not None and all(type(actual[key]) in (int, float) and
                    math.isfinite(actual[key]) and abs(actual[key] - value) <= threshold
                    for key, value in expected.items()))

        correct = correct and sites_match(answer)
        reference_correct = reference_correct and sites_match(reference)
    if task.expected.outcome == "answer" and (
            task.expected.values_hartree is not None or task.expected.ranking is not None):
        field = "ranking" if task.expected.ranking is not None else "values_hartree"
        expected = getattr(task.expected, field)

        def collection_matches(headline):
            actual = (headline or {}).get(field)
            if not isinstance(actual, list) or len(actual) != len(expected):
                return False
            if field == "ranking":
                return actual == expected
            return threshold is not None and all(
                type(number) in (int, float) and math.isfinite(number) and
                abs(number - target) <= threshold for number, target in zip(actual, expected))

        completed = (isinstance((answer or {}).get(field), list) if field == "ranking"
                     else value is not None)
        correct = collection_matches(answer)
        reference_correct = collection_matches(reference)
    relation = answer.get("relation") if answer else None
    required = [gate.relation for gate in task.spec.gates if gate.kind == "relation_at_least"]
    relation_ok = (None if answer is None else
                   relation in {item.value for item in TargetRelation} and
                   all(satisfies(TargetRelation(relation), minimum) for minimum in required))
    events = [code for entry in journal for code in sorted(set(
        INTEGRITY.findall(json.dumps(entry))))]
    llm_failed = any(entry.get("state") == "LLM_FAILED" for entry in journal)
    quota = next((entry.get("decision", {}) for entry in journal
                  if entry.get("state") == "LLM_FAILED" and
                  (entry.get("decision", {}).get("quota") is True or
                   is_quota(entry.get("decision", {}).get("code")))), None)
    design_states = {"SCREEN": "screened", "DESIGN_DECLINED": "declined",
                     "DESIGN_REFUSED": "refused"}
    design = next((design_states[entry["state"]] for entry in reversed(journal)
                   if entry.get("state") in design_states), None)
    answer_from_candidate = answer_node is not None and any(
        entry.get("state") == "SCREEN" and answer_node in entry.get("outputs", [])
        for entry in journal)
    return {
        "id": task.id, "category": task.category,
        "llm_failed": llm_failed, "valid": not llm_failed,
        **({"quota": True, "quota_code": quota.get("code"),
            "reset_at": quota.get("reset_at")} if quota is not None else {}),
        **({"difficulty": task.difficulty} if task.difficulty is not None else {}),
        "completed": completed,
        "answer_node": answer_node, "answer_value": value,
        "design": design, "answer_from_candidate": answer_from_candidate,
        "reference_correct": reference_correct, "correct": correct,
        "gate_passed": bool(answer and answer.get("gate_verdict") == "PASS"),
        "relation": relation, "relation_ok": relation_ok, "integrity_events": events,
        "decision": decision,
        "decision_ok": (decision in task.expected.decision_in
                        if task.expected.decision_in is not None else None),
        "refusal_code": refusal.get("code") if refused else None,
        **{resource: summary.get("spent", {}).get(resource) for resource in RESOURCES},
        "llm_calls": sum(entry.get("state") in ("LLM_CALL", "LLM_ERROR") for entry in journal),
        "tool_calls": sum(len(entry.get("tool_calls", [])) for entry in journal),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    return {"n": len(rows), "invalid": sum(not row["valid"] for row in rows),
            "not_run": sum(row["decision"] == "not_run" for row in rows),
            "errors": sum(row["decision"] == "error" for row in rows),
            "correct_valid": sum(row["correct"] for row in rows if row["valid"]),
            **{field: sum(row[field] for row in rows) for field in
            ("completed", "correct", "reference_correct", "gate_passed",
             "llm_calls", "tool_calls")},
            "relation_ok": sum(row["relation_ok"] is True for row in rows),
            "relation_rated": sum(row["relation_ok"] is not None for row in rows),
            **{f"design_{status}": sum(row.get("design") == status for row in rows)
               for status in ("screened", "declined", "refused")},
            "answer_from_candidate": sum(row.get("answer_from_candidate") is True for row in rows),
            "decision_ok": sum(row["decision_ok"] is True for row in rows),
            "integrity_events": sum(len(row["integrity_events"]) for row in rows),
            **{resource: (None if any(row[resource] is None for row in rows) else
                          sum(row[resource] for row in rows)) for resource in RESOURCES}}


def _spectral_values(directory: Path, summary: dict) -> dict[str, float]:
    if not (directory / "results.sqlite").exists():
        return {}
    connection = sqlite3.connect(f"{(directory / 'results.sqlite').as_uri()}?mode=ro", uri=True)
    try:
        values = {}
        for headline in summary.get("results", []):
            row = connection.execute("SELECT value FROM results WHERE id=? AND record_hash=? "
                                     "AND node_id=? AND unit='dimensionless'",
                                     (headline.get("result_id"), summary["record_hash"],
                                      headline["node"])).fetchone()
            if row:
                values[headline["node"]] = row[0]
        return values
    finally:
        connection.close()


def write_report(results: dict, out: Path) -> None:
    metadata = results["metadata"]
    aggregate = results["aggregate"]
    lines = ["# ChemFlow agent evaluation", "",
             (f"Suite: {metadata['suite']}; policy: {metadata['policy']}; "
             f"model: {metadata['model'] or 'provider default / not applicable'}; "
             f"roles: {metadata['roles']}"), f"Git commit: {metadata['git_commit']}", "",
             (f"Policy accuracy: {aggregate['correct_valid']}/"
              f"{aggregate['n'] - aggregate['invalid']} valid rows correct; "
              f"{aggregate['invalid']} invalid rows excluded."), "",
             ("Invalid rows (LLM_FAILED or not_run) are infrastructure/model failures or "
              "unstarted tasks, not policy decisions."),
             "Raw correctness and resource fields are retained for diagnostics only.", ""]
    quota = next((row for row in results["rows"] if row.get("quota")), None)
    if quota is not None:
        lines.extend([_quota_reason(quota), ""])
    columns = ["id", "valid", "llm_failed", "category", "difficulty", "completed",
               "answer_node", "answer_value", "correct", "design", "answer_from_candidate",
               "reference_correct", "gate_passed", "relation", "relation_ok", "integrity_events",
               "decision", "decision_ok", "refusal_code", *RESOURCES, "llm_calls", "tool_calls"]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] *
                  len(columns)) + " |"])
    for row in results["rows"]:
        lines.append("| " + " | ".join(str(row.get(column)) for column in columns) + " |")
    errors = [row for row in results["rows"] if row["decision"] == "error"]
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.extend([f"### {row['id']}", ""])
            lines.extend("> " + line for line in row["error"].splitlines())
            lines.append("")
    lines.extend(["", "## Aggregate", "", "| Metric | Total |", "| --- | --- |"])
    lines.extend(f"| {key} | {value} |" for key, value in results["aggregate"].items())
    (out / "report.md").write_text("\n".join(lines) + "\n")


def _run_task(task: EvaluationTask, directory: Path, public_path: Path,
              policy: str, model: str | None, roles: str,
              max_turns: int = DEFAULT_MAX_TURNS) -> dict:
    error = None
    with (directory / "agent.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            agent_cli.run(argparse.Namespace(task=directory / "task.json", public_key=public_path,
                          out=directory, policy=policy, model=model, roles=roles, rounds=2,
                          max_turns=max_turns))
        except SystemExit as exc:
            if exc.code != 2:
                raise
        except Exception as exc:
            logger.exception("Evaluation task %s failed", task.id)
            error = f"{type(exc).__name__}: {exc}"
            if len(error) > ERROR_LIMIT:
                error = error[:ERROR_LIMIT - 3] + "..."
    return _score_artifacts(task, directory, error)


def _score_artifacts(task: EvaluationTask, directory: Path, error: str | None = None) -> dict:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    journal_path = directory / "journal.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()
               if line.strip()] if journal_path.exists() else []
    scalars = (_spectral_values(directory, summary)
               if task.spec.target.kind == "spectral_function" else None)
    row = score_task(task, summary, journal, spectral_values=scalars)
    if error is not None:
        row.update(decision="error", completed=False, correct=False, error=error,
                   decision_ok=False if task.expected.decision_in is not None else None)
    return row


def _run_process(work: tuple, max_turns: int = DEFAULT_MAX_TURNS) -> dict:
    task, directory, public_path, policy, model, roles = work
    payload = {"task": task.model_dump(mode="json"), "directory": str(directory),
               "public_path": str(public_path), "policy": policy, "model": model, "roles": roles,
               "max_turns": max_turns}
    result = subprocess.run([sys.executable, "-m", "chemflow.eval._worker"],
                            input=json.dumps(payload), capture_output=True, text=True, check=False)
    if result.returncode:
        error = result.stderr.strip() or f"Evaluation worker exited with code {result.returncode}"
        if len(error) > ERROR_LIMIT:
            error = "..." + error[-(ERROR_LIMIT - 3):]
        return _score_artifacts(task, directory, error)
    return json.loads(result.stdout)


def _quota_reason(row: dict) -> str:
    reset = f"resets at {row['reset_at']}" if row.get("reset_at") else "reset time unknown"
    return f"Run stopped on provider quota ({row.get('quota_code') or 'usage limit'}); {reset}."


def _not_run(task: EvaluationTask, quota: dict) -> dict:
    row = score_task(task, None, [])
    row.update(decision="not_run", valid=False, completed=False, correct=False,
               error=_quota_reason(quota))
    return row


def run(args) -> dict:
    started = time.perf_counter()
    suite = load_suite(args.suite, args.suite_file)
    tasks = select_tasks(suite, args.tasks)
    jobs = getattr(args, "jobs", 1)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("--jobs must be a positive integer")
    agent_cli.validate_options(argparse.Namespace(**vars(args), rounds=2))
    max_turns = getattr(args, "max_turns", DEFAULT_MAX_TURNS)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output directory must be empty (no artifact reuse)")
    out.mkdir(parents=True, exist_ok=True)
    passphrase = secrets.token_bytes(32)
    private, public = generate_keypair(passphrase)
    signer = TaskSigner(private, passphrase)
    del private, passphrase
    public_path = out / "public.pem"
    public_path.write_bytes(public)
    work = []
    for task in tasks:
        directory = out / task.id
        directory.mkdir()
        task_path = directory / "task.json"
        task_path.write_text(signer.freeze(task.spec, "eval-harness").model_dump_json(indent=2))
        work.append((task, directory, public_path, args.policy, args.model, args.roles))
    quota = None
    if jobs == 1:
        rows = []
        for item in work:
            row = (_not_run(item[0], quota) if quota is not None else
                   _run_task(*item, max_turns=max_turns))
            rows.append(row)
            if row.get("quota") and quota is None:
                quota = row
    else:
        quota_lock = Lock()

        def run_if_available(item):
            nonlocal quota
            with quota_lock:
                if quota is not None:
                    return _not_run(item[0], quota)
            row = _run_process(item, max_turns=max_turns)
            if row.get("quota"):
                with quota_lock:
                    if quota is None:
                        quota = row
            return row

        with ThreadPoolExecutor(max_workers=min(jobs, len(work))) as pool:
            pending = {}
            by_index = {}
            next_index = 0
            while pending or (quota is None and next_index < len(work)):
                while quota is None and len(pending) < jobs and next_index < len(work):
                    future = pool.submit(run_if_available, work[next_index])
                    pending[future] = next_index
                    next_index += 1
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future.result()
                    by_index[pending.pop(future)] = row
            rows = [by_index[index] if index in by_index else _not_run(item[0], quota)
                    for index, item in enumerate(work)]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    results = {"metadata": {"suite": suite.name, "policy": args.policy,
                            "model": args.model or (agent_cli.DEFAULT_GLM_MODEL
                                                     if args.policy == "glm" else None),
                            "roles": args.roles, "git_commit": commit, "jobs": jobs,
                            "max_turns": max_turns,
                            "elapsed_s": time.perf_counter() - started},
               "rows": rows, "aggregate": aggregate_rows(rows)}
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    write_report(results, out)
    return results
