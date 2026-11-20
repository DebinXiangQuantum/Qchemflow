"""Run a signed task as QCL screens: refuse, compile, execute, grade, decide, report.

This is the QCL engine behind ``chemflow agent run``. Each screening round is one QCL
plan (see :mod:`chemflow.agent.screen`) executed with lane overlap; its outputs are
graded against the exact in-model reference and the deterministic critic decides to
stop (answer with a method), escalate (a candidate passes where no affordable classical
control does) or revise (try the next candidate round).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any

from chemflow.agent import screen as screens
from chemflow.agent.critic import CriticDecision, decision_items, review
from chemflow.agent.journal import Journal
from chemflow.agent.memory import ResultsDB
from chemflow.diagnostics import AdapterError

MAX_ROUNDS = 2
# Predicted sequential seconds above which a screen runs its methods on parallel lanes.
OVERLAP_MIN_S = 2.0


class Refusal(Exception):
    def __init__(self, code: str, message: str, hint: str, node: str | None = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.hint, self.node = code, message, hint, node


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def _check_effects(spec: Any) -> None:
    if "pure" not in spec.allowed_effects:
        raise Refusal("CF-E120", "Effect pure not allowed by task", "Freeze a new task version")


def _remaining(spec: Any, ledger: Any) -> dict[str, float]:
    budget = spec.budget
    remaining = {key: budget[key] - (ledger.spent.get(key) or 0) for key in budget
                 if key in ledger.limits}
    if remaining.get("wall_s", 300) <= 0:
        raise Refusal("CF-E070", "wall_s budget exceeded",
                      "Reduce predicted cost to the remaining task budget")
    if "cpu_core_s" in remaining and remaining["cpu_core_s"] <= 0:
        raise Refusal("CF-E070", "cpu_core_s budget exceeded",
                      "Reduce predicted cost to the remaining task budget")
    return remaining


@contextmanager
def _signed_policy(directory: Any, cluster: Any, remaining: dict[str, float]):
    """The operator QPU policy of a signed task: its shot budget, local simulators only.

    QCL refuses to sample without an operator policy; for a signed task the signature is
    the operator's decision, so the policy is derived from it rather than from ambient
    configuration, and real QPUs stay unavailable to built-in screens.
    """
    simulators = [device.id for device in cluster.qpus if device.simulator_host]
    simulators.append(screens.CPU_SIMULATOR_ID)
    path = directory / "qcl-policy.toml"
    shots = int(max(0, remaining.get("sim_shots", 0)))
    path.write_text("# Derived from the signed task budget by chemflow agent run.\n"
                    f"[budget]\nqpu_shots = {shots}\nqpu_s = 0\n"
                    f"allowed_qpus = {json.dumps(simulators)}\n")
    previous = os.environ.get("QCL_POLICY_FILE")
    os.environ["QCL_POLICY_FILE"] = str(path)
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("QCL_POLICY_FILE", None)
        else:
            os.environ["QCL_POLICY_FILE"] = previous


def _execute(task_spec: Any, cluster: Any, built: Any, device_id: str | None,
             remaining: dict[str, float], round_index: int, active: Any,
             directory: Any) -> dict[str, Any]:
    """Run one screen; isolate failing methods by rerunning entries one at a time."""
    with _signed_policy(directory, cluster, remaining):
        return _execute_screen(task_spec, cluster, built, device_id, remaining,
                               round_index, active)


def _execute_screen(task_spec: Any, cluster: Any, built: Any, device_id: str | None,
                    remaining: dict[str, float], round_index: int,
                    active: Any) -> dict[str, Any]:
    started = time.monotonic()
    timeout = remaining.get("wall_s", 300)
    cpu = remaining.get("cpu_core_s")
    plan = built.workflow.compile(*built.args)
    relations = plan.relations()
    failures: dict[str, str] = {}
    plans = [plan]
    try:
        predicted = plan.schedule().get("predicted", {})
    except Exception:  # noqa: BLE001 - the schedule only decides whether to overlap
        predicted = {}
    # Lanes pay for themselves only when there is work to overlap.
    overlap = (predicted.get("sequential_s") or 0) >= OVERLAP_MIN_S
    try:
        values = plan.run(timeout_s=timeout, cpu_core_s=cpu, overlap=overlap)
    except Exception as error:
        if isinstance(error, TimeoutError) or any(
                limit in str(error) for limit in ("core-time limit", "wall-time limit")):
            raise
        values = {}
        relations = {}
        for entry in built.entries:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                failures[entry.name] = "wall_s budget exhausted before this method ran"
                continue
            single = screens.build(task_spec, cluster, round_index=round_index,
                                   device_id=device_id, active=active,
                                   controls=entry.role != "candidate",
                                   only={entry.name},
                                   proposed=tuple(item for item in built.entries
                                                  if item.role == "candidate"))
            single_plan = single.workflow.compile(*single.args)
            plans.append(single_plan)
            try:
                result = single_plan.run(timeout_s=timeout - elapsed, cpu_core_s=cpu)
            except Exception as failure:  # noqa: BLE001
                failures[entry.name] = f"{type(failure).__name__}: {failure}"[:500]
                continue
            values[entry.name] = result[entry.name] if isinstance(result, dict) else result
            relations[entry.name] = single_plan.relations()
            relations[entry.name] = (relations[entry.name][entry.name]
                                     if isinstance(relations[entry.name], dict)
                                     else relations[entry.name])
    elapsed = time.monotonic() - started
    profile = []
    cpu_seconds = 0.0
    gpu_seconds = 0.0
    for item in plans:
        try:
            records = item.profile()
        except RuntimeError:
            continue
        profile.extend(records)
        cpu_seconds += item.cpu_core_s or 0
        gpu_seconds += sum(record["duration_s"] for record in records
                           if str(record.get("device", "")).startswith("gpu"))
    if not isinstance(relations, dict):
        relations = {built.entries[0].name: relations}
    if not isinstance(values, dict):
        values = {built.entries[0].name: values}
    return {"plan": plan, "plan_hash": plan.hash, "values": values, "relations": relations,
            "failures": failures, "wall_s": elapsed, "cpu_core_s": cpu_seconds,
            "gpu_s": gpu_seconds, "profile": profile, "predicted": predicted,
            "overlap": overlap,
            "explain": plan.explain(format="json")}


def validity_map(spec: Any, cluster: Any, entries: list[Any], active: Any,
                 remaining: dict[str, float], spent: dict[str, float],
                 directory: Any) -> dict[str, dict]:
    """Map the signed validity domain for screened methods (batched QCL plans)."""
    from chemflow.agent.validity import map_methods

    def execute(combined: Any) -> dict[str, Any]:
        sim_left = remaining.get("sim_shots")
        if combined.sim_shots and sim_left is not None and combined.sim_shots > sim_left:
            raise Refusal("CF-E070", "sim_shots budget exceeded by the validity map",
                          "Reduce validity points or candidate shots")
        started = time.monotonic()
        plan = combined.workflow.compile()
        with _signed_policy(directory, cluster, remaining):
            try:
                values = plan.run(timeout_s=max(1.0, remaining.get("wall_s", 300)),
                                  cpu_core_s=remaining.get("cpu_core_s"),
                                  overlap=True)
            except Exception:  # noqa: BLE001 - unevaluable points are reported unknown
                values = {}
        elapsed = time.monotonic() - started
        for key, amount in (("wall_s", elapsed), ("cpu_core_s", plan.cpu_core_s or 0),
                            ("sim_shots", combined.sim_shots)):
            spent[key] = spent.get(key, 0) + amount
            if key in remaining:
                remaining[key] -= amount
        if remaining.get("wall_s", 1) <= 0:
            raise Refusal("CF-E070", "wall_s budget exhausted by the validity map",
                          "Reduce validity points or raise the wall-time budget")
        return values

    return map_methods(spec, cluster, entries, active=active, execute=execute)


def _reference_value(task_spec: Any, built: Any, run: dict[str, Any]) -> Any:
    target = task_spec.target
    if target.kind == "spectral_function":
        import qcl
        from chemflow.domains.fock_reference import reference

        exact = reference(target.model_dump(mode="json"), target.model["impurity_model"])
        return qcl.SpectralEstimate(tuple(exact.omega), tuple(exact.spectral), exact.unit,
                                    "equivalent", {}, {"implementation": "fock_reference"})
    return run["values"].get(built.reference)


def _headline(task_spec: Any, name: str, entry: Any, value: Any, graded: Any,
              reference: bool, built: Any = None) -> dict[str, Any]:
    import numpy as np

    import qcl

    target = task_spec.target
    headline: dict[str, Any] = {"node": name, "method": entry.method,
                                "role": entry.role, **({"params": entry.options}
                                                       if entry.params else {})}
    if isinstance(value, qcl.SpectralEstimate):
        headline["value"] = (f"{float(np.trapezoid(value.spectral, value.omega)):.12g} "
                             "dimensionless")
        headline["sum_rule"] = float(np.trapezoid(value.spectral, value.omega))
    elif isinstance(value, qcl.Ranking):
        values = screens.scalars(value, "hartree")
        headline.update(value=f"{values[value.order[0]]:.12g} hartree",
                        values_hartree=values,
                        ranking=[target.items[index].name for index in value.order])
    elif isinstance(value, qcl.SiteScores):
        top = max(value.sites, key=lambda site: site["score"]) if value.sites else None
        headline.update(value=f"{top['score']:.12g} dimensionless" if top else None,
                        sites=list(value.sites), top_site=value.top_site)
    elif isinstance(value, list):
        values = screens.scalars(value, "hartree")
        headline.update(value=f"{values[0]:.12g} hartree", values_hartree=values)
    elif isinstance(value, tuple):
        first = value[0]
        headline["value"] = f"{first.value:.12g} {first.unit}"
        if len(value) > 1 and isinstance(value[1], qcl.Estimate):
            headline["rate_constant"] = {"k_per_s": value[1].value, "unit": value[1].unit}
    elif isinstance(value, qcl.Estimate):
        headline["value"] = f"{value.value:.12g} {value.unit}"
    label = quantity_label(task_spec)
    if label:
        headline["quantity_label"] = label
    if graded.certificate:
        headline["relation_certificate"] = graded.certificate
    if graded.validity:
        headline.update(validity=graded.validity,
                        validity_status=graded.validity["task_point"]["status"])
    if built is not None:
        headline.update(deployment(entry, built))
    headline.update(relation=graded.relation or "unknown",
                    relation_status="proved" if graded.relation else "unknown",
                    gate_verdict="PASS" if graded.passed else f"FAIL ({graded.failure})",
                    error=graded.error if math.isfinite(graded.error) else None,
                    gates=graded.gates)
    if reference:
        headline["screening_reference"] = True
    return headline


def deployment(entry: Any, built: Any) -> dict[str, Any]:
    """Where a method could run in production: sampled candidates need a catalogued QPU."""
    from chemflow.backends import QPU_CATALOG

    if entry.method != "sqd":
        return {"deployable": True, "deployment": {}, "deployment_reason": "",
                "classical_nodes": [entry.name]}
    shots = entry.options.get("shots", 0)
    for implementation, limits in QPU_CATALOG["sample_configurations"].items():
        if (built.qubits <= limits["qubits"] and
                limits["min_shots"] <= shots <= limits["max_shots"]):
            return {"deployable": True, "deployment": {"sample": implementation},
                    "deployment_reason": "", "classical_nodes": []}
    return {"deployable": False, "deployment": {},
            "deployment_reason": f"No catalogued QPU fits {built.qubits} qubits and "
                                 f"{shots} shots", "classical_nodes": []}


def _item(graded: Any, built: Any) -> dict[str, Any]:
    entry = graded.entry
    shots = entry.options.get("shots", 0)
    return {"id": graded.name, "method": entry.method, "relation": graded.relation or "unknown",
            "relation_status": "proved" if graded.relation else "unknown",
            **({"relation_certificate": graded.certificate} if graded.certificate else {}),
            **({"validity_status": graded.validity["task_point"]["status"],
                "validity": graded.validity} if graded.validity else {}),
            "evidence": True, "gate_passed": graded.passed,
            "error": graded.error if math.isfinite(graded.error) else None,
            "cost": {"sim_shots": shots, "qpu_shots": 0},
            "selection_work": built.work.get(graded.name), **deployment(entry, built)}


def certify(plan: Any, name: str, relation: str | None, directory: Any) -> str | None:
    """Write the compiler's relation derivation for one output; returns its identifier.

    The certificate lists every node the output depends on with its op and the relation
    the primitive table assigns it, so the inference can be re-checked by hand.
    """
    from qcl._qcl import relations_json

    from chemflow.ids import content_hash

    if relation is None:
        return None
    graph = plan.graph_json()
    output = plan._output.get(name) if isinstance(plan._output, dict) else plan._output
    roots = [value.node_id for value in _values(output)]
    if not roots:
        return None
    by_node = json.loads(relations_json(json.dumps(graph)))
    nodes = {node["id"]: node for node in graph["nodes"]}
    seen, pending = set(), list(roots)
    while pending:
        identifier = pending.pop()
        if identifier in seen:
            continue
        seen.add(identifier)
        pending.extend(nodes[identifier]["inputs"])
    body = {"node": name, "relation": relation, "plan_hash": plan.hash,
            "method": "qcl-ir relation inference over primitives.toml",
            "outputs": roots,
            "derivation": [{"id": identifier, "op": nodes[identifier]["op"],
                            "inputs": nodes[identifier]["inputs"],
                            "relation": by_node[identifier]}
                           for identifier in sorted(seen) if by_node[identifier] is not None]}
    identifier = content_hash(body)
    folder = directory / "certificates"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{identifier}.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return identifier


def _values(output: Any) -> list[Any]:
    if hasattr(output, "node_id"):
        return [output]
    if isinstance(output, (list, tuple)):
        return [value for item in output for value in _values(item)]
    if isinstance(output, dict):
        return [value for item in output.values() for value in _values(item)]
    return []


def _refuse(args: Any, record: Any, journal: Journal, ledger: Any, refusal: Refusal,
            paths: dict[str, str], *, nothing_consumed: bool = True) -> None:
    detail = {"code": refusal.code, "node": refusal.node,
              "message": f"{refusal.code}: {refusal.message}", "fix_hints": [refusal.hint],
              "nothing_consumed": nothing_consumed, "spent_at_refusal": dict(ledger.spent)}
    journal.append(state="REFUSED", role="orchestrator", observation=detail,
                   decision="refused", reason=refusal.message)
    ResultsDB(args.out / "results.sqlite").connection.close()
    (args.out / "report.md").write_text(
        f"# QCL screen\n\nTask: {record.spec.question}\n\nDecision: refused "
        f"({refusal.code}: {refusal.message})\n")
    summary = {"record_hash": record.hash, "engine": "qcl", "decision": "refused",
               "candidate": None,
               "reason": ("Effect refusal" if refusal.code == "CF-E120" else
                          "Unsupported model or implementation" if refusal.code == "CF-E080"
                          else "Budget refusal"),
               "results": [], "refusal": detail, "policy": "scripted", "roles": [],
               "spent": dict(ledger.spent), "limits": ledger.limits,
               "paths": {**paths, "report": str(args.out / "report.md"),
                         "results": str(args.out / "results.sqlite")}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{refusal.code}: {refusal.message}", file=sys.stderr)
    raise SystemExit(2)


def run(args: Any, record: Any, cluster: Any, *, device_id: str | None = None,
        policies: dict[str, Any] | None = None, policy: str = "scripted",
        proposals: list[dict[str, Any]] | None = None) -> dict:
    """Answer a signed task. ``policies`` maps LLM roles (modeler, designer, critic,
    analyst) to :class:`chemflow.agent.llm.LLMPolicy` objects; absent roles are scripted.
    ``proposals`` (typed sci/sqd candidates, as ``design.submit``) replace the scripted
    designer's rounds with one round of the caller's candidates (the MCP ``screen.run``).
    """
    from chemflow.agent.qcl_tools import ScreenState, make_qcl_tools
    from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetLedger
    from chemflow.types import CostRecord

    policies = policies or {}
    state = ScreenState(record, cluster, device_id=device_id)
    tools = make_qcl_tools(state)

    spec = record.spec
    target = spec.target
    args.out.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.out / "journal.jsonl")
    limits = {key: value for key, value in spec.budget.items() if key in RESOURCE_DIMENSIONS}
    ledger_path = args.out / f"{record.content_hash}.ledger.json"
    ledger = BudgetLedger(limits, ledger_path)
    paths = {"journal": str(journal.path), "ledger": str(ledger_path),
             "summary": str(args.out / "summary.json")}
    journal.append(state="INTAKE", role="orchestrator",
                   observation={"question": spec.question, "record_hash": record.hash},
                   decision="accepted")
    for llm in policies.values():
        llm.tools, llm.record, llm.journal = tools, record, journal
        llm.prompts = LLM_PROMPTS
    if "planner" in policies:
        # The task is already signed: the planner may comment, but cannot change it.
        policies["planner"].decide({"task": spec.model_dump(mode="json")})
    lock = (args.out / f"{record.content_hash}.ledger.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        ledger.reload()
        _check_effects(spec)
        remaining = _remaining(spec, ledger)
        active, diagnostics = screens.resolve_active(target)
        reason = ("size from MP2 natural occupations; orbitals: canonical HF frontier window"
                  if diagnostics else "Signed model")
        if diagnostics is not None and "modeler" in policies:
            state.diagnostics = diagnostics
            policies["modeler"].decide({"diagnostics": diagnostics})
            choice = state.choice
            chosen = ({key: choice[key] for key in ("n_elec", "n_orb")}
                      if choice is not None else None)
            problem = screens.active_error(chosen, target)
            if problem is None:
                active, reason = chosen, choice.get("reason") or "Modeler choice"
            else:
                journal.append(state="MODEL_FALLBACK", role="orchestrator",
                               observation={"choice": choice}, decision={"active": active},
                               reason=problem)
        state.active = active
        journal.append(state="MODEL", role="modeler",
                       observation={"diagnostics": diagnostics} if diagnostics else {},
                       decision={"active": active}, reason=reason)
        designer = policies.get("designer")
        llm_candidates: tuple | None = None
        candidate_count = 0
        validity_maps: list[dict] = []
        rounds = []
        controls: list[dict] = []
        reference = None
        reference_graded = None
        headlines: list[dict] = []
        verdict = CriticDecision("stop", "No valid candidate")
        attempted: list[str] = []
        spent = {"wall_s": 0.0, "cpu_core_s": 0.0, "gpu_s": 0.0, "sim_shots": 0}
        values_by_name: dict[str, Any] = {}
        for round_index in range(MAX_ROUNDS):
            # An LLM designer sees round 0 (controls and reference) and proposes round 1.
            proposed = None
            if proposals is not None:
                if round_index > 0:
                    break
                proposed = screens.candidate_entries(spec, proposals, active=active)
            elif designer is not None:
                proposed = () if round_index == 0 else llm_candidates
                if proposed is None:
                    break
            built = screens.build(spec, cluster, round_index=round_index, device_id=device_id,
                                  active=active, controls=round_index == 0, proposed=proposed,
                                  first_index=candidate_count)
            if "simulator" not in spec.allowed_effects and built.sim_shots:
                built = screens.build(spec, cluster, round_index=round_index,
                                      device_id=device_id, active=active,
                                      controls=round_index == 0, simulator=False,
                                      proposed=proposed, first_index=candidate_count)
            candidate_count += len(built.names("candidate"))
            candidates = built.names("candidate")
            if round_index > 0 and not candidates:
                break
            sim_left = remaining.get("sim_shots")
            if built.sim_shots and sim_left is not None and built.sim_shots > sim_left:
                raise Refusal("CF-E070", "sim_shots budget exceeded",
                              "Reduce predicted cost to the remaining task budget",
                              node=candidates[-1] if candidates else None)
            journal.append(state="PROPOSE", role="designer",
                           observation={"round": round_index},
                           decision={"candidates": [built.entry(name).options | {
                               "template": built.entry(name).method}
                               for name in candidates]})
            executed = _execute(spec, cluster, built, device_id, remaining, round_index, active,
                                args.out)
            rounds.append(executed)
            spent["wall_s"] += executed["wall_s"]
            spent["cpu_core_s"] += executed["cpu_core_s"]
            spent["gpu_s"] += executed["gpu_s"]
            spent["sim_shots"] += built.sim_shots
            for key in ("wall_s", "cpu_core_s", "sim_shots"):
                if key in remaining:
                    remaining[key] -= {"wall_s": executed["wall_s"],
                                       "cpu_core_s": executed["cpu_core_s"],
                                       "sim_shots": built.sim_shots}[key]
            if round_index == 0:
                first = built
                reference = _reference_value(spec, built, executed)
                if reference is None:
                    raise AdapterError("CF-E080", built.reference,
                                       "The exact reference failed: "
                                       f"{executed['failures'].get(built.reference)}",
                                       "Use a smaller system or a supported model")
            graded_items = {}
            for entry in built.entries:
                if entry.name not in executed["values"]:
                    continue
                value = executed["values"][entry.name]
                values_by_name[entry.name] = value
                relation = screens.target_relation(spec, executed["relations"].get(entry.name))
                graded = screens.grade(spec, entry, value, reference, relation)
                graded.certificate = certify(executed["plan"], entry.name, relation, args.out)
                graded_items[entry.name] = graded
            if spec.validity:
                mapped = [entry for entry in built.entries
                          if entry.name in graded_items and entry.role != "reference"]
                maps = validity_map(spec, cluster, mapped, active, remaining, spent, args.out)
                for name, item in maps.items():
                    graded_items[name].validity = item
                    validity_maps.append(item)
            for entry in built.entries:
                if entry.name not in graded_items:
                    continue
                graded = graded_items[entry.name]
                headlines.append(_headline(spec, entry.name, entry, graded.value, graded,
                                           entry.role == "reference", built))
                if entry.role == "reference":
                    reference_graded = graded
            if round_index == 0 and not built.names("reference") and reference is not None:
                # Model-defined screens: the reference is the signed model itself (or, for a
                # spectral function, the independent Fock-space calculation).
                entry = screens.Entry("reference", "reference", "model")
                graded = screens.grade(spec, entry, reference, reference, "equivalent")
                headlines.append(_headline(spec, "reference", entry, reference, graded, True))
            for name, failure in executed["failures"].items():
                headlines.append({"node": name, "method": built.entry(name).method,
                                  "role": built.entry(name).role, "value": None,
                                  "relation": "unknown", "gate_verdict": f"FAIL ({failure})",
                                  "refused": failure})
            journal.append(state="SCREEN", role="orchestrator",
                           observation={"round": round_index, "plan_hash": executed["plan_hash"],
                                        "predicted": executed["predicted"],
                                        "failures": executed["failures"]},
                           decision="screened",
                           outputs=[name for name in candidates if name in executed["values"]],
                           calls=[{"name": "plan.compile",
                                   "inputs": {"round": round_index,
                                              "outputs": [entry.name
                                                          for entry in built.entries]}},
                                  {"name": "plan.run",
                                   "inputs": {"plan_hash": executed["plan_hash"],
                                              "overlap": executed["overlap"]}}])
            if round_index == 0:
                controls = [_item(graded_items[name], built)
                            for name in built.names("control") if name in graded_items]
            candidate_items = [_item(graded_items[name], built) for name in candidates
                               if name in graded_items]
            if designer is not None and round_index == 0:
                pool = controls + ([_item(reference_graded, built)]
                                   if reference_graded is not None else [])
                state.controls, state.headlines = pool, headlines
                state.remaining = dict(remaining)
                outcome = designer.decide({"controls": pool, "target": target.kind,
                                           "active": active})
                llm_candidates = _designer_outcome(spec, state, journal, outcome, active)
                if llm_candidates is not None:
                    continue
            if not candidate_items:
                pool = controls + ([_item(reference_graded, built)]
                                   if reference_graded is not None and
                                   reference_graded.name not in
                                   {item["id"] for item in controls} else [])
                verdict = review([], decision_items(pool))
                journal.append(state="CRITIQUE", role="critic",
                               observation={"candidates": [], "controls": pool},
                               decision=vars(verdict))
                break
            observation = {"candidates": candidate_items, "controls": controls}
            if "critic" in policies:
                state.headlines = headlines
                arguments = policies["critic"].decide(observation)
                journal.append(state="CRITIC_ARGUMENTS", role="critic",
                               observation={"candidates": [item["id"] for item in
                                                           candidate_items]},
                               decision={"content": arguments.get("content"),
                                         "end_reason": arguments.get("end_reason")})
            verdict = review(decision_items(candidate_items), decision_items(controls))
            journal.append(state="CRITIQUE", role="critic", observation=observation,
                           decision=vars(verdict))
            if (verdict.decision in ("escalate", "stop") or designer is not None or
                    proposals is not None):
                if verdict.decision == "revise":
                    verdict = CriticDecision("stop", "Maximum screening rounds reached")
                break
            attempted.extend(candidates)
        else:
            if verdict.decision == "revise":
                verdict = CriticDecision("stop", "Maximum screening rounds reached")
        if verdict.decision != "escalate" and verdict.candidate is None:
            pool = controls + ([_item(reference_graded, first)]
                               if reference_graded is not None and reference_graded.name not in
                               {item["id"] for item in controls} else [])
            fallback = review([], decision_items(pool))
            if fallback.candidate is not None:
                verdict = fallback
        journal.append(state=verdict.decision.upper(), role="orchestrator", observation={},
                       decision=vars(verdict), reason=verdict.reason)
        if "analyst" in policies:
            state.headlines = headlines
            policies["analyst"].decide({"decision": verdict.decision, "reason": verdict.reason,
                                        "candidate": verdict.candidate})
        ledger.record(CostRecord(**{key: value for key, value in spent.items()
                                    if key in limits}))
    except Refusal as refusal:
        _refuse(args, record, journal, ledger, refusal, paths)
    except AdapterError as error:
        diagnostic = error.diagnostics[0]
        if diagnostic.code not in ("CF-E070", "CF-E080"):
            raise
        consumed = any(value for value in spent.values()) if "spent" in locals() else False
        if consumed:
            ledger.record(CostRecord(**{key: value for key, value in spent.items()
                                        if key in limits}))
        hint = diagnostic.fix_hints[0] if diagnostic.fix_hints else "Use a supported model"
        _refuse(args, record, journal, ledger,
                Refusal(diagnostic.code, diagnostic.message.removeprefix(
                    f"{diagnostic.code}: "), hint, diagnostic.node), paths,
                nothing_consumed=not consumed)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    results = ResultsDB(args.out / "results.sqlite")
    for headline in headlines:
        if headline.get("value") is None:
            continue
        number, _, unit = headline["value"].partition(" ")
        headline["result_id"] = results.add_headline(
            record_hash=record.hash, plan_hash=rounds[0]["plan_hash"],
            node_id=headline["node"], operator=target.kind,
            relation=headline["relation"], value=float(number), unit=unit,
            headline=headline)
    results.add_qcl_run(record_hash=record.hash, plan_hash=rounds[0]["plan_hash"],
                        target_kind=target.kind, decision=verdict.decision,
                        result=serializable(values_by_name.get(verdict.candidate)),
                        profile=[item for executed in rounds for item in executed["profile"]],
                        spent=dict(ledger.spent))
    results.connection.close()
    chosen = values_by_name.get(verdict.candidate)
    if chosen is not None:
        (args.out / "qcl-result.json").write_text(
            json.dumps(serializable(chosen), indent=2, sort_keys=True) + "\n")
        paths["result"] = str(args.out / "qcl-result.json")
    (args.out / "qcl-profile.json").write_text(json.dumps(
        [item for executed in rounds for item in executed["profile"]], indent=2,
        sort_keys=True) + "\n")
    (args.out / "qcl-screen.json").write_text(json.dumps({
        "rounds": [{"plan_hash": executed["plan_hash"], "predicted": executed["predicted"],
                    "wall_s": executed["wall_s"], "cpu_core_s": executed["cpu_core_s"],
                    "failures": executed["failures"], "explanation": executed["explain"]}
                   for executed in rounds]}, indent=2, sort_keys=True, default=str) + "\n")
    report = _report(spec, verdict, headlines, rounds, values_by_name.get(verdict.candidate),
                     validity_maps)
    if state.report:
        report += "\n## Analyst\n\n" + state.report.strip() + "\n"
    (args.out / "report.md").write_text(report)
    paths.update(report=str(args.out / "report.md"), results=str(args.out / "results.sqlite"),
                 profile=str(args.out / "qcl-profile.json"),
                 screen=str(args.out / "qcl-screen.json"))
    summary = {"record_hash": record.hash, "engine": "qcl", "decision": verdict.decision,
               "candidate": verdict.candidate, "reason": verdict.reason,
               "results": headlines, "refusal": None, "policy": policy,
               "roles": sorted(policies),
               "plan_hashes": [executed["plan_hash"] for executed in rounds],
               "spent": dict(ledger.spent), "limits": ledger.limits, "paths": paths}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{verdict.decision} {paths['report']}")
    return summary


LLM_PROMPTS = {
    "designer": (
        "You are ChemFlow's designer on the QCL engine. The signed task, its gates and the "
        "classical controls and exact reference are fixed; observation.controls holds their "
        "graded results. Propose candidate methods that could meet the gates more cheaply or "
        "with a stronger relation: sci {select_cutoff} or sqd {shots} (closed-shell "
        "molecules only). Use methods.list for ranges, screen.review to compile a proposal "
        "and see its inferred relations, predicted time and simulator shots, then finish "
        "with design.submit {candidates} or design.decline {reason}. Either ends your turn. "
        "Never assert evidence, relations or gates; ChemFlow grades every output."),
    "modeler": (
        "You are ChemFlow's modeler on the QCL engine. For an automatic active space, read "
        "observation.diagnostics (or diagnose.model) and finish with model.choose "
        "{n_elec, n_orb, reason}: positive counts, at most 10 orbitals, at most two "
        "electrons per orbital, parity compatible with the molecule's spin. An infeasible "
        "choice falls back to the diagnostics' suggested_active."),
    "critic": (
        "You are ChemFlow's critic. Challenge each candidate against the classical "
        "controls using screen.controls and results.query; never loosen gates. The final "
        "verdict is computed deterministically from the graded results."),
    "planner": (
        "You are ChemFlow's planner. The task is already signed by a human and cannot be "
        "changed; read it with task.get and note anything the reviewer should know."),
    "analyst": (
        "You are ChemFlow's analyst. Read results.query and write a short report with "
        "report.write: relations, gate verdicts, costs and limitations. Quote only recorded "
        "energies; never claim advantage, speedup or certification."),
}


def _designer_outcome(spec: Any, state: Any, journal: Journal, outcome: dict,
                      active: Any) -> tuple | None:
    """The candidates an LLM designer submitted, or None to keep the controls."""
    if state.submitted:
        try:
            entries = screens.candidate_entries(spec, state.submitted, active=active)
        except ValueError as error:
            journal.append(state="DESIGN_REFUSED", role="orchestrator", observation={},
                           decision="refused", reason=str(error))
            return None
        journal.append(state="PROPOSE", role="designer", observation={"round": 1},
                       decision={"candidates": state.submitted})
        return entries
    if state.declined:
        journal.append(state="DESIGN_DECLINED", role="orchestrator", observation={},
                       decision="declined", reason=state.declined)
        return None
    journal.append(state="DESIGN_REFUSED", role="orchestrator", observation={},
                   decision="refused",
                   reason="Nothing submitted: designer ended with " +
                          str(outcome.get("end_reason", "unknown termination")))
    return None


def quantity_label(spec: Any) -> str | None:
    """What a model-defined or thermal answer means (as the legacy report labels)."""
    target = spec.target
    thermo = getattr(target, "thermo", None)
    if target.kind in ("reaction_energy", "reaction_barrier") and thermo is not None:
        if thermo.quantity == "enthalpy":
            return f"ΔH({thermo.temperature_K:g} K)"
        return f"ΔG({thermo.temperature_K:g} K, {thermo.pressure_Pa / 101325:g} atm)"
    if target.kind == "frontier_orbitals":
        method = target.level.functional or target.level.method
        return f"model-defined {method} {target.quantity}; not a physical IP/EA"
    if target.kind == "solvation_free_energy":
        method = target.level.functional or target.level.method
        label = f"model-defined {method}/{target.solvent.model} solvation"
        if target.solvent.model == "pcm":
            label += f"; IEF-PCM eps={target.solvent.eps}; electrostatic only"
        return label
    if target.kind == "metabolic_sites":
        return ("model-defined C-H reactivity score; not enzyme specificity or metabolic "
                "kinetics")
    return None


def _model_section(value: Any) -> list[str]:
    """The provenance a model-defined answer carries: implementation, flags, model card."""
    provenance = getattr(value, "provenance", None)
    if not isinstance(provenance, dict):
        return []
    lines = ["", "## Model", ""]
    if provenance.get("implementation"):
        lines.append(f"- Implementation: {provenance['implementation']}")
    method = provenance.get("method")
    if isinstance(method, dict) and method.get("family"):
        lines.append(f"- Method: {method['family']}")
    card = provenance.get("model_card")
    if card:
        lines.append(f"- Empirical model card: train RMSE {card.get('train_rmse'):.4g}, "
                     f"cross-validated RMSE {card.get('cv_rmse'):.4g}; dataset sha256 "
                     f"{str(card.get('dataset_sha256'))[:16]}")
    if provenance.get("validity_status"):
        lines.append(f"- Applicability: {provenance['validity_status']}")
    lines.extend(f"- {flag}" for flag in provenance.get("validity_flags", []))
    return lines if len(lines) > 3 else []


def _report(spec: Any, verdict: Any, headlines: list[dict], rounds: list[dict],
            chosen: Any = None, validity_maps: list[dict] | None = None) -> str:
    from chemflow.agent.validity import section
    label = quantity_label(spec)
    lines = ["# QCL screen", "", f"Task: {spec.question}", "",
             f"Decision: {verdict.decision}" + (f" ({verdict.candidate})"
                                               if verdict.candidate else ""),
             f"Reason: {verdict.reason}", *([f"Quantity: {label}"] if label else []),
             "", "## Methods", "",
             "| Output | Role | Method | Value | Error vs reference | Relation | Gates |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for headline in headlines:
        error = headline.get("error")
        lines.append(f"| {headline['node']} | {headline.get('role')} | {headline['method']} | "
                     f"{headline.get('value')} | "
                     f"{'' if error is None else f'{error:.3g}'} | {headline['relation']} | "
                     f"{headline['gate_verdict']} |")
    lines.extend(_model_section(chosen))
    lines.extend(section(validity_maps or []))
    lines.extend(["", "## Execution", ""])
    for index, executed in enumerate(rounds):
        predicted = executed["predicted"] or {}
        lines.append(f"- Round {index}: plan {executed['plan_hash'][:16]}, "
                     f"{executed['wall_s']:.2f} s wall, {executed['cpu_core_s']:.2f} CPU core-s"
                     + (f", predicted makespan {predicted['makespan_s']:.2f} s of "
                        f"{predicted['sequential_s']:.2f} s sequential"
                        if predicted.get("makespan_s") is not None and
                        predicted.get("sequential_s") is not None else ""))
    return "\n".join(lines) + "\n"


def review_screen(args: Any, record: Any, cluster: Any, *,
                  device_id: str | None = None) -> str:
    """Compile the first screening round without running it (``--dry-run``)."""
    spec = record.spec
    args.out.mkdir(parents=True, exist_ok=True)
    active, _ = screens.resolve_active(spec.target)
    built = screens.build(spec, cluster, device_id=device_id, active=active)
    plan = built.workflow.compile(*built.args)
    schedule = plan.schedule()
    review = {"record_hash": record.hash, "plan_hash": plan.hash,
              "outputs": [{"name": entry.name, "role": entry.role, "method": entry.method,
                           **({"params": entry.options} if entry.params else {})}
                          for entry in built.entries],
              "relations": plan.relations(), "reference": built.reference,
              "sim_shots": built.sim_shots, "predicted": schedule.get("predicted"),
              "explanation": plan.explain(format="json")}
    (args.out / "qcl-review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, default=str) + "\n")
    (args.out / "summary.json").write_text(json.dumps(
        {"record_hash": record.hash, "engine": "qcl", "decision": "reviewed",
         "plan_hash": plan.hash, "paths": {"review": str(args.out / "qcl-review.json")}},
        indent=2, sort_keys=True) + "\n")
    print(plan.hash)
    return plan.hash
