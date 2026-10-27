from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from pathlib import Path

from chemflow.agent.costs import FEATURES, estimate_cost
from chemflow.agent.critic import CriticDecision, decision_items, review
from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy
from chemflow.agent.memory import ResultsDB
from chemflow.agent.policies import Policy, ScriptedPolicy
from chemflow.agent.report import result_headline, result_method, result_summary, write_report
from chemflow.agent.task import TaskRecord, TaskSpec, TaskStore
from chemflow.agent.tools import make_tools
from chemflow.cards import load_cards
from chemflow.check.static import check
from chemflow.diagnostics import AdapterError, CheckError
from chemflow.ir.schema import Program
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime import Executor
from chemflow.types import CostRecord, EnergyCurve, Estimate, GateVerdict, SpectralEstimate


@dataclass
class AgentRun:
    record: TaskRecord
    decision: CriticDecision
    report: str
    journal: Path
    results: list[dict]


class Orchestrator:
    def __init__(self, store: TaskStore, approver: Callable[[TaskSpec], TaskRecord],
                 output_dir: str | Path, max_rounds: int = 2, max_repairs: int = 2,
                 policies: dict[str, Policy] | None = None,
                 plan_directory: Path | None = None):
        self.store = store
        self.approver = approver
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self.max_repairs = max_repairs
        self.plan_directory = plan_directory
        self.journal = Journal(self.output_dir / "journal.jsonl")
        self.memory = ResultsDB(self.output_dir / "results.sqlite")
        self.registry = molecular_registry()
        self.executor = Executor(self.registry)
        self.policies = {role: ScriptedPolicy(role) for role in
                         ("planner", "modeler", "designer", "critic", "analyst")}
        self.policies.update(policies or {})
        for policy in self.policies.values():
            if isinstance(policy, LLMPolicy):
                policy.journal = self.journal

    def _transition(self, state: str, role: str = "orchestrator",
                    observation: dict | None = None, decision: dict | None = None,
                    reason: str = "", calls: list | None = None,
                    outputs: list | None = None) -> dict:
        observation = observation or {}
        decision = decision if decision is not None else (
            self.policies[role].decide(observation) if role in self.policies else {})
        self.journal.append(state=state, role=role, observation=observation,
                            decision=decision, reason=reason, calls=calls, outputs=outputs)
        return decision

    def run(self, task: TaskSpec | dict, diagnostics: dict | None = None) -> AgentRun:
        spec = TaskSpec.model_validate(task)
        self._transition("INTAKE", observation={"question": spec.question})
        self._transition("DRAFT_TASK", "planner", {"task": spec.model_dump(mode="json")})
        record = self.approver(spec)
        self.store.verify(record)
        if record.spec != spec:
            raise ValueError("Approver changed the drafted task")
        if spec.target.kind != "molecular_ground_energy":
            from chemflow.domains import load_registry, pack_for_target

            if spec.target.kind == "periodic_energy":
                raise ValueError("Periodic symmetry tasks run through the materials benchmark")
            pack = pack_for_target(spec.target.kind)
            self.registry = load_registry(["molecular", pack.name])
            self.executor = Executor(self.registry)
            if spec.target.kind in ("lattice_ground_energy", "spectral_function",
                                    "pairing_correlation"):
                declared_active = {"n_orb": spec.target.sector["n_orb"], "n_elec":
                                   spec.target.sector["n_alpha"] + spec.target.sector["n_beta"]}
            elif spec.target.kind in ("reaction_energy", "reaction_barrier"):
                declared_active = {key: sum(item.active[key] for item in spec.target.species)
                                   for key in ("n_orb", "n_elec")}
            else:
                declared_active = None
            if declared_active is None:
                diagnostics = {"full_molecular_space": True, **(diagnostics or {})}
            else:
                diagnostics = {"suggested_active": declared_active, **(diagnostics or {})}
        else:
            declared_active = spec.target.active
            if declared_active != "auto" and diagnostics is None:
                diagnostics = {"suggested_active": declared_active}
        self._transition("AWAIT_FREEZE", observation={"record_hash": record.hash},
                         reason="Human approver signed the task")
        tools = make_tools(self.registry, self.executor, record, self.store, self.memory,
                           plan_directory=self.plan_directory)
        for policy in self.policies.values():
            if isinstance(policy, LLMPolicy):
                policy.record, policy.tools, policy.journal = record, tools, self.journal
        diagnostic_call = []
        if diagnostics is None:
            import json

            inputs = {"system_json": json.dumps(spec.target.molecule),
                      "model_json": json.dumps(spec.target.electronic_model)}
            diagnostics = tools.call("modeler", "diagnose.model", inputs)["data"]
            diagnostic_call = [{"name": "diagnose.model", "inputs": inputs}]
        active, fallback = self._model_active(spec, declared_active, diagnostics, diagnostic_call)
        self._transition("CONTROLS", observation={"active": active, "fallback": fallback},
                         reason=("size from MP2 natural occupations; orbitals: canonical HF "
                                 "frontier window") if
                         declared_active == "auto" else "Fixed active space")
        seed_id = tools.call("modeler", "plan.new", {"identifier": "base"})["data"]["plan_id"]
        base_inputs = {"plan_id": seed_id, "active": active}
        base = tools.call("modeler", "plan.base", base_inputs)["data"]
        plan_id, controls = base["plan_id"], base["controls"]
        plan = tools.plans.get(plan_id)
        reference = plan.references[0]
        self._transition("REFERENCE", observation={"references": plan.references,
                         "controls": controls}, reason="Exact grading reference is screening-only",
                         calls=[{"name": "plan.base", "inputs": base_inputs}])
        control_inputs = {"plan_id": plan_id, "scale": "screen"}
        control_results = self._decode_run(tools.call("designer", "run", control_inputs)["data"],
                                           self.executor.store)
        self._transition("CONTROLS_RESULT", observation={"plan_hash": plan.hash},
                         calls=[{"name": "run", "inputs": control_inputs}],
                         outputs=controls)
        control_gates = {item.node: item for item in control_results.get("gates", [])}
        cards = [card.model_dump(mode="json") for card in load_cards(self.registry)]
        if isinstance(self.policies["designer"], LLMPolicy):
            return self._llm_screen(record, spec, tools, plan, controls, reference,
                                    control_results, control_gates, cards, active)
        proposal = self._transition("PROPOSE", "designer", {"cards": cards,
            "target": spec.target.operator, "n_orb": active["n_orb"] if active else 0})
        candidates = proposal["candidates"]
        attempted = []
        verdict = CriticDecision("stop", "No valid candidate")
        outcomes = []
        for round_index in range(self.max_rounds):
            candidates = [candidate for candidate in candidates
                          if self._cost_allowed(candidate, cards, spec.budget)]
            candidate_inputs = {"plan_id": plan_id,
                                "candidates": candidates}
            built = tools.call("designer", "plan.candidates", candidate_inputs)["data"]
            trial_id, ids = built["plan_id"], built["ids"]
            trial = tools.plans.get(trial_id)
            for repair in range(self.max_repairs + 1):
                check_inputs = {"plan_id": trial_id}
                diagnostics_found = [item.model_dump(mode="json") for item in
                                     check(trial, self.registry, purpose="deploy")]
                self._transition("CHECK_REPAIR", observation={"round": round_index,
                    "repair": repair, "diagnostics": [item["code"] for item in diagnostics_found]},
                    calls=[{"name": "plan.candidates", "inputs": candidate_inputs},
                           {"name": "check", "inputs": check_inputs}] if repair == 0 else
                          [{"name": "check", "inputs": check_inputs}])
                if not diagnostics_found:
                    break
                if repair == self.max_repairs:
                    ids = {}
            self._transition("COST_FILTER", observation={"round": round_index,
                "candidate_count": len(ids), "budget": spec.budget},
                reason="Known cost upper bounds within each budget dimension; "
                       "unmodeled resource costs remain unknown")
            if not ids:
                verdict, outcomes = self._review_controls(
                    controls, reference, control_results, control_gates, spec, plan)
                self._transition(verdict.decision.upper(), reason=verdict.reason)
                break
            run_inputs = {"plan_id": trial_id, "scale": "screen"}
            computed = self._decode_run(tools.call("designer", "run", run_inputs)["data"],
                                       self.executor.store)
            self._transition("SCREEN", observation={"plan_hash": trial.hash},
                             calls=[{"name": "run", "inputs": run_inputs}],
                             outputs=list(ids.values()))
            gates = {item.node: item for item in computed.get("gates", [])}
            control_items = [self._item(node_id, control_results, control_gates,
                                       reference, spec, plan) | {"method": result_method(node_id, plan)}
                             for node_id in controls]
            candidate_items = [self._item(node_id, computed, gates,
                                         reference, spec, trial) for node_id in ids.values()]
            verdict_data = self._transition("CRITIQUE", "critic", {
                "candidates": candidate_items, "controls": control_items,
                "references": plan.references})
            verdict = CriticDecision(**verdict_data.get("verdict", verdict_data))
            outcomes = [self._headline(node_id, control_results, control_gates,
                                       reference, spec, plan) for node_id in [*controls, reference]
                        if node_id not in control_results.get("refused", {})]
            outcomes[-1]["screening_reference"] = True
            outcomes += [self._headline(node_id, computed, gates, reference, spec,
                                        trial) for node_id in ids.values()
                         if node_id not in computed.get("refused", {})]
            sector = next(node.params for node in trial.nodes if node.op == "sector")
            target_dimension = (comb(sector["n_orb"], sector["n_alpha"]) *
                                comb(sector["n_orb"], sector["n_beta"]))
            for headline in outcomes:
                node_id = headline["node"]
                source = computed if node_id in computed else control_results
                node = next((item for item in trial.nodes if item.id == node_id), None)
                template = node.refinement if node else None
                if not template or template not in FEATURES:
                    continue
                feature = FEATURES[template]
                diagnostic = source[node_id].diagnostics
                size = diagnostic.get({"determinants": "determinant_count",
                                       "subspace_dimension": "subspace_dimension"}.get(
                                           feature, feature))
                if feature == "sector_dimension":
                    size = target_dimension
                if feature == "shots":
                    size = node.params.get("shots")
                measured = source["costs"][node_id].wall_s
                if size and measured and measured > 0:
                    self.memory.add_cost(template, feature, size, "wall_s", measured)
                target_size = (node.params.get("target_size") or target_dimension
                               if feature != "shots" else (size or 1) * 2)
                prediction = estimate_cost(self.memory, template, target_size)
                if prediction:
                    headline["target_cost"] = {**prediction, "dimension": "wall_s"}
            if verdict.decision == "escalate":
                self._transition("ESCALATE", reason="Screen passed; larger runs remain subject to the task budget")
                break
            if verdict.decision == "stop":
                self._transition("STOP", reason=verdict.reason)
                break
            self._transition("REVISE", reason=verdict.reason)
            attempted.extend(candidates)
            proposal = self._transition("PROPOSE", "designer", {"cards": cards,
                "target": spec.target.operator, "n_orb": active["n_orb"],
                "attempted": attempted})
            candidates = proposal["candidates"]
            if not candidates:
                verdict, control_outcomes = self._review_controls(
                    controls, reference, control_results, control_gates, spec, plan)
                outcomes = outcomes or control_outcomes
                self._transition(verdict.decision.upper(), reason=verdict.reason)
                break
        else:
            if verdict.decision == "revise":
                verdict = CriticDecision("stop", "Maximum screening rounds reached")
                self._transition("STOP", reason=verdict.reason)
        if verdict.decision != "escalate" and verdict.candidate is None:
            control_verdict, control_outcomes = self._review_controls(
                controls, reference, control_results, control_gates, spec, plan)
            if control_verdict.candidate is not None:
                verdict = control_verdict
                outcomes = list({item["node"]: item for item in
                                 [*outcomes, *control_outcomes]}.values())
                self._transition("STOP", reason=verdict.reason)
        for headline in outcomes:
            self.memory.record_critic(headline["result_id"], vars(verdict))
        report = write_report(spec.question, verdict.decision, verdict.reason, outcomes,
                              results=self.memory, record_hash=record.hash,
                              answer_node=verdict.candidate,
                              orbital_provenance=self._orbital_provenance(spec))
        self._transition("REPORT", "analyst", {"decision": verdict.decision,
                         "reason": verdict.reason})
        (self.output_dir / "report.md").write_text(report)
        self._transition("LEARN", observation={"result_count": len(self.memory.query())})
        return AgentRun(record, verdict, report, self.journal.path, outcomes)

    @staticmethod
    def _active_error(active, target_spin):
        if (not isinstance(active, dict) or any(type(active.get(key)) is not int
                                               for key in ("n_elec", "n_orb"))):
            return "Modeler did not choose integer active-space counts with model.choose"
        if not 0 < active["n_orb"] <= 10:
            return "Active space exceeds the screening orbital cap or has no orbitals"
        spin_twice = 2 * (target_spin or 0)
        if (active["n_elec"] <= 0 or active["n_elec"] > 2 * active["n_orb"] or
                active["n_elec"] < spin_twice or (active["n_elec"] - spin_twice) % 2):
            return "Modeler chose an infeasible spin/electron active space"
        return None

    def _model_active(self, spec, declared_active, diagnostics, diagnostic_call):
        observation = {"diagnostics": diagnostics}
        if declared_active != "auto":
            self._transition("MODEL", observation=observation,
                             decision={"active": declared_active, "diagnostics": diagnostics},
                             reason="Fixed active space" if declared_active is not None else
                                    "Full molecular space; no active-space choice is needed",
                             calls=diagnostic_call)
            if declared_active is not None and declared_active["n_orb"] > 10:
                raise ValueError("Active space exceeds the screening orbital cap")
            return declared_active, False
        chosen = self._transition("MODEL", "modeler", observation, calls=diagnostic_call)
        llm_modeler = isinstance(self.policies["modeler"], LLMPolicy)
        if llm_modeler:
            choice = next((output["result"]["data"] for output in chosen["outputs"]
                           if output.get("name") == "model.choose" and "error" not in output), None)
            active = ({key: choice[key] for key in ("n_elec", "n_orb")}
                      if choice is not None else None)
        else:
            active = chosen["active"]
        reason = self._active_error(active, spec.target.target_spin)
        fallback = llm_modeler and reason is not None
        if fallback:
            active = diagnostics.get("suggested_active")
            self._transition("MODEL_FALLBACK", observation=observation,
                             decision={"active": active}, reason=reason)
            reason = self._active_error(active, spec.target.target_spin)
        if reason:
            raise ValueError(reason)
        return active, fallback

    def _review_controls(self, controls, reference, computed, gates, spec, plan):
        node_ids = list(dict.fromkeys([*controls, reference]))
        control_items = [self._item(node_id, computed, gates, reference, spec, plan)
                         for node_id in node_ids]
        verdict = review([], decision_items(control_items))
        self._transition("CRITIQUE", "critic", {"candidates": [], "controls": control_items,
                         "references": plan.references}, decision=vars(verdict))
        outcomes = [self._headline(node_id, computed, gates, reference, spec, plan)
                    for node_id in node_ids if node_id not in computed.get("refused", {})]
        for headline in outcomes:
            if headline["node"] == reference:
                headline["screening_reference"] = True
        return verdict, outcomes

    def _llm_screen(self, record, spec, tools, plan, controls, reference,
                    control_results, control_gates, cards, active) -> AgentRun:
        base_plan_id = next(identifier for identifier, stored in tools.plans._plans.items()
                            if stored.hash == plan.hash)
        tools.design_base_plan_id = base_plan_id
        proposal = self._transition("PROPOSE", "designer", {
            "cards": cards, "target": spec.target.operator,
            "n_orb": active["n_orb"] if active else 0,
            "base_plan_id": base_plan_id})
        outcomes = []
        candidate_items = []
        fallback_controls = [self._item(node_id, control_results, control_gates,
                                        reference, spec, plan)
                             for node_id in dict.fromkeys([*controls, reference])]
        verdict = CriticDecision("stop", "No valid LLM candidate plan")
        terminal = next((output for output in proposal["outputs"] if "error" not in output
                         and output.get("name") in {"design.submit", "design.decline"}), None)
        if terminal and terminal["name"] == "design.decline":
            declined = terminal["result"]["data"]
            if "refusal" in declined:
                refusal = declined["refusal"]
                self._transition("DESIGN_REFUSED", observation=declined,
                                 reason=refusal["message"])
                raise AdapterError(refusal["code"], refusal["node"], refusal["message"],
                                   "Reduce predicted cost to the remaining task budget",
                                   nothing_consumed=True)
            self._transition("DESIGN_DECLINED", observation={key: declined[key] for key in
                             ("unscreened_candidate_ids", "force") if key in declined},
                             reason=terminal["result"]["data"]["reason"])
            verdict, outcomes = self._review_controls(
                controls, reference, control_results, control_gates, spec, plan)
            return self._finish_llm_screen(record, spec, verdict, outcomes, candidate_items,
                                           fallback_controls)
        try:
            built = terminal["result"]["data"] if terminal else next((
                         output["result"]["data"] for output in reversed(proposal["outputs"])
                         if "error" not in output and
                         output.get("name") in {"plan.candidates", "plan.add_node", "plan.refine",
                                               "plan.bind", "plan.remove_node"}), None)
            if built is None:
                raise ValueError("Nothing built: designer ended with " +
                                 proposal.get("end_reason", "unknown termination"))
            trial_id = built["plan_id"]
            trial = tools.plans.get(trial_id)
            if (trial.gates or trial.nodes[:len(plan.nodes)] != plan.nodes or
                    trial.references != plan.references or
                    not set(plan.targets) <= set(trial.targets)):
                raise ValueError("Designer changed frozen controls, references or gates")
            ids = built.get("ids") or {node_id: node_id for node_id in trial.targets
                                       if node_id not in plan.targets}
            if not ids:
                raise ValueError("No candidate targets")
            diagnostics = [item.model_dump(mode="json") for item in
                           check(trial, self.registry, purpose="deploy")]
            self._transition("CHECK_REPAIR", observation={"diagnostics": diagnostics},
                             calls=[{"name": "check", "inputs": {"plan_id": trial_id}}])
            if diagnostics:
                raise ValueError("Candidate plan failed static check")
            computed = self._decode_run(tools.call("designer", "run", {
                "plan_id": trial_id, "scale": "screen"})["data"], self.executor.store)
            self._transition("SCREEN", observation={"plan_hash": trial.hash},
                             calls=[{"name": "run", "inputs": {"plan_id": trial_id,
                                                               "scale": "screen"}}],
                             outputs=list(ids.values()))
            gates = {item.node: item for item in computed.get("gates", [])}
            candidate_items = [self._item(node_id, computed, gates, reference, spec, trial)
                               for node_id in ids.values()]
            control_items = [self._item(node_id, control_results, control_gates,
                                        reference, spec, plan) | {"method": result_method(node_id, plan)}
                             for node_id in controls]
            observation = {"candidates": candidate_items, "controls": control_items,
                           "references": plan.references}
            if isinstance(self.policies["critic"], LLMPolicy):
                arguments = self.policies["critic"].decide(observation)
                self._transition("CRITIC_ARGUMENTS", observation=observation,
                                 decision=arguments)
            verdict = review(decision_items(candidate_items), decision_items(control_items))
            self._transition("CRITIQUE", "critic", observation, decision=vars(verdict))
            outcomes = [self._headline(node_id, control_results, control_gates,
                                       reference, spec, plan) for node_id in [*controls, reference]
                        if node_id not in control_results.get("refused", {})]
            outcomes[-1]["screening_reference"] = True
            outcomes.extend(self._headline(node_id, computed, gates, reference, spec, trial)
                            for node_id in ids.values() if node_id not in computed.get("refused", {}))
        except (CheckError, ValueError, KeyError, StopIteration, PermissionError, TypeError) as exc:
            self._transition("DESIGN_REFUSED", observation={},
                             reason=str(exc).strip() or type(exc).__name__)
            candidate_items = []
            verdict, outcomes = self._review_controls(
                controls, reference, control_results, control_gates, spec, plan)
        return self._finish_llm_screen(record, spec, verdict, outcomes, candidate_items,
                                       fallback_controls)

    def _finish_llm_screen(self, record, spec, verdict, outcomes, candidate_items,
                           control_items) -> AgentRun:
        if candidate_items and verdict.decision == "revise":
            verdict = CriticDecision("stop", "Maximum screening rounds reached")
        if verdict.decision != "escalate" and verdict.candidate is None:
            control_verdict = review([], decision_items(control_items))
            if control_verdict.candidate is not None:
                verdict = control_verdict
        self._transition(verdict.decision.upper(),
                         reason=verdict.reason)
        for headline in outcomes:
            self.memory.record_critic(headline["result_id"], vars(verdict))
        report = write_report(spec.question, verdict.decision, verdict.reason, outcomes,
                              results=self.memory, record_hash=record.hash,
                              answer_node=verdict.candidate,
                              orbital_provenance=self._orbital_provenance(spec))
        self._transition("REPORT", "analyst", {"decision": verdict.decision,
                         "reason": verdict.reason})
        (self.output_dir / "report.md").write_text(report)
        self._transition("LEARN", observation={"result_count": len(self.memory.query())})
        return AgentRun(record, verdict, report, self.journal.path, outcomes)

    @staticmethod
    def _cost_allowed(candidate: dict, cards: list[dict], budget: dict[str, float]) -> bool:
        if (candidate["template"] == "sqd" and "sim_shots" in budget and
                candidate["shots"] > budget["sim_shots"]):
            return False
        card = next((item for item in cards if item["template"] == candidate["template"]),
                    None)
        model = card.get("cost_model") if card else None
        if not model:
            return True
        for dimension, limit in budget.items():
            predicted = model.get(dimension)
            upper = predicted.get("upper") if isinstance(predicted, dict) else predicted
            if upper is not None and upper > limit:
                return False
        return True

    @staticmethod
    def _orbital_provenance(spec: TaskSpec) -> str:
        if spec.target.kind == "metabolic_sites":
            return ("Empirical C-H site reactivity: SASA_fraction * sigmoid((90 - BDE_kcal/mol)/3). "
                    "Model-defined, uncalibrated evidence; not enzyme or metabolic-rate prediction.")
        if spec.target.kind == "aqueous_solubility":
            from chemflow.empirical import evaluate, model_card

            target = spec.target
            card = model_card(target.model)
            result = evaluate(target.model, target.molecule_smiles, target.melting_point_C)
            return (f"Empirical model: {target.model}; train RMSE: {card['train_rmse']}; "
                    f"5-fold CV RMSE: {card['cv_rmse']}; applicability: "
                    f"{result['validity_status']}. Model estimate of the experimental "
                    "property (model_change), not experimental accuracy evidence." +
                    (" GSE RMSE is unavailable: the dataset lacks melting points; "
                     "applicability uses the ESOL structural domain as a proxy."
                     if target.model == "gse" else ""))
        if spec.target.kind in ("solvation_free_energy", "frontier_orbitals",
                                "ionization_energy", "electron_affinity"):
            return "Full molecular orbital space; fixed geometry and declared model"
        if spec.target.kind in ("bond_dissociation", "ranking"):
            return "Full molecular orbital spaces; neutral rigid radicals use ROHF references"
        if spec.target.kind in ("pes_scan", "interaction_scan"):
            return "Full molecular orbital spaces at each signed scan geometry"
        if spec.target.kind == "binding_free_energy":
            return ("Model-defined Amber14/GBn2 MM/GBSA; relation to experimental binding "
                    "free energy is model_change. No configurational entropy correction.")
        if spec.target.kind == "spectral_function":
            return "Impurity and bath orbitals in the frozen task sector"
        if spec.target.kind == "interaction_energy":
            return "Full molecular orbital spaces for the complex and fragments"
        if spec.target.kind in ("lattice_ground_energy", "pairing_correlation"):
            return "Hubbard site basis; fixed lattice sector"
        if spec.target.kind in ("reaction_energy", "reaction_barrier"):
            return "per-species declared active spaces; canonical HF orbitals"
        prefix = ("size from MP2 natural occupations" if spec.target.active == "auto"
                  else "fixed active space")
        return f"{prefix}; orbitals: canonical HF frontier window"

    @staticmethod
    def _base_plan(record: TaskRecord, active: dict) -> tuple[Program, list[str]]:
        from chemflow.domains import pack_for_target

        return pack_for_target(record.spec.target.kind).base_plan(record, active)

    @staticmethod
    def _candidate_plan(plan: Program, candidates: list[dict]) -> tuple[Program, dict[str, str]]:
        from chemflow.ir.schema import Node

        trial = plan.model_copy(deep=True)
        ids = {}
        if not candidates:
            return trial, ids
        nodes = {node.id: node for node in trial.nodes}
        if not plan.references or plan.references[0] not in nodes:
            raise ValueError("plan has no reference node")
        reference = nodes[plan.references[0]]
        if reference.op in {"component_energy", "app_component_energy"}:
            from chemflow.domains.component_candidates import ground_reference

            additions, energy = ground_reference(nodes, reference)
            energy_plan = plan.model_copy(deep=True)
            energy_plan.nodes = [energy if node.id == reference.id else node
                                 for node in energy_plan.nodes] + additions
            expanded, identifiers = Orchestrator._candidate_plan(energy_plan, candidates)
            expanded.nodes = [reference if node.id == reference.id else node
                              for node in expanded.nodes]
            return expanded, identifiers
        if reference.op == "correlation_function":
            energy_plan = plan.model_copy(deep=True)
            energy_plan.references = [reference.inputs["state"]]
            energy_plan.targets = [reference.inputs["state"]]
            expanded, identifiers = Orchestrator._candidate_plan(energy_plan, candidates)
            expanded.targets = list(plan.targets)
            wrapped = {}
            for candidate, identifier in identifiers.items():
                result = Node(id=f"correlation_{identifier}", op="correlation_function",
                              inputs={"state": identifier}, params=dict(reference.params))
                expanded.nodes.append(result)
                expanded.targets.append(result.id)
                wrapped[candidate] = result.id
            expanded.references = list(plan.references)
            for node in expanded.nodes:
                if node.op == "prepare_state" and node.id not in nodes:
                    node.params["init"] = "random"
                    node.params["seed"] = 11
            return expanded, wrapped
        composites = {"reaction_energy", "ranking", "pes_scan", "interaction_scan",
                      "scan_interaction", "bond_dissociation"}
        if reference.op not in {"ground_energy", *composites}:
            raise ValueError(f"candidates are not available for {reference.op} references "
                             "(supported: ground_energy, reaction_energy, "
                             f"{', '.join(sorted(composites - {'reaction_energy'}))})")
        if reference.op in composites:
            for index, candidate in enumerate(candidates):
                inputs = {}
                for port, source in reference.inputs.items():
                    if port.startswith("thermo"):
                        inputs[port] = source
                        continue
                    species_plan = plan.model_copy(deep=True)
                    species_plan.references = [source]
                    species_plan.targets = [source]
                    expanded, identifiers = Orchestrator._candidate_plan(species_plan, [candidate])
                    if not identifiers:
                        continue
                    additions = [node for node in expanded.nodes if node.id not in nodes]
                    rename = {node.id: f"reaction{index}_{port}_{node.id}" for node in additions}
                    for node in additions:
                        node.id = rename[node.id]
                        node.inputs = {key: rename.get(value, value)
                                       for key, value in node.inputs.items()}
                    trial.nodes.extend(additions)
                    inputs[port] = rename[next(iter(identifiers.values()))]
                if len(inputs) == len(reference.inputs):
                    node = Node(id=f"reaction_candidate{index}", op=reference.op,
                                inputs=inputs, params=dict(reference.params))
                    trial.nodes.append(node)
                    trial.targets.append(node.id)
                    ids[str(candidate)] = node.id
            return trial, ids
        hamiltonian = nodes[reference.inputs["hamiltonian"]]
        sector = nodes[reference.inputs["sector"]]
        for index, candidate in enumerate(candidates):
            template = candidate["template"]
            if template == "vqe":
                raise ValueError("VQE statevector is verification-only; no deployable "
                                 "implementation: a QPU Pauli estimator is required")
            if template not in ("sci", "sqd", "vqe_sampled"):
                continue
            params = {"s": reference.params.get("s")}
            if template == "sci":
                params["select_cutoff"] = (1e-2 if candidate.get("select_cutoff") is None
                                           else candidate["select_cutoff"])
                binding = "sci_pyscf"
            elif template == "vqe_sampled":
                params.update(ansatz="uccsd", optimizer="cobyla", max_evaluations=60,
                              shots_per_group=candidate.get("shots") or 1024, seed=11)
                binding = "vqe_shots_ideal"
            else:
                prep = Node(id=f"prep{index}", op="prepare_state",
                    inputs={"hamiltonian": hamiltonian.id, "sector": sector.id},
                    params={"ansatz": "lucj", "init":
                            "zero" if hamiltonian.op == "lattice_model" else "ccsd"})
                trial.nodes.append(prep)
                sample = Node(id=f"sample{index}", op="sample_configurations",
                              inputs={"prep": prep.id},
                              params={"shots": (1024 if candidate.get("shots") is None
                                                else candidate["shots"]), "seed": 11})
                trial.nodes.append(sample)
                params.update(n_batches=2, max_iter=2)
                binding = "sqd_sector"
            node = Node(id=f"candidate{index}", op="ground_energy",
                        inputs={"hamiltonian": hamiltonian.id, "sector": sector.id,
                                **({"samples": sample.id} if template == "sqd" else {})},
                        params=params, refinement=template, binding=binding)
            trial.nodes.append(node)
            trial.targets.append(node.id)
            ids[str(candidate)] = node.id
        return trial, ids

    @staticmethod
    def _decode_run(data: dict, store=None) -> dict:
        from chemflow.runtime.evidence import decode_empirical_estimate

        return {**{node_id: (decode_empirical_estimate(estimate, store)
                            if estimate.get("kind") == "aqueous_solubility" else
                            (EnergyCurve if "values_hartree" in estimate else
                             SpectralEstimate if "spectral" in estimate
                             else Estimate).model_validate(estimate)) for node_id, estimate in
                   data["estimates"].items()},
                "headlines": data["headlines"],
                "refused": data.get("refused", {}),
                "budget_overruns": data.get("budget_overruns", []),
                "costs": {node_id: CostRecord.model_validate(cost) for node_id, cost in
                          data["costs"].items()},
                "gates": [GateVerdict.model_validate(gate) for gate in data["gates"]]}

    @staticmethod
    def _item(node_id: str, computed: dict, gates: dict,
              reference: str, spec: TaskSpec, plan: Program | None = None) -> dict:
        from chemflow.check.deployment import deployment

        item = result_summary(node_id, computed, gates, reference, spec, plan)
        if plan is not None:
            headline = computed.get("headlines", {}).get(node_id, {})
            if "deployable" in headline:
                item.update({key: headline[key] for key in (
                    "deployable", "deployment", "deployment_reason", "classical_nodes")})
            else:
                item.update(deployment(plan, molecular_registry(), targets=[node_id])[0])
        return item

    @staticmethod
    def _headline(node_id: str, computed: dict, gates: dict,
                  reference: str, spec: TaskSpec, plan: Program | None = None) -> dict:
        if "headlines" in computed:
            return dict(computed["headlines"][node_id])
        return result_headline(node_id, computed, gates, reference, spec, plan, molecular_registry())
