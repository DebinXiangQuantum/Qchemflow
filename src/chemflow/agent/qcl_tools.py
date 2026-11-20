"""Role-scoped tools over QCL screens, for LLM policies and the MCP server.

Agents never edit the signed task, its gates or the controls and reference: they read the
task and the graded screen, choose an active space (modeler), and propose candidates from
a typed menu (designer) that the trusted screen builder compiles into QCL. Every value an
agent sees comes from a compiled or executed plan.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from chemflow.diagnostics import AdapterError
from chemflow.ids import canonical_json

ROLES = frozenset({"planner", "modeler", "designer", "critic", "analyst", "orchestrator"})
DESCRIPTIONS = {
    "task.get": "Read the signed task's question, target and gates (read-only).",
    "methods.list": "List the screen's classical ladder, exact reference and candidate "
                    "templates with their parameter ranges and target relations.",
    "diagnose.model": "Diagnose the signed molecule: RHF energy, HOMO-LUMO gap, MP2 natural "
                      "occupations and a suggested active space.",
    "model.choose": "Choose the active space {n_elec, n_orb, reason}; ends the modeler turn.",
    "screen.controls": "Read the graded controls and exact reference of the first round.",
    "screen.review": "Compile proposed candidates without running them: QCL plan hash, "
                     "inferred relations, predicted cost and simulator shots.",
    "design.submit": "Submit candidates for screening; ends the designer turn.",
    "design.decline": "Keep only the controls (give a task-specific reason); ends the "
                      "designer turn.",
    "results.query": "Read every graded output recorded so far.",
    "report.write": "Write the report text; every energy it quotes must match a result.",
}
TERMINAL = {"designer": {"design.submit", "design.decline"}, "modeler": {"model.choose"}}


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(ToolInput):
    template: Literal["sci", "sqd"]
    select_cutoff: float | None = Field(None, description="SCI selection cutoff (sci only)")
    shots: int | None = Field(None, description="Simulator shots (sqd only)")


class CandidatesInput(ToolInput):
    candidates: list[Candidate] = Field(description="Candidates to compile or screen")


class DeclineInput(ToolInput):
    reason: str = Field(min_length=1, description="Task-specific reason to keep controls")


class ChooseInput(ToolInput):
    n_elec: int = Field(description="Active electrons")
    n_orb: int = Field(description="Active orbitals (at most 10)")
    reason: str = Field("", description="Why this space")


class ReportInput(ToolInput):
    text: str = Field(min_length=1, description="Report text")


class ToolOutput(BaseModel):
    data: Any


def _strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    definitions = schema.pop("$defs", {})

    def inline(value: Any) -> Any:
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            resolved = inline(definitions[value["$ref"].removeprefix("#/$defs/")])
            return {**resolved, **{key: inline(item) for key, item in value.items()
                                   if key != "$ref"}}
        result = {key: inline(item) for key, item in value.items()}
        if result.get("type") == "object" or "properties" in result:
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
            for name, prop in result.get("properties", {}).items():
                prop.setdefault("description", name.replace("_", " ").capitalize())
        result.pop("default", None)
        result.pop("title", None)
        return result

    return inline(schema)


class QclToolRegistry:
    """The same interface as the legacy registry: ``schemas(role)`` and ``call``."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, frozenset[str], type[BaseModel]]] = {}
        self.task_record = None

    def register(self, name: str, roles: set[str], function: Callable,
                 input_model: type[BaseModel] = ToolInput) -> None:
        if not roles <= ROLES or name in self._tools or name not in DESCRIPTIONS:
            raise ValueError("Invalid tool registration")
        self._tools[name] = (function, frozenset(roles), input_model)

    def schemas(self, role: str) -> list[dict]:
        return [{"name": name, "description": DESCRIPTIONS[name],
                 "parameters": _strict_schema(model),
                 "output": ToolOutput.model_json_schema()}
                for name, (_, roles, model) in self._tools.items() if role in roles]

    def call(self, role: str, name: str, data: dict | None = None) -> dict:
        if name not in self._tools:
            raise PermissionError(f"Tool unavailable: {name}")
        function, roles, model = self._tools[name]
        if role not in roles:
            raise PermissionError(f"Role {role} cannot call {name}")
        inputs = model.model_validate(data or {})
        try:
            result = ToolOutput.model_validate({"data": function(inputs)}).model_dump(mode="json")
            return json.loads(canonical_json(result))
        except (ValueError, PermissionError, KeyError):
            raise
        except AdapterError as error:
            raise ValueError(str(error)) from error
        except Exception as exc:
            raise AdapterError("CF-E080", None, f"Tool {name} failed: {exc}",
                               "Check tool inputs") from exc


class ScreenState:
    """What the tools can see: the signed task and the graded screen so far."""

    def __init__(self, record: Any, cluster: Any, *, device_id: str | None = None):
        self.record = record
        self.cluster = cluster
        self.device_id = device_id
        self.active: dict[str, int] | None = None
        self.diagnostics: dict[str, Any] | None = None
        self.controls: list[dict] = []
        self.headlines: list[dict] = []
        self.remaining: dict[str, float] = {}
        self.choice: dict[str, Any] | None = None
        self.submitted: list[dict] | None = None
        self.declined: str | None = None
        self.report: str | None = None


def _task_view(record: Any) -> dict[str, Any]:
    spec = record.spec
    target = spec.target.model_dump(mode="json", exclude={"operator"})
    return {"question": spec.question, "target": target,
            "gates": [gate.model_dump(mode="json") for gate in spec.gates],
            "allowed_effects": list(spec.allowed_effects)}


def make_qcl_tools(state: ScreenState) -> QclToolRegistry:
    from chemflow.agent import screen as screens

    tools = QclToolRegistry()
    tools.task_record = state.record
    everyone = {"planner", "modeler", "designer", "critic", "analyst", "orchestrator"}
    spec = state.record.spec

    tools.register("task.get", everyone, lambda data: _task_view(state.record))

    def methods(data: ToolInput) -> dict:
        target = spec.target
        ladders = {"ranking": screens.REACTIVITY_LADDER,
                   "bond_dissociation": screens.REACTIVITY_LADDER,
                   "ionization_energy": screens.CHARGE_LADDER,
                   "electron_affinity": screens.CHARGE_LADDER,
                   "lattice_ground_energy": screens.LATTICE_LADDER}
        ladder = ladders.get(target.kind, screens.LADDER)
        energy = target.kind in ("molecular_ground_energy", "lattice_ground_energy",
                                 *screens.COMPOSITE)
        return {"controls": [{"method": method, "relation": "variational_upper_bound"
                              if method == "hf" else "approximate"} for method in ladder]
                if energy else [],
                "reference": ("ed" if target.kind in ("lattice_ground_energy",
                                                       "pairing_correlation")
                              else "fci" if energy else "model"),
                "candidates": [
                    {"template": "sci", "relation": "variational_upper_bound",
                     "select_cutoff": list(screens.SCI_CUTOFFS),
                     "cost": "grows as the cutoff falls; exact at zero"},
                    {"template": "sqd", "relation": "variational_upper_bound",
                     "shots": list(screens.SQD_SHOTS), "cost": "simulator shots",
                     "applies": "closed-shell molecular species"}] if energy else [],
                "scripted_rounds": [[{"template": method, ("select_cutoff" if method == "sci"
                                                           else "shots"): value}
                                     for method, value in round_]
                                    for round_ in screens.CANDIDATE_ROUNDS]}

    tools.register("methods.list", everyone, methods)

    def diagnose(data: ToolInput) -> dict:
        if state.diagnostics is None:
            raise ValueError("Diagnostics apply to automatic molecular active spaces")
        return state.diagnostics

    tools.register("diagnose.model", {"modeler"}, diagnose)

    def choose(data: ChooseInput) -> dict:
        state.choice = data.model_dump()
        return state.choice

    tools.register("model.choose", {"modeler"}, choose, ChooseInput)
    tools.register("screen.controls", {"designer", "critic", "analyst"},
                   lambda data: {"controls": state.controls})

    def review(data: CandidatesInput) -> dict:
        proposed = screens.candidate_entries(
            spec, [item.model_dump(exclude_none=True) for item in data.candidates],
            active=state.active)
        built = screens.build(spec, state.cluster, device_id=state.device_id,
                              active=state.active, controls=False, proposed=proposed)
        plan = built.workflow.compile(*built.args)
        predicted = plan.schedule().get("predicted", {})
        budget = state.remaining.get("sim_shots")
        return {"plan_hash": plan.hash, "candidates": [entry.name for entry in proposed],
                "relations": plan.relations(), "sim_shots": built.sim_shots,
                "affordable": budget is None or built.sim_shots <= budget,
                "predicted_s": predicted.get("sequential_s"),
                "predicted_makespan_s": predicted.get("makespan_s")}

    tools.register("screen.review", {"designer"}, review, CandidatesInput)

    def submit(data: CandidatesInput) -> dict:
        reviewed = review(data)
        if not reviewed["affordable"]:
            raise ValueError("The candidates exceed the remaining sim_shots budget")
        state.submitted = [item.model_dump(exclude_none=True) for item in data.candidates]
        return reviewed

    tools.register("design.submit", {"designer"}, submit, CandidatesInput)

    def decline(data: DeclineInput) -> dict:
        state.declined = data.reason
        return {"reason": data.reason}

    tools.register("design.decline", {"designer"}, decline, DeclineInput)
    tools.register("results.query", {"designer", "critic", "analyst"},
                   lambda data: {"results": state.headlines})

    def report(data: ReportInput) -> dict:
        from chemflow.agent.report_lint import lint_numbers

        lint_numbers(data.text, state.headlines)
        state.report = data.text
        return {"accepted": True}

    tools.register("report.write", {"analyst"}, report, ReportInput)
    return tools


DESCRIPTIONS.update({
    "screen.run": "Run the signed task's QCL screen and return the decision: the scripted "
                  "candidate rounds, or one round of your typed candidates. Every run "
                  "spends the task's signed budget.",
})


class RunInput(ToolInput):
    candidates: list[Candidate] | None = Field(
        None, description="Typed candidates for one round; omit for the scripted rounds")


def make_session_tools(record: Any, cluster: Any, directory: Any) -> QclToolRegistry:
    """MCP tools for an externally driven agent: read, compile and run signed screens.

    Runs share one output directory, so the task's budget ledger accumulates across calls
    and a run beyond the signed budget is refused.
    """
    import contextlib
    import sys
    from argparse import Namespace

    from chemflow.agent import screen as screens
    from chemflow.agent.qcl_run import run as run_screen

    state = ScreenState(record, cluster)
    state.active, state.diagnostics = screens.resolve_active(record.spec.target)
    tools = make_qcl_tools(state)
    tools._tools = {name: entry for name, entry in tools._tools.items()
                    if name in ("task.get", "methods.list", "diagnose.model",
                                "screen.review", "results.query")}
    for name, (function, roles, model) in list(tools._tools.items()):
        tools._tools[name] = (function, roles | {"orchestrator"}, model)

    def execute(data: RunInput) -> dict:
        proposals = ([item.model_dump(exclude_none=True) for item in data.candidates]
                     if data.candidates is not None else None)
        summary_path = directory / "summary.json"
        with contextlib.redirect_stdout(sys.stderr):
            try:
                summary = run_screen(Namespace(out=directory), record, cluster,
                                     policy="mcp", proposals=proposals)
            except SystemExit:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state.headlines = summary.get("results", [])
        return {key: summary.get(key) for key in
                ("decision", "candidate", "reason", "refusal", "spent", "limits",
                 "plan_hashes")} | {
            "results": [{key: item.get(key) for key in
                         ("node", "role", "method", "params", "value", "relation",
                          "gate_verdict", "error", "screening_reference",
                          "values_hartree", "ranking", "validity_status")
                         if item.get(key) is not None} for item in summary.get("results", [])],
            "report": (directory / "report.md").read_text(encoding="utf-8")}

    tools.register("screen.run", {"orchestrator"}, execute, RunInput)
    return tools
