"""Provider-neutral, bounded and permissioned LLM policy (no network adapter)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Protocol

from chemflow.agent.critic import CriticDecision, decision_items, review
from chemflow.agent.journal import Journal
from chemflow.agent.llm_errors import is_quota, quota_reset
from chemflow.agent.task import TaskRecord
from chemflow.agent.tools import ToolRegistry
from chemflow.diagnostics import CheckError
from chemflow.text import cap_strings as _cap_strings

CONTRACTS = {
    "planner": "Draft the question and gates; never freeze or execute the task.",
    "modeler": "Choose the model and active space; do not change frozen gates.",
    "designer": "Compose and screen plans; checker and resource budget rules are mandatory.",
    "critic": "Challenge candidates against classical controls; never loosen gates.",
    "analyst": "Report relations, evidence, costs and limitations without overclaiming.",
}
FORBIDDEN = {"task", "task_record", "freeze", "gates", "gate", "evidence",
             "signature", "record_hash", "content_hash", "budget"}
DESIGNER_PROMPT = (
    "You are ChemFlow's designer. observation.base_plan_id holds the frozen task's controls "
    "and screening reference; do not change them or the gates. Candidates must be built on "
    "observation.base_plan_id; plans started with plan.new are refused. "
    "vqe_sampled is deployable on MQ02 through vqe_lqcloud (vqe_shots_ideal at screen scale), "
    "while statevector VQE remains verification-only. "
    "Add candidates with plan.candidates: sci and sqd apply to ground_energy references "
    "and componentwise to reaction_energy (including reaction barriers), ranking, pes_scan, "
    "interaction_scan and bond_dissociation references. "
    "Other reference kinds require composition with plan.add_node; mark candidate results "
    "as targets with target=true. Use registry tools to inspect operators and bindings. "
    "Retain the returned plan_id after each edit. Optionally check and run at scale=screen; "
    "Pass capped text_sha256 objects back as plan parameter values to reuse exact frozen text. "
    "Use algorithm.prove {plan_id, node_id} for a budgeted symbolic relation certificate "
    "and independent kernel verdict before claiming a bound or equality. "
    "Use algorithm.validity {plan_id, node_id} to map the signed domain and bar, and "
    "algorithm.falsify {plan_id, node_id} to search for implementation/relation counterexamples; "
    "a clean search is not a proof. "
    "resource budget rules are mandatory. Finish with design.submit {plan_id} to screen that "
    "plan, or design.decline {reason} to keep only the controls. Either ends your turn "
    "on success. Decline with plan_id only to refuse a specific plan as unaffordable; "
    "Submit identity-valid affordable candidates for screening rather than rejecting them from "
    "other-system card evidence, or decline with force=true and a task-specific reason. "
    "ChemFlow verifies the claim, and the run then ends as a budget refusal. "
    "Never assert your own evidence or gates."
)
MODELER_PROMPT = (
    "You are ChemFlow's modeler. Fixed active spaces are handled without consulting you. "
    "For active=auto, inspect observation.diagnostics and finish with "
    "model.choose {n_elec, n_orb, reason}. This ends your turn immediately. "
    "Choose positive integer counts with at most 10 orbitals, no more than two electrons "
    "per orbital, and electron/spin parity compatible with the frozen target. "
    "If you do not choose or your choice is infeasible, the diagnostics' suggested_active "
    "space is used. Free text is not a choice. Never change the frozen task or its gates."
)
TERMINAL_TOOLS = {"designer": {"design.submit", "design.decline"}, "modeler": {"model.choose"}}
DEFAULT_MAX_TURNS = 24


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


class FakeLLMClient:
    def __init__(self, responses: list[dict]):
        self.responses = iter(responses)
        self.messages: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        self.messages.append(list(messages))
        return next(self.responses)


def _summary(value: object) -> object:
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "kind": "summary"}
    return value


def _forbidden(value: object) -> bool:
    if isinstance(value, dict):
        return any((key.lower() in FORBIDDEN and not
                    ((key == "gates" and item == []) or
                     (key == "budget" and item == {}))) or
                   _forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_forbidden(item) for item in value)
    return False


def _redact(value, secret: str):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _decode_arguments(value, schema):
    options = [schema, *schema.get("anyOf", []), *schema.get("oneOf", [])]
    if isinstance(value, str):
        containers = {option.get("type") for option in options} & {"object", "array"}
        if containers:
            try:
                decoded = json.loads(value)
            except ValueError:
                pass
            else:
                if (("object" in containers and isinstance(decoded, dict)) or
                        ("array" in containers and isinstance(decoded, list))):
                    value = decoded
    for option in options:
        if isinstance(value, dict) and option.get("type") == "object":
            properties = option.get("properties", {})
            return {key: _decode_arguments(item, properties.get(key, {}))
                    for key, item in value.items()}
        if isinstance(value, list) and option.get("type") == "array":
            return [_decode_arguments(item, option.get("items", {})) for item in value]
    return value


class LLMPolicy:
    def __init__(self, role: str, client: LLMClient, tools: ToolRegistry,
                 record: TaskRecord | None = None, journal: Journal | None = None,
                 max_retries: int = 2, provider: str | None = None,
                 max_turns: int = DEFAULT_MAX_TURNS, retry_backoff_s: float = 2.0,
                 max_provider_attempts: int = 10):
        if role not in CONTRACTS:
            raise ValueError(f"Unknown LLM role: {role}")
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if (not isinstance(max_provider_attempts, int) or
                isinstance(max_provider_attempts, bool) or max_provider_attempts < 1):
            raise ValueError("max_provider_attempts must be a positive integer")
        if (not isinstance(retry_backoff_s, (int, float)) or
                isinstance(retry_backoff_s, bool) or not math.isfinite(retry_backoff_s) or
                retry_backoff_s < 0):
            raise ValueError("retry_backoff_s must be finite and nonnegative")
        self.role, self.client, self.tools = role, client, tools
        self.provider = provider
        self.record = record or getattr(tools, "task_record", None)
        self.journal = journal or Journal(Path(tempfile.mkdtemp(prefix="chemflow-llm-")) /
                                          "journal.jsonl")
        self.max_retries = max_retries
        self.max_turns = max_turns
        self.retry_backoff_s = retry_backoff_s
        self.max_provider_attempts = max_provider_attempts

    def _complete(self, messages: list[dict], schemas: list[dict], turn: int,
                  secret: str) -> dict:
        delay = min(self.retry_backoff_s, 60.0)
        for attempt in range(1, self.max_provider_attempts + 1):
            started = time.perf_counter()
            response = _redact(self.client.complete(messages, schemas), secret)
            if (not isinstance(response, dict) or
                    not isinstance(response.get("tool_calls"), list) or
                    not isinstance(response.get("content"), str)):
                raise TypeError("Malformed LLM response")
            error = response.get("error")
            if not error:
                return response
            self.journal.append(state="LLM_ERROR", role=self.role, observation={},
                                decision={"attempt": attempt, "code": error.get("code"),
                                          "error": error, "turn": turn,
                                          "provider": self.provider,
                                          "model": getattr(self.client, "model", None),
                                          "usage": response.get("usage", {}),
                                          "request_id": response.get("request_id", ""),
                                          "latency_s": time.perf_counter() - started})
            quota = is_quota(error.get("code"), error.get("message", ""))
            if quota or not error.get("retryable") or attempt == self.max_provider_attempts:
                decision = {"code": error.get("code"), "attempts": attempt}
                if quota:
                    decision["quota"] = True
                    reset_at = quota_reset(error.get("message", ""))
                    if reset_at:
                        decision["reset_at"] = reset_at
                self.journal.append(state="LLM_FAILED", role=self.role, observation={},
                                    decision=decision)
                return response
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise AssertionError("Provider attempt budget must be positive")

    def decide(self, observation: dict) -> dict:
        schemas = self.tools.schemas(self.role)
        allowed = {schema["name"]: schema for schema in schemas}
        spec = self.record.spec if self.record else None
        context = {"task": {"question": spec.question, "target": _summary(
            spec.target.model_dump(mode="json")),
            "gates": [gate.model_dump(mode="json") for gate in spec.gates],
            "budget": spec.budget, "decision": spec.decision} if spec else {},
            "observation": {key: ([{field: card[field] for field in
                                    ("id", "template", "target", "cost_model") if field in card}
                                   for card in value] if key == "cards" else
                                   _summary(value) if key == "diagnostics" else value)
                            for key, value in observation.items()}}
        prompt = {"designer": DESIGNER_PROMPT, "modeler": MODELER_PROMPT}.get(self.role, (
            f"You are ChemFlow's {self.role}. {CONTRACTS[self.role]}\n"
            "Use the available tools for plan changes. Only screen runs are allowed. "
            "Return a plan through a tool result; never assert your own evidence or gates."))
        if self.role == "planner":
            from chemflow.apps import catalog

            prompt += "\nUse app.draft for unsigned drafts. Application catalogue:\n" + "\n".join(
                f"{entry['id']}: {entry['description']}" for entry in catalog())
        messages = [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(_cap_strings(context), sort_keys=True)}]
        outputs = []
        failures = 0
        end_reason = "turn limit"
        for turn in range(1, self.max_turns + 1):
            remaining = self.max_turns - turn + 1
            terminal_tools = TERMINAL_TOOLS.get(self.role)
            if terminal_tools and remaining <= 2:
                messages.append({"role": "user", "content": (
                    f"{remaining} turn(s) remain, including this one. "
                    f"Finish with a terminal tool: {', '.join(sorted(terminal_tools))}.")})
            started = time.perf_counter()
            secret = os.environ.get("GLM_API_KEY", "") if self.provider == "glm" else ""
            try:
                response = self._complete(messages, schemas, turn, secret)
            except (ValueError, TypeError, StopIteration) as exc:
                response = {"tool_calls": [], "content": ""}
                error = {"code": "malformed_response", "message": _redact(str(exc), secret)}
                self.journal.append(state="LLM_ERROR", role=self.role, observation={},
                                    decision={**error, "turn": turn,
                                              "latency_s": time.perf_counter() - started})
                outputs.append({"error": error})
                failures += 1
                if failures > self.max_retries or turn == self.max_turns:
                    self.journal.append(state="LLM_FAILED", role=self.role, observation={},
                                        decision={"code": error["code"], "attempts": failures})
                    end_reason = "LLM_FAILED"
                    break
                messages.append({"role": "user", "content": json.dumps(_cap_strings(error))})
                continue
            if response.get("error"):
                outputs.append({"error": response["error"]})
                end_reason = "LLM_FAILED"
                break
            self.journal.append(state="LLM_CALL", role=self.role, observation={},
                                decision={"stop_reason": response.get("stop_reason"),
                                          "provider": self.provider,
                                          "model": getattr(self.client, "model", None),
                                          "turn": turn,
                                          "latency_s": time.perf_counter() - started,
                                          "usage": response.get("usage", {}),
                                          "request_id": response.get("request_id", ""),
                                          "error": response.get("error")})
            for index, call in enumerate(response["tool_calls"]):
                if isinstance(call, dict) and not call.get("id"):
                    call["id"] = f"call-{turn}-{index}"
            content = _cap_strings(response["content"])
            messages.append({"role": "assistant", "content": content if isinstance(content, str)
                             else json.dumps(content),
                             "tool_calls": _cap_strings(response["tool_calls"])})
            call_outputs = []
            for call in response["tool_calls"]:
                name = call.get("name") if isinstance(call, dict) else None
                args = call.get("arguments") if isinstance(call, dict) else None
                try:
                    if isinstance(call, dict) and call.get("arguments_error") is not None:
                        raise ValueError(f"Invalid tool arguments JSON: {call['arguments_error']}")
                    if not isinstance(name, str) or not isinstance(args, dict):
                        raise TypeError("Malformed tool call")
                    if name not in allowed:
                        raise PermissionError("Tool outside role")
                    args = _decode_arguments(args, allowed[name]["parameters"])
                    if (name.startswith("task.") and name != "task.get") or _forbidden(args):
                        raise PermissionError("Task, gates or evidence are immutable")
                    if name == "run" and args.get("scale") != "screen":
                        raise PermissionError("Only screening runs are permitted")
                    fields = allowed[name]["parameters"]["properties"]
                    if set(args) - set(fields):
                        raise ValueError("Unknown tool arguments")
                    result = self.tools.call(self.role, name, args)
                    output = {"name": name, "result": result}
                except CheckError as exc:
                    output = {"name": name, "error": {
                        "code": exc.diagnostics[0].code,
                        "message": str(exc),
                        "diagnostics": [issue.model_dump(mode="json")
                                        for issue in exc.diagnostics]}}
                except (ValueError, TypeError, PermissionError, KeyError) as exc:
                    output = {"name": name, "error": {"code": type(exc).__name__,
                                                      "message": str(exc)}}
                if secret:
                    output = _redact(output, secret)
                self.journal.append(state="LLM_TOOL", role=self.role, observation={},
                                    decision=output, calls=[call])
                outputs.append(output)
                call_outputs.append({"role": "tool", "tool_call_id": (
                    call.get("id") if isinstance(call, dict) else None) or
                                     f"call-{turn}-{len(call_outputs)}",
                                     "content": json.dumps(_cap_strings(output), sort_keys=True,
                                                           default=str),
                                     "is_error": "error" in output})
                if "error" not in output and name in TERMINAL_TOOLS.get(self.role, set()):
                    end_reason = name
                    break
            messages.extend(call_outputs)
            if end_reason in TERMINAL_TOOLS.get(self.role, set()):
                break
            if not response["tool_calls"]:
                end_reason = "text reply"
                break
            if any(output["is_error"] for output in call_outputs):
                failures += 1
                if failures > self.max_retries:
                    end_reason = "consecutive tool errors"
                    break
                messages.append({"role": "user", "content": "Correct structured tool errors."})
            else:
                failures = 0
        else:
            self.journal.append(state="LLM_TURN_LIMIT", role=self.role, observation={},
                                decision={"turns": turn})
        self.journal.append(state="LLM_DECISION", role=self.role, observation={},
                            decision={"turns": turn, "end_reason": end_reason})
        result = {"content": response["content"], "outputs": outputs, "end_reason": end_reason}
        if self.role == "critic":
            core = review(decision_items(observation.get("candidates", [])),
                          decision_items(observation.get("controls", [])))
            result["verdict"] = vars(CriticDecision(core.decision, core.reason, core.candidate))
        return result
