"""Run a human-signed task without access to signing credentials."""

from __future__ import annotations

import json
import os
import re
import sys

from chemflow.agent.llm import CONTRACTS, DEFAULT_MAX_TURNS, LLMPolicy
from chemflow.agent.orchestrator import Orchestrator
from chemflow.agent.task import TaskRecord
from chemflow.agent.tools import ToolRegistry
from chemflow.diagnostics import AdapterError, CheckError, EffectDenied
from chemflow.runtime.budget import BudgetLedger
from chemflow.signing import TaskVerifier

DEFAULT_GLM_MODEL = "glm-5.3"


def client_factory(provider: str, role: str, model: str | None):
    if provider == "anthropic":
        from chemflow.agent.llm_anthropic import ClaudeClient

        return ClaudeClient.for_role(role, **({"model": model} if model else {}))
    from chemflow.agent.llm_openai import OpenAIClient

    if provider == "glm":
        import openai

        return OpenAIClient(model=model or DEFAULT_GLM_MODEL, client=openai.OpenAI(
            api_key=os.environ["GLM_API_KEY"],
            base_url=os.environ.get("CHEMFLOW_GLM_BASE_URL",
                                    "https://open.bigmodel.cn/api/paas/v4/"),
            max_retries=0, timeout=120))
    return OpenAIClient(model=model)


def validate_options(args) -> list[str]:
    roles = [role.strip() for role in args.roles.split(",")]
    if any(role not in CONTRACTS for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("--roles must contain distinct roles: " + ", ".join(CONTRACTS))
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    max_turns = getattr(args, "max_turns", DEFAULT_MAX_TURNS)
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("--max-turns must be a positive integer")
    if args.policy != "scripted":
        if args.policy == "openai" and not args.model:
            raise ValueError("--model is required for openai")
        if os.environ.get("CHEMFLOW_LIVE_LLM") != "1":
            raise ValueError("Live policies require CHEMFLOW_LIVE_LLM=1")
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "glm": "GLM_API_KEY"}[args.policy]
        if not os.environ.get(key, "").strip():
            raise ValueError(f"Live policy requires {key}")
    return roles


def run(args) -> None:
    try:
        if (getattr(args, "program", None) is not None or
                getattr(args, "engine", "legacy") == "qcl" or
                getattr(args, "screen", False) or
                getattr(args, "candidate_program", None)):
            if args.policy != "scripted" or args.model is not None:
                raise ValueError("QCL programs use the scripted policy")
            from chemflow.agent.qcl import run_cli

            run_cli(args)
            return
        if (getattr(args, "cluster", None) is not None or
                getattr(args, "dry_run", False) or getattr(args, "device", None) is not None):
            raise ValueError("--cluster, --device and --dry-run require --program or --engine qcl")
        try:
            record = TaskRecord.model_validate_json(args.task.read_text())
            verifier = TaskVerifier(args.public_key.read_bytes())
            verifier.verify(record)
        except (OSError, ValueError, TypeError) as exc:
            raise AdapterError("CF-E120", None, "Invalid signed task or public key",
                               "Supply a signed task and its public key") from exc
        roles = validate_options(args)
        policies = {}
        if args.policy != "scripted":
            policies = {role: LLMPolicy(role, client_factory(args.policy, role, args.model),
                                        ToolRegistry(), record=record,
                                        max_turns=getattr(args, "max_turns", DEFAULT_MAX_TURNS),
                                        provider=args.policy) for role in roles}

        def approve(spec):
            if spec != record.spec:
                raise AdapterError("CF-E120", None, "Draft differs from the signed task",
                                   "Use the verified task unchanged")
            return record

        agent = Orchestrator(verifier, approve, args.out, max_rounds=args.rounds,
                             policies=policies, plan_directory=args.out)
        ledger_path = args.out / f"{record.content_hash}.ledger.json"
        refusal = None
        result = None
        try:
            result = agent.run(record.spec)
        except AdapterError as exc:
            if exc.diagnostics[0].code not in ("CF-E070", "CF-E080") and not isinstance(
                    exc, EffectDenied):
                raise
            refusal = {"code": exc.diagnostics[0].code, "node": exc.diagnostics[0].node,
                       "message": exc.diagnostics[0].message,
                       "fix_hints": exc.diagnostics[0].fix_hints,
                       "nothing_consumed": exc.nothing_consumed,
                       "spent_at_refusal": BudgetLedger(record.spec.budget, ledger_path).spent}
            agent.journal.append(state="REFUSED", role="orchestrator", observation=refusal,
                                 decision="refused", reason=str(exc))
            print(str(exc), file=sys.stderr)
        ledger = BudgetLedger(record.spec.budget, ledger_path)
        paths = {name: str(args.out / filename) for name, filename in (
            ("report", "report.md"), ("journal", "journal.jsonl"),
            ("results", "results.sqlite"), ("summary", "summary.json"))}
        paths["ledger"] = str(ledger_path)
        summary = {"record_hash": record.hash,
                   "decision": result.decision.decision if result else "refused",
                   "candidate": result.decision.candidate if result else None,
                   "reason": result.decision.reason if result else (
                       "Effect refusal" if refusal["code"] == "CF-E120" else
                       "Unsupported model or implementation" if refusal["code"] == "CF-E080" else
                       "Budget refusal"),
                   "results": result.results if result else [], "refusal": refusal,
                   "policy": args.policy, "roles": roles if policies else [],
                   "spent": ledger.spent, "limits": ledger.limits, "paths": paths}
        if record.spec.validity:
            summary["validity"] = [item["validity"] for item in summary["results"]
                                   if item.get("validity")]
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if refusal is not None:
            raise SystemExit(2)
        print(f"{summary['decision']} {paths['report']}")
    except (CheckError, OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        message = str(exc)
        print(message if re.search(r"CF-E\d{3}", message) else f"CF-E080: {message}",
              file=sys.stderr)
        raise SystemExit(2) from None
