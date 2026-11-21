"""Run a human-signed task without access to signing credentials."""

from __future__ import annotations

import os
import re
import sys

from chemflow.agent.llm import CONTRACTS, DEFAULT_MAX_TURNS, LLMPolicy
from chemflow.agent.task import TaskRecord
from chemflow.diagnostics import AdapterError, CheckError
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
                getattr(args, "screen", False) or
                getattr(args, "candidate_program", None) or
                getattr(args, "builtin", False) or
                getattr(args, "device", None) is not None):
            if args.policy != "scripted" or args.model is not None:
                raise ValueError("QCL programs use the scripted policy")
            from chemflow.agent.qcl import run_cli

            run_cli(args)
            return
        from chemflow.agent.qcl_run import run as run_screens

        roles = validate_options(args)
        try:
            record = TaskRecord.model_validate_json(args.task.read_text(encoding="utf-8"))
            TaskVerifier(args.public_key.read_bytes()).verify(record)
        except (OSError, ValueError, TypeError) as exc:
            raise AdapterError("CF-E120", None, "Invalid signed task or public key",
                               "Supply a signed task and its public key") from exc
        import qcl

        cluster = (qcl.Cluster.load(args.cluster) if getattr(args, "cluster", None)
                   else qcl.Cluster.local())
        if getattr(args, "dry_run", False):
            from chemflow.agent.qcl_run import review_screen

            review_screen(args, record, cluster)
            return
        policies = {}
        if args.policy != "scripted":
            from chemflow.agent.qcl_tools import QclToolRegistry

            policies = {role: LLMPolicy(role, client_factory(args.policy, role, args.model),
                                        QclToolRegistry(), record=record,
                                        max_turns=getattr(args, "max_turns", DEFAULT_MAX_TURNS),
                                        provider=args.policy) for role in roles}
        run_screens(args, record, cluster, policies=policies, policy=args.policy)
    except (CheckError, OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        message = str(exc)
        print(message if re.search(r"CF-E\d{3}", message) else f"CF-E080: {message}",
              file=sys.stderr)
        raise SystemExit(2) from None
