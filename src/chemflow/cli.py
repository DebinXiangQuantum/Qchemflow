"""Out-of-band human operations; never registered as agent tools."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import sys
from functools import partial
from pathlib import Path

import yaml

from chemflow.agent.llm import DEFAULT_MAX_TURNS
from chemflow.agent.task import TaskSpec
from chemflow.agent.tools import DESCRIPTIONS
from chemflow.mcp import PROFILES, create_tools
from chemflow.plugin_build import TARGETS, build
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def _passphrase() -> bytes:
    return (os.environ.get("CHEMFLOW_KEY_PASSPHRASE") or getpass.getpass(
        "ChemFlow key passphrase: ")).encode()


def _signer(directory: Path) -> TaskSigner:
    private = directory / "private.pem"
    if stat.S_IMODE(private.stat().st_mode) != 0o600:
        raise ValueError("Private key must be 0600")
    return TaskSigner(private.read_bytes(), _passphrase())


def main() -> None:
    parser = argparse.ArgumentParser(prog="chemflow", allow_abbrev=False)
    parser.add_argument("--key-dir", type=Path,
                        default=Path(os.environ.get("CHEMFLOW_KEY_DIR", ".chemflow-keys")))
    commands = parser.add_subparsers(dest="command", required=True,
                                     parser_class=partial(argparse.ArgumentParser, allow_abbrev=False))
    from chemflow.apps.cli import configure

    configure(commands)
    agent = commands.add_parser("agent")
    agent_commands = agent.add_subparsers(dest="action", required=True,
        parser_class=partial(argparse.ArgumentParser, allow_abbrev=False))
    agent_run = agent_commands.add_parser("run")
    agent_run.add_argument("--task", type=Path, required=True)
    agent_run.add_argument("--public-key", type=Path, required=True)
    agent_run.add_argument("--out", type=Path, required=True)
    agent_run.add_argument("--program", type=Path)
    agent_run.add_argument("--engine", choices=("legacy", "qcl"), default="legacy")
    agent_run.add_argument("--device")
    agent_run.add_argument("--cluster", type=Path)
    agent_run.add_argument("--dry-run", action="store_true")
    agent_run.add_argument("--screen", action="store_true",
                           help="compare reviewed QCL candidate programs")
    agent_run.add_argument("--candidate-program", type=Path, action="append",
                           help="QCL program to include in a signed candidate screen")
    agent_run.add_argument("--policy", choices=("scripted", "anthropic", "openai", "glm"),
                           default="scripted")
    agent_run.add_argument("--model")
    agent_run.add_argument("--roles", default="designer")
    agent_run.add_argument("--rounds", type=int, default=2)
    agent_run.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                           help="Maximum model calls per LLM role turn (default: 24)")
    qcl_run = agent_commands.add_parser("qcl-run")
    qcl_run.add_argument("--task", type=Path, required=True)
    qcl_run.add_argument("--public-key", type=Path, required=True)
    qcl_run.add_argument("--program", type=Path, required=True)
    qcl_run.add_argument("--cluster", type=Path)
    qcl_run.add_argument("--out", type=Path, required=True)
    qcl_run.add_argument("--dry-run", action="store_true")
    evaluation = commands.add_parser("eval")
    evaluations = evaluation.add_subparsers(dest="action", required=True)
    evaluations.add_parser("list")
    evaluation_run = evaluations.add_parser("run")
    suite = evaluation_run.add_mutually_exclusive_group(required=True)
    suite.add_argument("--suite")
    suite.add_argument("--suite-file", type=Path)
    evaluation_run.add_argument("--policy", choices=("scripted", "anthropic", "openai", "glm"),
                                required=True)
    evaluation_run.add_argument("--model")
    evaluation_run.add_argument("--roles", default="designer")
    evaluation_run.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                                help="Maximum model calls per LLM role turn (default: 24)")
    evaluation_run.add_argument("--tasks")
    evaluation_run.add_argument("--jobs", type=int, default=1,
                                help="Number of independent task worker processes (default: 1)")
    evaluation_run.add_argument("--out", type=Path, required=True)
    keys = commands.add_parser("keys")
    keys.add_argument("action", choices=("init",))
    keys.add_argument("--dir", type=Path, required=True)
    task = commands.add_parser("task")
    tasks = task.add_subparsers(dest="action", required=True,
                                parser_class=partial(argparse.ArgumentParser, allow_abbrev=False))
    draft = tasks.add_parser("draft")
    draft.add_argument("--from", dest="source", required=True)
    freeze = tasks.add_parser("freeze")
    freeze.add_argument("--from", dest="source")
    freeze.add_argument("--approver", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--activate", action="store_true")
    tool = commands.add_parser("tool")
    calls = tool.add_subparsers(dest="action", required=True,
                                parser_class=partial(argparse.ArgumentParser, allow_abbrev=False))
    call = calls.add_parser("call")
    call.add_argument("--profile", choices=PROFILES, required=True)
    call.add_argument("--task", required=True)
    call.add_argument("--public-key", required=True)
    call.add_argument("--key-fingerprint", required=True)
    call.add_argument("--name", required=True)
    call.add_argument("--args-json", required=True)
    call.add_argument("--packs", default="molecular")
    plugin = commands.add_parser("plugin")
    builds = plugin.add_subparsers(dest="action", required=True,
                                   parser_class=partial(argparse.ArgumentParser, allow_abbrev=False))
    package = builds.add_parser("build")
    package.add_argument("--target", choices=TARGETS, required=True)
    package.add_argument("--out", type=Path, required=True)
    package.add_argument("--public-key", type=Path, required=True)
    package.add_argument("--task", type=Path)
    package.add_argument("--profiles")
    package.add_argument("--packs")
    package.add_argument("--server-command", type=Path)
    package.add_argument("--hook-command", type=Path)
    plan = commands.add_parser("plan")
    plan_commands = plan.add_subparsers(dest="action", required=True)
    show = plan_commands.add_parser("show")
    show.add_argument("--plan-id", required=True)
    show.add_argument("--plan-dir", type=Path, required=True)
    show.add_argument("--qpu", choices=("MQ02",))
    journal = commands.add_parser("journal")
    journal.add_argument("action", choices=("show",))
    journal.add_argument("path")
    qpu = commands.add_parser("qpu")
    qpu_commands = qpu.add_subparsers(dest="action", required=True)
    ledger = qpu_commands.add_parser("ledger")
    ledger.add_argument("--backend", default="MQ02")
    ledger.add_argument("--state-dir", type=Path)
    reconcile = qpu_commands.add_parser("reconcile")
    reconcile.add_argument("key")
    action = reconcile.add_mutually_exclusive_group(required=True)
    action.add_argument("--abandon", action="store_true")
    action.add_argument("--job-id")
    reconcile.add_argument("--backend", default="MQ02")
    reconcile.add_argument("--state-dir", type=Path)
    bench = commands.add_parser("bench")
    benchmarks = bench.add_subparsers(dest="action", required=True)
    benchmarks.add_parser("list")
    benchmark_run = benchmarks.add_parser("run")
    benchmark_run.add_argument("pack")
    benchmark_run.add_argument("name")
    benchmark_run.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "app":
            from chemflow.apps.cli import run

            run(args)
        elif args.command == "eval":
            from chemflow.eval.runner import list_suites, run

            if args.action == "list":
                print("\n".join(list_suites()))
            else:
                run(args)
        elif args.command == "agent":
            if args.action == "qcl-run":
                from chemflow.agent.qcl import run_cli

                run_cli(args)
            else:
                from chemflow.agent.cli import run

                run(args)
        elif args.command == "bench":
            from chemflow.domains.benchmarks import list_benchmarks, run_benchmark

            if args.action == "list":
                for pack, name in list_benchmarks():
                    print(f"{pack} {name}")
            else:
                report = run_benchmark(args.pack, args.name)
                text = json.dumps(report, indent=2)
                if args.json_out:
                    args.json_out.write_text(text + "\n")
                print(text)
        elif args.command == "keys":
            args.dir.mkdir(parents=True, exist_ok=True)
            private, public = generate_keypair(_passphrase())
            descriptor = os.open(args.dir / "private.pem", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(private)
            (args.dir / "public.pem").write_bytes(public)
            print(TaskVerifier(public).fingerprint)
        elif args.command == "task":
            source = Path(args.source).read_text() if args.source else sys.stdin.read()
            spec = TaskSpec.model_validate(yaml.safe_load(source))
            if args.action == "draft":
                print(spec.model_dump_json(indent=2))
            else:
                record = _signer(args.key_dir).freeze(spec, args.approver)
                Path(args.out).write_text(record.model_dump_json(indent=2) + "\n")
                if args.activate:
                    pointer = Path.home() / ".chemflow/tasks/current.json"
                    pointer.parent.mkdir(parents=True, exist_ok=True)
                    pointer.write_text(record.model_dump_json(indent=2) + "\n")
        elif args.command == "tool":
            tools = create_tools(args.profile, args.task, public_key=args.public_key,
                                 key_fingerprint=args.key_fingerprint, packs=args.packs)
            name = next((candidate for candidate in DESCRIPTIONS
                         if candidate.replace(".", "_") == args.name), args.name)
            arguments = json.loads(args.args_json)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            role = "orchestrator" if args.profile == "full" else args.profile
            if args.profile == "full":
                role = next((candidate for candidate in PROFILES[:-1]
                             if name in {item["name"] for item in tools.schemas(candidate)}), role)
            print(json.dumps(tools.call(role, name, arguments)))
        elif args.command == "plugin":
            build(args.target, args.out, args.public_key, task=args.task,
                  profiles=args.profiles, server_command=args.server_command,
                  hook_command=args.hook_command, packs=args.packs)
        elif args.command == "plan":
            from chemflow.plan_view import show_stored_plan

            print(show_stored_plan(args.plan_id, args.plan_dir, args.qpu))
        elif args.command == "qpu":
            from chemflow.backends.lqcloud import LqcloudBackend

            backend = LqcloudBackend(backend=args.backend, state_dir=args.state_dir)
            if args.action == "reconcile":
                record = backend.reconcile(args.key, abandon=args.abandon, job_id=args.job_id)
                print(f"{record['key']} reconciled")
            else:
                totals = {}
                paths = sorted(backend.state_dir.glob("*.json"))
                paths += sorted((backend.state_dir / "archive").glob("*.json"))
                for path in paths:
                    record = json.loads(path.read_text())
                    device = record.get("backend", args.backend)
                    total = totals.setdefault(device, {"qpu_shots": 0, "qpu_s": 0, "unresolved": 0})
                    state = record["state"]
                    predicted = record.get("predicted_cost", {})
                    status = record.get("status_info", {})
                    shots = seconds = 0
                    if state in ("prepared", "submitted"):
                        total["unresolved"] += 1
                    elif state in ("completed", "failed"):
                        shots = status.get("shots")
                        shots = shots if shots is not None else record.get("shots", 0)
                        seconds = status.get("execution_time")
                        seconds = seconds if seconds is not None else predicted.get("qpu_s", 0)
                    total["qpu_shots"] += shots or 0
                    total["qpu_s"] += seconds or 0
                    location = " archived" if path.parent.name == "archive" else ""
                    print(f"{record.get('key', path.stem)} {state}{location} {device} "
                          f"{record.get('job_id', '-')} qpu_shots={shots} qpu_s={seconds} "
                          f"{record.get('created', '-')}")
                for device, total in sorted(totals.items()):
                    print(f"total {device}: qpu_shots={total['qpu_shots']:g} "
                          f"qpu_s={total['qpu_s']:g}; unresolved {total['unresolved']}")

        else:
            for line in Path(args.path).read_text().splitlines():
                print(json.dumps(json.loads(line), indent=2))
    except (OSError, ValueError, TypeError, KeyError, PermissionError, yaml.YAMLError) as exc:
        if args.command == "tool":
            print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
            raise SystemExit(2) from None
        if args.command == "eval":
            print(f"ChemFlow evaluation failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        print("ChemFlow operation failed: invalid input or integrity check", file=sys.stderr)
        raise SystemExit(2) from None
