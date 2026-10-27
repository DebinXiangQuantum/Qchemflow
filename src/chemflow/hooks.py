"""Portable, fail-closed framework hook. Hooks are defence in depth, not authority."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

from chemflow.agent.tools import DESCRIPTIONS
from chemflow.ids import canonical_json

WRITE_TOOLS = frozenset({"write", "edit", "multiedit", "notebookedit", "apply_patch",
                         "write_file", "edit_file", "create_file", "delete_file",
                         "replace_file", "str_replace_editor", "file_write", "file_edit"})
SHELL_TOOLS = frozenset({"bash", "shell", "execute", "execute_command", "run_command",
                         "terminal", "run_shell_command"})
TARGET_KEYS = ("file_path", "path", "target_path", "destination", "dest", "notebook_path")


def _inside(candidate: str, directories: tuple[str, ...], cwd: Path) -> bool:
    if not candidate or candidate.startswith("-"):
        return False
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()
    return any(path.is_relative_to(Path(directory).expanduser().resolve()) for directory in directories)


def _shell_targets(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    targets = []
    segment = []
    for index, token in enumerate(tokens):
        if token.startswith(">") and set(token) == {">"} and index + 1 < len(tokens):
            targets.append(tokens[index + 1])
        if token in ("|", "||", "&&", ";", "&") or index == len(tokens) - 1:
            if index == len(tokens) - 1 and token not in ("|", "||", "&&", ";", "&"):
                segment.append(token)
            words = [word for word in segment if word not in ("<", ">", ">>")]
            for position, word in enumerate(words):
                verb = Path(word).name
                operands = [item for item in words[position + 1:] if not item.startswith("-")]
                if verb == "tee":
                    targets.extend(operands)
                elif verb in ("cp", "mv") and operands:
                    targets.append(operands[-1])
                elif verb in ("rm", "chmod"):
                    targets.extend(operands[1:] if verb == "chmod" else operands)
                elif verb == "sed" and any(item == "-i" or item.startswith("-i") for item in
                                             words[position + 1:]):
                    targets.extend(operands[1:])
            segment = []
        else:
            segment.append(token)
    return targets


def _write_targets(name: str, arguments: dict) -> list[str]:
    tool = name.rsplit("__", 1)[-1].lower()
    if tool in WRITE_TOOLS:
        paths = [arguments[key] for key in TARGET_KEYS if isinstance(arguments.get(key), str)]
        if tool == "apply_patch":
            patch = str(arguments.get("patch", arguments.get("patch_text", "")))
            paths.extend(line.split(": ", 1)[1] for line in patch.splitlines()
                         if line.startswith(("*** Add File: ", "*** Update File: ",
                                             "*** Delete File: ", "*** Move to: ")))
        return paths
    if tool in SHELL_TOOLS:
        return _shell_targets(str(arguments.get("command", arguments.get("cmd", ""))))
    return []


def normalize_tool(name: str) -> str | None:
    for tool in sorted(DESCRIPTIONS, key=len, reverse=True):
        suffix = tool.replace(".", "_")
        if name == suffix or name == tool or name.endswith(("__" + suffix, "_" + suffix)):
            return tool
    return None


def event_kind(event: dict) -> str:
    name = str(event.get("hook_event_name", event.get("event", ""))).lower()
    if "pre" in name or "before" in name:
        return "pre"
    if "post" in name or "after" in name:
        return "post"
    if "stop" in name or "session_end" in name:
        return "stop"
    raise ValueError("Unknown hook event")


def journal_path(event: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    session = str(event.get("session_id") or event.get("tool_call_id") or
                  datetime.now(UTC).date().isoformat())
    safe_session = "".join(character if character.isalnum() or character in "-_" else "_"
                           for character in session)
    return Path.home() / ".chemflow/journals" / f"journal-{safe_session}.jsonl"


def _record(path: Path, event: dict, tool: str, arguments: dict, *, blocked: str | None = None) -> None:
    entry = {"event": "blocked" if blocked else "post", "tool": tool,
             "session_id": event.get("session_id") or event.get("tool_call_id"),
             "profile": event.get("profile"),
             "scale": arguments.get("scale", "screen") if tool == "run" else None,
             "outcome": "blocked" if blocked else
             "error" if event.get("is_error") or event.get("tool_error") else "ok"}
    if blocked:
        entry["reason"] = blocked
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(entry) + "\n")
    except OSError:
        if not blocked:
            raise


def evaluate(event: dict, *, journal: str | None = None,
             protected: tuple[str, ...] | None = None) -> str | None:
    kind = event_kind(event)
    name = str(event.get("tool_name", ""))
    tool = normalize_tool(name)
    arguments = event.get("tool_input") or {}
    if not isinstance(arguments, dict):
        raise TypeError("Invalid tool input")
    path = journal_path(event, journal)
    if kind == "pre":
        directories = protected if protected is not None else (str(Path.home() / ".chemflow"),)
        cwd = Path(event.get("cwd") or event.get("working_dir") or ".").resolve()
        if tool is None and any(_inside(target, directories, cwd) for target in
                                _write_targets(name, arguments)):
            reason = "Framework tool cannot modify protected ChemFlow paths"
            _record(path, event, name, arguments, blocked=reason)
            return reason
        return None
    if kind == "post":
        if tool is not None:
            _record(path, event, tool, arguments)
        return None
    entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    screened = any(item.get("tool") == "run" and item.get("scale") == "screen" and
                   item.get("outcome") == "ok" for item in entries)
    reported = any(item.get("tool") == "report.write" and item.get("outcome") == "ok"
                   for item in entries)
    if screened and not reported:
        reason = "ChemFlow screened results require a successful report_write before stop"
        _record(path, event, name, arguments, blocked=reason)
        return reason
    return None


def main() -> None:
    parser = argparse.ArgumentParser(prog="chemflow-hook", allow_abbrev=False)
    parser.add_argument("--journal")
    parser.add_argument("--protect", action="append")
    args = parser.parse_args()
    try:
        event = json.load(sys.stdin)
        reason = evaluate(event, journal=args.journal,
                          protected=tuple(args.protect) if args.protect else None)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        reason = "ChemFlow hook failed closed"
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)
