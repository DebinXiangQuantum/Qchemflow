"""LLM role policies on the QCL tools: prompts, argument decoding, permissions and turns.

Ports the engine-independent contracts of the retired legacy reviewer tests (R112 retries,
R116 JSON-encoded arguments, R128 turn reminders) to chemflow.agent.qcl_tools.
"""

import json

import pytest
from llm_tools import ToolRegistry
from pydantic import Field

from chemflow.agent.journal import Journal
from chemflow.agent.llm import CONTRACTS, PROMPTS, TERMINAL_TOOLS, FakeLLMClient, LLMPolicy
from chemflow.agent.qcl_tools import DESCRIPTIONS, CandidatesInput, QclToolRegistry, ToolInput


class Proposal(ToolInput):
    candidates: list[dict] = Field(description="Candidates")
    options: dict = Field(default_factory=dict, description="Options")


def _turn(name=None, arguments=None, content=""):
    return {"content": content, "tool_calls": [] if name is None else [
        {"id": f"call-{name}", "name": name, "arguments": arguments or {}}]}


def _journal(tmp_path):
    return Journal(tmp_path / "journal.jsonl")


def _entries(journal):
    return [json.loads(line) for line in journal.path.read_text().splitlines()]


def test_every_role_has_a_qcl_prompt_naming_its_terminal_tools():
    assert set(PROMPTS) == set(CONTRACTS)
    for role, tools in TERMINAL_TOOLS.items():
        assert tools <= set(DESCRIPTIONS)
        assert all(tool in PROMPTS[role] for tool in tools)
    client = FakeLLMClient([_turn(content="nothing to do")])
    LLMPolicy("critic", client, QclToolRegistry()).decide({})
    assert client.messages[0][0] == {"role": "system", "content": PROMPTS["critic"]}


def test_json_encoded_object_arguments_are_decoded_once(tmp_path):
    received = []
    tools = ToolRegistry()
    tools.register("design.submit", {"designer"},
                   lambda data: received.append(data.model_dump()) or "ok", Proposal)
    candidates = [{"template": "sci", "select_cutoff": 1e-3}]
    client = FakeLLMClient([_turn("design.submit", {
        "candidates": json.dumps(candidates), "options": json.dumps({"seed": 1})})])
    result = LLMPolicy("designer", client, tools, journal=_journal(tmp_path)).decide({})
    assert received == [{"candidates": candidates, "options": {"seed": 1}}]
    assert result["end_reason"] == "design.submit"


def test_the_last_two_turns_are_announced_to_roles_that_must_finish(tmp_path):
    tools = ToolRegistry()
    tools.register("methods.list", {"designer", "analyst"}, lambda data: [], ToolInput)
    looping = [_turn("methods.list")] * 3
    designer = FakeLLMClient(list(looping))
    LLMPolicy("designer", designer, tools, journal=_journal(tmp_path), max_turns=3).decide({})
    reminders = [message["content"] for message in designer.messages[-1]
                 if message["role"] == "user" and "turn(s) remain" in message["content"]]
    tail = "Finish with a terminal tool: design.decline, design.submit."
    assert reminders == [f"{count} turn(s) remain, including this one. {tail}"
                         for count in (2, 1)]
    analyst = FakeLLMClient(list(looping))
    LLMPolicy("analyst", analyst, tools, journal=_journal(tmp_path), max_turns=3).decide({})
    assert not any("turn(s) remain" in message["content"] for message in analyst.messages[-1]
                   if message["role"] == "user")


def test_terminal_tools_end_the_turn_and_later_calls_do_not_run(tmp_path):
    calls = []
    tools = ToolRegistry()
    tools.register("design.decline", {"designer"}, lambda data: calls.append("decline"))
    tools.register("methods.list", {"designer"}, lambda data: calls.append("list"))
    response = {"content": "", "tool_calls": [
        {"id": "a", "name": "design.decline", "arguments": {}},
        {"id": "b", "name": "methods.list", "arguments": {}}]}
    result = LLMPolicy("designer", FakeLLMClient([response]), tools,
                       journal=_journal(tmp_path)).decide({})
    assert calls == ["decline"] and result["end_reason"] == "design.decline"


@pytest.mark.parametrize("arguments,error", [
    ({"candidates": [], "gates": [{"kind": "abs_error_vs_exact"}]}, "PermissionError"),
    ({"candidates": [], "budget": {"wall_s": 1e9}}, "PermissionError"),
    ({"candidates": [], "unexpected": 1}, "ValueError"),
])
def test_task_gates_and_budget_are_immutable_and_unknown_arguments_refused(
        tmp_path, arguments, error):
    tools = ToolRegistry()
    tools.register("screen.review", {"designer"}, lambda data: "reviewed", CandidatesInput)
    result = LLMPolicy("designer", FakeLLMClient([_turn("screen.review", arguments), _turn()]),
                       tools, journal=_journal(tmp_path)).decide({})
    assert result["outputs"][0]["error"]["code"] == error


def test_tools_outside_the_role_are_refused(tmp_path):
    tools = ToolRegistry()
    tools.register("report.write", {"analyst"}, lambda data: "written")
    result = LLMPolicy("designer", FakeLLMClient([_turn("report.write"), _turn()]), tools,
                       journal=_journal(tmp_path)).decide({})
    assert result["outputs"][0]["error"] == {"code": "PermissionError",
                                             "message": "Tool outside role"}


def test_non_retryable_provider_errors_stop_at_once(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("chemflow.agent.llm.time.sleep", sleeps.append)
    failure = {"tool_calls": [], "content": "", "error": {
        "code": "invalid_request", "message": "bad request", "retryable": False}}
    journal = _journal(tmp_path)
    result = LLMPolicy("designer", FakeLLMClient([failure]), ToolRegistry(),
                       journal=journal).decide({})
    assert result["end_reason"] == "LLM_FAILED" and sleeps == []
    assert [entry["decision"] for entry in _entries(journal)
            if entry["state"] == "LLM_FAILED"] == [{"code": "invalid_request", "attempts": 1}]


def test_the_critic_verdict_is_deterministic_whatever_the_model_says(tmp_path):
    controls = [{"id": "ccsd", "method": "ccsd", "gate_passed": True, "error": 1e-4,
                 "relation": "approximate"}]
    result = LLMPolicy("critic", FakeLLMClient([_turn(content="Escalate everything!")]),
                       ToolRegistry(), journal=_journal(tmp_path)).decide(
        {"candidates": [], "controls": controls})
    assert result["verdict"]["decision"] == "stop"
    assert result["verdict"]["candidate"] == "ccsd"
