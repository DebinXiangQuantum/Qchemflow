import hashlib
import json

import pytest

from chemflow.agent.journal import Journal
from chemflow.agent.llm import LLMPolicy, _cap_strings


def test_caps_nested_strings_without_mutation():
    text = "λ" * 2049
    payload = {"nested": [{"text": text}], "boundary": "x" * 2048}
    capped = _cap_strings(payload)
    assert capped["nested"][0]["text"] == {
        "text_chars": 2049, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "head": text[:200],
        "ref": "pass this object back as the parameter value to reuse the frozen text"}
    assert capped["boundary"] == payload["boundary"]
    assert payload["nested"][0]["text"] == text
    assert _cap_strings(capped) == capped


@pytest.mark.parametrize("tool_name", ["plan.render", "registry.cards", "results.get"])
def test_all_conversation_payloads_are_capped(tool_name, tmp_path):
    text = "payload" * 1000

    class Tools:
        def schemas(self, role):
            return [{"name": tool_name, "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}}}]

        def call(self, role, name, args):
            assert args == {"query": text}
            return {"data": {"nested": [text]}}

    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, messages, schemas):
            self.calls.append(json.loads(json.dumps(messages)))
            return {"content": text, "tool_calls": [{"id": "call", "name": tool_name,
                "arguments": {"query": text}}] if len(self.calls) == 1 else []}

    client = Client()
    policy = LLMPolicy("designer", client, Tools(), journal=Journal(tmp_path / "journal.jsonl"))
    policy.decide({"nested": [text]})
    assert len(client.calls) == 2
    expected = _cap_strings(text)
    messages = client.calls[-1]
    assert json.loads(messages[1]["content"])["observation"]["nested"] == [expected]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert json.loads(assistant["content"]) == expected
    assert assistant["tool_calls"][0]["arguments"]["query"] == expected
    tool = next(message for message in messages if message["role"] == "tool")
    assert json.loads(tool["content"])["result"]["data"]["nested"] == [expected]
