"""OpenAI adapter using SDK 3.28.0 ``client.chat.completions.create``.

Verified offline by inspecting ``Completions.create`` signature,
``ChatCompletion``, ``Choice``, ``ChatCompletionMessage``,
``ChatCompletionMessageFunctionToolCall``, ``CompletionUsage`` model fields,
``FunctionDefinition`` annotations (including ``strict``), and SDK errors.
The package is imported only when completing a request.
"""

from __future__ import annotations

import json
from copy import deepcopy

from chemflow.agent.llm_errors import status_error


class OpenAIClient:
    def __init__(self, model: str, client=None):
        if not model:
            raise ValueError("An explicit OpenAI model is required")
        self.model = model
        self.client = client
        self._neutral: list[dict] = []
        self._native: list[dict] = []
        self._assistant: dict | None = None

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        import openai

        if self.client is None:
            self.client = openai.OpenAI(max_retries=0)
        if messages[:len(self._neutral)] != self._neutral:
            raise ValueError("Neutral conversation prefix changed")
        prefix = deepcopy(self._native)
        for message in messages[len(self._neutral):]:
            role = message["role"]
            if role == "assistant":
                if message != self._assistant:
                    raise ValueError("Assistant turn cannot be rewritten")
                self._assistant = None
            elif role == "tool":
                self._native.append({"role": "tool", "tool_call_id": message["tool_call_id"],
                                     "content": message["content"]})
            else:
                if role not in {"system", "user"}:
                    raise ValueError("Unknown conversation role")
                self._native.append({"role": role, "content": message["content"]})
        assert self._native[:len(prefix)] == prefix
        self._neutral = deepcopy(messages)
        names = {f"cf_{name.encode().hex()}": name for name in (item["name"] for item in tools)}
        definitions = [{"type": "function", "function": {
            "name": safe, "description": item.get("description", "Inspect or update a plan"),
            "parameters": item["parameters"], "strict": True}}
            for item in tools for safe, original in names.items() if original == item["name"]]
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=deepcopy(self._native), tools=definitions,
                tool_choice="auto")
        except openai.APIStatusError as exc:
            code, retryable = status_error(exc)
            return self._error(code, str(exc), retryable)
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            return self._error("connection", str(exc), True)
        choice = response.choices[0]
        message = choice.message
        usage = response.usage
        calls = []
        for call in message.tool_calls or []:
            if call.type != "function":
                continue
            parsed = {"id": call.id, "name": names.get(call.function.name, call.function.name),
                      "arguments": call.function.arguments}
            try:
                parsed["arguments"] = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                parsed["arguments_error"] = str(exc)
            calls.append(parsed)
        result = {"tool_calls": calls, "content": message.content or "",
                  "stop_reason": choice.finish_reason,
                  "usage": {"input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens} if usage else {},
                  "request_id": getattr(response, "_request_id", "") or response.id,
                  "error": None}
        if message.refusal or choice.finish_reason not in {"stop", "tool_calls"}:
            result["error"] = {"code": "refusal" if message.refusal else
                               choice.finish_reason, "retryable":
                               choice.finish_reason == "length", "message": message.refusal}
            return result
        self._native.append({"role": "assistant", "content": message.content,
                             "tool_calls": [{"id": call.id, "type": "function",
                                             "function": {"name": call.function.name,
                                                          "arguments": call.function.arguments}}
                                            for call in (message.tool_calls or [])]})
        self._assistant = {"role": "assistant", "content": result["content"],
                           "tool_calls": calls}
        return result

    @staticmethod
    def _error(code: str, message: str, retryable: bool) -> dict:
        return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
                "request_id": "", "error": {"code": code, "message": message,
                                             "retryable": retryable}}
