"""Claude adapter for the neutral LLM client; no SDK import at package import time."""

from __future__ import annotations

from copy import deepcopy

from chemflow.agent.llm_errors import status_error


class ClaudeClient:
    @classmethod
    def for_role(cls, role: str, **overrides):
        models = {"designer": ("claude-opus-5-5", "high"),
                  "critic": ("claude-opus-5-5", "high"),
                  "planner": ("claude-sonnet-5-5", "medium"),
                  "modeler": ("claude-sonnet-5-5", "medium"),
                  "analyst": ("claude-sonnet-5-5", "medium")}
        model, effort = models[role]
        return cls(**{"model": model, "effort": effort, **overrides})

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high",
                 fallbacks: bool = True, client=None):
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid Claude effort")
        self.model, self.effort, self.fallbacks = model, effort, fallbacks
        self.client = client
        self._native: list[dict] = []
        self._neutral: list[dict] = []
        self._assistant: dict | None = None
        self._system: str | None = None

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        import anthropic

        if self.client is None:
            self.client = anthropic.Anthropic(max_retries=0)
        if messages[:len(self._neutral)] != self._neutral:
            raise ValueError("Neutral conversation prefix changed")
        native_prefix = deepcopy(self._native)
        added = messages[len(self._neutral):]
        if not self._neutral:
            if not added or added[0]["role"] != "system":
                raise ValueError("Missing stable system contract")
            self._system = added[0]["content"]
            added = added[1:]
        pending_results = []
        for message in added:
            role = message["role"]
            if role == "system":
                raise ValueError("System contract cannot change")
            if role == "assistant":
                if message != self._assistant:
                    raise ValueError("Assistant turn cannot be rewritten")
                self._assistant = None
            elif role == "tool":
                pending_results.append({"type": "tool_result",
                                        "tool_use_id": message["tool_call_id"],
                                        "content": message["content"],
                                        "is_error": message["is_error"]})
            else:
                if pending_results:
                    self._native.append({"role": "user", "content": pending_results})
                    pending_results = []
                self._native.append({"role": "user", "content": message["content"]})
        if pending_results:
            self._native.append({"role": "user", "content": pending_results})
        assert self._native[:len(native_prefix)] == native_prefix
        self._neutral = deepcopy(messages)
        names = {item["name"].replace(".", "__"): item["name"] for item in tools}
        if len(names) != len(tools):
            raise ValueError("Tool name collision")
        definitions = [{"name": safe, "description": item.get("description", "Use this tool to inspect or update a ChemFlow plan."),
                        "input_schema": item["parameters"], "strict": True}
                       for safe, name in sorted(names.items())
                       for item in tools if item["name"] == name]
        kwargs = {"model": self.model, "max_tokens": 16000,
                  "output_config": {"effort": self.effort},
                  "system": [{"type": "text", "text": self._system,
                              "cache_control": {"type": "ephemeral"}}],
                  "messages": deepcopy(self._native), "tools": definitions,
                  "tool_choice": {"type": "auto"}}
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            code, retryable = status_error(exc)
            return self._error(code, str(exc), retryable)
        except anthropic.APIConnectionError as exc:
            return self._error("connection", str(exc), True)
        usage = {key: getattr(response.usage, key, 0) or 0 for key in (
            "cache_read_input_tokens", "cache_creation_input_tokens",
            "input_tokens", "output_tokens")}
        result = {"tool_calls": [], "content": "", "stop_reason": response.stop_reason,
                  "usage": usage, "request_id": getattr(response, "_request_id", ""),
                  "error": None}
        if response.stop_reason == "refusal":
            details = response.stop_details
            result["error"] = {"code": "refusal", "retryable": False,
                               "category": getattr(details, "category", None),
                               "explanation": getattr(details, "explanation", None)}
            return result
        result["content"] = "\n".join(block.text for block in response.content
                                      if block.type == "text")
        result["tool_calls"] = [{"id": block.id, "name": names.get(block.name, block.name),
                                 "arguments": block.input}
                                for block in response.content if block.type == "tool_use"]
        if response.stop_reason not in {"end_turn", "tool_use"}:
            result["error"] = {"code": response.stop_reason or "unknown_stop",
                               "retryable": response.stop_reason in {"max_tokens", "pause_turn"}}
            return result
        self._native.append({"role": "assistant", "content": response.content})
        self._assistant = {"role": "assistant", "content": result["content"],
                           "tool_calls": result["tool_calls"]}
        return result

    @staticmethod
    def _error(code: str, message: str, retryable: bool) -> dict:
        return {"tool_calls": [], "content": "", "stop_reason": "error", "usage": {},
                "request_id": "", "error": {"code": code, "message": message,
                                             "retryable": retryable}}
