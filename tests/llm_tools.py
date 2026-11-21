"""A role-scoped tool registry for exercising LLMPolicy without running an engine.

It behaves like chemflow.agent.qcl_tools.QclToolRegistry (strict input schemas, role
checks, ``{"data": ...}`` results) but accepts any tool name.
"""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from chemflow.agent.qcl_tools import ROLES, ToolInput, ToolOutput, _strict_schema
from chemflow.diagnostics import AdapterError


class QueryInput(ToolInput):
    model_config = ConfigDict(extra="forbid")
    query: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, frozenset[str], type[BaseModel]]] = {}
        self.task_record = None

    def register(self, name: str, roles: set[str], function: Callable,
                 input_model: type[BaseModel] = ToolInput) -> None:
        if not roles <= ROLES or name in self._tools:
            raise ValueError("Invalid tool registration")
        self._tools[name] = (function, frozenset(roles), input_model)

    def schemas(self, role: str) -> list[dict]:
        return [{"name": name, "description": f"Test tool {name}",
                 "parameters": _strict_schema(model)}
                for name, (_, roles, model) in self._tools.items() if role in roles]

    def call(self, role: str, name: str, data: dict | None = None) -> dict:
        if name not in self._tools:
            raise PermissionError(f"Tool unavailable: {name}")
        function, roles, model = self._tools[name]
        if role not in roles:
            raise PermissionError(f"Role {role} cannot call {name}")
        inputs = model.model_validate(data or {})
        try:
            return ToolOutput.model_validate({"data": function(inputs)}).model_dump(mode="json")
        except (ValueError, PermissionError, KeyError):
            raise
        except AdapterError as error:
            raise ValueError(str(error)) from error
        except Exception as exc:  # as QclToolRegistry: unexpected failures are CF-E080
            raise AdapterError("CF-E080", None, f"Tool {name} failed: {exc}",
                               "Check tool inputs") from exc
