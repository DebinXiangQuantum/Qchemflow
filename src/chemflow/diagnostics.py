from pydantic import BaseModel


class Diagnostic(BaseModel, frozen=True):
    code: str
    severity: str = "error"
    node: str | None = None
    message: str
    fix_hints: list[str] = []


class CheckError(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        if not diagnostics:
            raise ValueError("CheckError requires at least one diagnostic")
        self.diagnostics = diagnostics
        super().__init__("; ".join(f"{d.code}: {d.message}" for d in diagnostics))


class AdapterError(CheckError):
    def __init__(self, code: str, node: str | None, message: str, hint: str, *,
                 nothing_consumed: bool = False, incurred_cost: dict | None = None,
                 job_ids: tuple[str, ...] = ()):
        self.nothing_consumed = nothing_consumed
        self.incurred_cost = dict(incurred_cost or {})
        self.job_ids = job_ids
        super().__init__([Diagnostic(code=code, node=node, message=message, fix_hints=[hint])])


class EffectDenied(AdapterError):
    def __init__(self, node: str, effect: str):
        super().__init__("CF-E120", node, f"Effect {effect} not allowed by task",
                         "Freeze a new task version", nothing_consumed=True)


class TextReferenceError(AdapterError):
    def __init__(self, available: str):
        super().__init__("CF-E121", None, "Unknown frozen text hash or mismatched text_chars",
                         f"Reuse a frozen text reference with its exact length; available: {available}",
                         nothing_consumed=True)
