"""Normalize HTTP provider failures without mistaking exhausted credit for throttling."""

from __future__ import annotations

import re


def is_quota(code: str, message: str = "", kind: str = "") -> bool:
    exhausted = {"1113", "1308", "insufficient_quota", "quota_exceeded", "quota_exhausted",
                 "insufficient_balance", "billing_hard_limit_reached", "credit_balance_too_low"}
    return str(code) in exhausted or kind in exhausted or any(
        text in message.lower() for text in (
            "余额不足", "无可用资源包", "使用上限", "usage limit", "limit will reset",
            "insufficient balance", "insufficient quota", "credit balance",
            "quota exhausted", "exceeded your current quota"))


def quota_reset(message: str) -> str | None:
    if not any(marker in message.lower() for marker in ("reset", "重置")):
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?",
                      message)
    return match.group() if match else None


def status_error(exc) -> tuple[str, bool]:
    body = getattr(exc, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    details = details if isinstance(details, dict) else {}
    code = str(details.get("code") or getattr(exc, "code", "") or "")
    kind = str(details.get("type", ""))
    message = f"{details.get('message', '')} {exc}".lower()
    if is_quota(code, message, kind):
        return code or "quota_exhausted", False
    status = exc.status_code
    label = {400: "bad_request", 401: "authentication", 403: "permission", 408: "timeout",
             429: "rate_limit"}.get(status, "api_status")
    return code or label, status in {408, 429} or 500 <= status < 600
