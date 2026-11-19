from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("exception_name,status,body,message,retryable", [
    ("RateLimitError", 429, None, "slow down", True),
    ("AuthenticationError", 401, None, "bad key", False),
    ("PermissionDeniedError", 403, None, "not allowed", False),
    ("BadRequestError", 400, None, "invalid input", False),
    ("APIStatusError", 408, None, "request timeout", True),
    ("APIStatusError", 500, None, "server error", True),
    ("APIStatusError", 503, None, "unavailable", True),
    ("APIConnectionError", None, None, "connection failed", True),
    ("APITimeoutError", None, None, "timeout", True),
    ("RateLimitError", 429, {"error": {"code": "1113"}}, "余额不足或无可用资源包", False),
    ("RateLimitError", 429, {"code": 1113}, "balance depleted", False),
    ("RateLimitError", 429, {"error": {"code": "1302"}}, "rate limit", True),
    ("RateLimitError", 429, {"error": {"code": "insufficient_quota"}}, "quota", False),
    ("RateLimitError", 429, {"type": "insufficient_quota"}, "quota", False),
    ("RateLimitError", 429, None, "Your credit balance is too low", False),
])
def test_fake_sdk_errors(provider, exception_name, status, body, message, retryable):
    sdk = pytest.importorskip(provider)
    request = SimpleNamespace()
    exception_class = getattr(sdk, exception_name)
    if status is not None:
        exception = exception_class(message, response=SimpleNamespace(
            status_code=status, request=request, headers={}), body=body)
    elif exception_name == "APITimeoutError":
        exception = exception_class(request=request)
    else:
        exception = exception_class(message=message, request=request)
    requests = []

    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        raise exception

    endpoint = SimpleNamespace(create=create)
    if provider == "openai":
        from chemflow.agent.llm_openai import OpenAIClient

        adapter = OpenAIClient("offline", client=SimpleNamespace(
            chat=SimpleNamespace(completions=endpoint)))
    else:
        from chemflow.agent.llm_anthropic import ClaudeClient

        adapter = ClaudeClient(client=SimpleNamespace(beta=SimpleNamespace(messages=endpoint)))
    messages = [{"role": "system", "content": "Contract"}, {"role": "user", "content": "Task"}]
    for _ in range(2):
        result = adapter.complete(messages, [])
        assert result["error"]["retryable"] is retryable
        if body and isinstance(body.get("error", body), dict):
            expected_code = body.get("error", body).get("code")
            if expected_code is not None:
                assert result["error"]["code"] == str(expected_code)
    assert requests[0] == requests[1]
