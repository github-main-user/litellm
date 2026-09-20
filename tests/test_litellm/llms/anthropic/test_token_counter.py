from typing import Final

import httpx
import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.types.utils import CallTypes


class StubHTTPClient:
    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response

    async def post(self, *args, **kwargs) -> httpx.Response:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
async def test_handler_preserves_upstream_status_and_retry_after(monkeypatch) -> None:
    from litellm.llms.anthropic.common_utils import AnthropicError

    response = httpx.Response(
        429,
        text='{"type":"error"}',
        headers={"Retry-After": "17"},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
    )
    monkeypatch.setattr(
        "litellm.llms.anthropic.count_tokens.handler.get_async_httpx_client",
        lambda **kwargs: StubHTTPClient(response),
    )

    with pytest.raises(AnthropicError) as exc_info:
        await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-test",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-ant-api-test",
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "17"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected_status"),
    [
        (b"not-json", 502),
        (b'{"input_tokens": true}', 502),
        (b'{"input_tokens": -1}', 502),
        (b'{"input_tokens": "private-upstream-data"}', 502),
    ],
)
async def test_handler_rejects_invalid_success_responses(monkeypatch, content, expected_status) -> None:
    from litellm.llms.anthropic.common_utils import AnthropicError

    response = httpx.Response(
        200,
        content=content,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
    )
    monkeypatch.setattr(
        "litellm.llms.anthropic.count_tokens.handler.get_async_httpx_client",
        lambda **kwargs: StubHTTPClient(response),
    )

    with pytest.raises(AnthropicError) as exc_info:
        await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-test",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-ant-api-test",
        )

    assert exc_info.value.status_code == expected_status
    assert "private-upstream-data" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,status_code",
    [
        (httpx.ReadTimeout("private-transport-data"), 504),
        (httpx.ConnectError("private-transport-data"), 502),
        (httpx.LocalProtocolError("private-transport-data"), 502),
    ],
)
async def test_transport_failures_preserve_status_without_exposing_details(monkeypatch, failure, status_code):
    from litellm.llms.anthropic.common_utils import AnthropicError

    monkeypatch.setattr(
        "litellm.llms.anthropic.count_tokens.handler.get_async_httpx_client",
        lambda **kwargs: StubHTTPClient(failure),
    )
    with pytest.raises(AnthropicError) as error:
        await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-test", messages=[{"role": "user", "content": "Hello"}], api_key="sk-ant-api-test"
        )
    assert error.value.status_code == status_code
    assert "private-transport-data" not in str(error.value)
    assert error.value.__cause__ is None


class RecordingHandler(AnthropicCountTokensHandler):
    def __init__(self, input_tokens: int = 17) -> None:
        self.input_tokens = input_tokens
        self.requests: list[dict[str, object]] = []

    async def handle_count_tokens_request(self, **kwargs):
        self.requests.append(dict(kwargs))
        return {"input_tokens": self.input_tokens}


class ResolvingHook:
    def __init__(self, access_token: str = "sk-ant-oat-refreshed") -> None:
        self.access_token = access_token
        self.requests: list[dict[str, object]] = []

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object]:
        self.requests.append(dict(kwargs))
        return {**kwargs, "api_key": self.access_token, "api_base": "https://api.anthropic.com"}


class RejectingHook:
    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object]:
        raise AuthenticationError(
            message="managed credential rejected",
            model=str(kwargs.get("model", "anthropic")),
            llm_provider="anthropic",
        )


@pytest.mark.asyncio
async def test_managed_credential_is_resolved_before_counting() -> None:
    hook: Final = ResolvingHook()
    handler: Final = RecordingHandler()
    counter: Final = AnthropicTokenCounter(oauth_credential_hook=hook, count_tokens_handler=handler)

    result: Final = await counter.count_tokens(
        model_to_use="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={
            "litellm_params": {
                "litellm_credential_name": "subscription",
                "custom_llm_provider": "anthropic",
            }
        },
        request_model="managed-model",
    )

    assert result is not None
    assert result.total_tokens == handler.input_tokens
    assert hook.requests[0]["litellm_credential_name"] == "subscription"
    assert handler.requests[0]["api_key"] == hook.access_token


@pytest.mark.asyncio
async def test_managed_credential_resolution_failure_does_not_use_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key-must-not-be-used")
    handler: Final = RecordingHandler()
    counter: Final = AnthropicTokenCounter(oauth_credential_hook=RejectingHook(), count_tokens_handler=handler)

    result: Final = await counter.count_tokens(
        model_to_use="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={
            "litellm_params": {
                "litellm_credential_name": "subscription",
                "custom_llm_provider": "openai",
                "api_base": "https://example.invalid",
            }
        },
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 401
    assert handler.requests == []


@pytest.mark.asyncio
async def test_api_key_path_does_not_invoke_managed_credential_hook() -> None:
    handler: Final = RecordingHandler()
    hook: Final = ResolvingHook()
    counter: Final = AnthropicTokenCounter(oauth_credential_hook=hook, count_tokens_handler=handler)

    result: Final = await counter.count_tokens(
        model_to_use="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {"api_key": "sk-ant-api-test"}},
    )

    assert result is not None
    assert result.error is False
    assert hook.requests == []
    assert handler.requests[0]["api_key"] == "sk-ant-api-test"


@pytest.mark.asyncio
@pytest.mark.parametrize("repeated_rejection", [False, True])
async def test_managed_count_retries_rejected_token_only_once(monkeypatch, repeated_rejection):
    import litellm
    from litellm.llms.anthropic.common_utils import AnthropicError

    class RecoveringHook(ResolvingHook):
        async def recover_rejected_token(self, credential_name, rejected_access_token):
            assert credential_name == "subscription"
            assert rejected_access_token == self.access_token
            return "sk-ant-oat-rotated"

    class RejectedCount(RecordingHandler):
        async def handle_count_tokens_request(self, **kwargs):
            self.requests.append(dict(kwargs))
            if len(self.requests) == 1 or repeated_rejection:
                raise AnthropicError(status_code=401, message="Rejected token")
            return {"input_tokens": self.input_tokens}

    hook = RecoveringHook()
    handler = RejectedCount()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    result = await AnthropicTokenCounter(hook, handler).count_tokens(
        model_to_use="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={"litellm_params": {"litellm_credential_name": "subscription"}},
    )
    assert [request["api_key"] for request in handler.requests] == [hook.access_token, "sk-ant-oat-rotated"]
    assert result.error is repeated_rejection
    if repeated_rejection:
        assert result.status_code == 401
    else:
        assert result.total_tokens == handler.input_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503, 504])
async def test_counting_preserves_transient_refresh_status(status):
    from litellm.llms.anthropic.oauth_client import AnthropicOAuthError

    class UnavailableHook(ResolvingHook):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            raise AnthropicOAuthError("token refresh", status, "5")

    handler = RecordingHandler()
    result = await AnthropicTokenCounter(UnavailableHook(), handler).count_tokens(
        model_to_use="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hello"}],
        contents=None,
        deployment={"litellm_params": {"litellm_credential_name": "subscription"}},
    )
    assert result.error is True
    assert result.status_code == status
    assert result.retry_after == "5"
    assert "retry_after" not in result.model_dump()
    assert handler.requests == []


@pytest.mark.asyncio
async def test_provider_error_preserves_retry_after_for_proxy() -> None:
    from litellm.llms.anthropic.common_utils import AnthropicError

    class RateLimitedHandler(RecordingHandler):
        async def handle_count_tokens_request(self, **kwargs):
            raise AnthropicError(429, "rate limited", headers=httpx.Headers({"retry-after": "11"}))

    result = await AnthropicTokenCounter(count_tokens_handler=RateLimitedHandler()).count_tokens(
        model_to_use="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        contents=None,
        deployment={"litellm_params": {"api_key": "sk-ant-api-test"}},
    )

    assert result is not None
    assert result.status_code == 429
    assert result.retry_after == "11"
