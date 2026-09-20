from typing import Final

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.types.utils import CallTypes


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
        model_to_use="claude-sonnet-5", messages=[{"role": "user", "content": "Hello"}], contents=None,
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
        model_to_use="claude-sonnet-5", messages=[{"role": "user", "content": "Hello"}], contents=None,
        deployment={"litellm_params": {"litellm_credential_name": "subscription"}},
    )
    assert result.error is True
    assert result.status_code == status
    assert handler.requests == []
