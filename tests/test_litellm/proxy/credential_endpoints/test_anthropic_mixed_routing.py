import json
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.anthropic.common_utils import ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT
from litellm.llms.anthropic.oauth_client import AnthropicOAuthTokens
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.models.credentials import CredentialItem
from litellm.proxy.credential_endpoints.anthropic_oauth import (
    ANTHROPIC_CREDENTIAL_VALUE_KEY,
    AnthropicOAuthCredentialHook,
)
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

MODEL_GROUP = "mixed-chat"
ANTHROPIC_A = "anthropic-subscription-a"
ANTHROPIC_B = "anthropic-subscription-b"
ANTHROPIC_API = "anthropic-api-key"
CHATGPT_SUBSCRIPTION = "chatgpt-subscription"
OPENAI_API = "openai-api-key"


def _anthropic_oauth_credential(name: str, token: str, account_id: str) -> CredentialItem:
    tokens = AnthropicOAuthTokens(
        access_token=token,
        refresh_token=f"refresh-{account_id}",
        expires_at=time.time() + 3600,
        account_id=account_id,
    )
    return CredentialItem(
        credential_name=name,
        credential_info={"provider": "anthropic", "auth_type": "oauth"},
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def _api_key_credential(name: str, provider: str, api_key: str) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_info={"provider": provider, "auth_type": "api_key"},
        credential_values={"api_key": api_key},
    )


def _chatgpt_credential() -> CredentialItem:
    tokens = ChatGPTTokens(
        access_token="chatgpt-access",
        refresh_token="chatgpt-refresh",
        id_token="chatgpt-id",
        expires_at=int(time.time()) + 3600,
        account_id="chatgpt-account",
    )
    return CredentialItem(
        credential_name="chatgpt-subscription",
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def _deployment_by_id(deployments: list[dict], deployment_id: str) -> dict:
    return next(item for item in deployments if item["model_info"]["id"] == deployment_id)


def _chat_response(sequence: int = 1) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-mixed-{sequence}",
        model="test-model",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _responses_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp-mixed-chatgpt",
        created_at=1741476542,
        status="completed",
        model="gpt-5.4",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def _anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_mixed",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _openai_response() -> dict[str, Any]:
    return _chat_response().model_dump()


def _captured_request(request: httpx.Request) -> dict[str, Any]:
    return {
        "url": str(request.url),
        "headers": dict(request.headers),
        "data": json.loads(request.content),
    }


def _header(headers: dict[str, Any], name: str) -> object | None:
    return next((value for key, value in headers.items() if key.lower() == name.lower()), None)


def _system_texts(data: dict[str, Any]) -> list[str]:
    system = data.get("system", [])
    if isinstance(system, str):
        return [system]
    if not isinstance(system, list):
        return []
    return [
        block["text"]
        for block in system
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]


class _UpstreamMock:
    def __init__(self) -> None:
        self.handler: Any = None
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        assert self.handler is not None, f"Unexpected upstream request: {request.method} {request.url}"
        return self.handler(request)


@pytest.fixture
def mock_upstream(monkeypatch: pytest.MonkeyPatch) -> Iterator[_UpstreamMock]:
    from litellm.anthropic_beta_headers_manager import reload_beta_headers_config

    upstream = _UpstreamMock()
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    reload_beta_headers_config()
    previous_cache = litellm.in_memory_llm_clients_cache
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_: upstream.transport),
    )
    try:
        yield upstream
    finally:
        litellm.in_memory_llm_clients_cache = previous_cache


@pytest.fixture
def mixed_router(mock_upstream: _UpstreamMock) -> Iterator[litellm.Router]:
    previous_credentials = litellm.credential_list
    previous_callbacks = list(litellm.callbacks)
    litellm.credential_list = [
        _anthropic_oauth_credential("anthropic-a", "sk-ant-oat-a", "anthropic-account-a"),
        _anthropic_oauth_credential("anthropic-b", "sk-ant-oat-b", "anthropic-account-b"),
        _api_key_credential("anthropic-key", "anthropic", "sk-ant-api03-named"),
        _chatgpt_credential(),
        _api_key_credential("openai-key", "openai", "sk-openai-named"),
    ]
    litellm.callbacks = [AnthropicOAuthCredentialHook(), ChatGPTOAuthCredentialHook()]
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "litellm_credential_name": "anthropic-a",
                },
                "model_info": {"id": ANTHROPIC_A},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "litellm_credential_name": "anthropic-b",
                },
                "model_info": {"id": ANTHROPIC_B},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "litellm_credential_name": "anthropic-key",
                },
                "model_info": {"id": ANTHROPIC_API},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "chatgpt-subscription",
                },
                "model_info": {"id": CHATGPT_SUBSCRIPTION},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-5.4",
                    "litellm_credential_name": "openai-key",
                },
                "model_info": {"id": OPENAI_API},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )
    try:
        yield router
    finally:
        router.discard()
        litellm.credential_list = previous_credentials
        litellm.callbacks = previous_callbacks


@pytest.mark.asyncio
async def test_chat_routes_apply_only_the_selected_named_credential(
    mixed_router: litellm.Router,
    mock_upstream: _UpstreamMock,
) -> None:
    targets = iter([ANTHROPIC_A, ANTHROPIC_B, ANTHROPIC_API, CHATGPT_SUBSCRIPTION, OPENAI_API])
    calls: list[dict[str, Any]] = []

    def select(deployments: list[dict]) -> dict:
        return _deployment_by_id(deployments, next(targets))

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(_captured_request(request))
        if request.url.host == "api.anthropic.com":
            body = _anthropic_response()
        elif request.url.path.endswith("/responses"):
            response = _responses_response().model_dump()
            response["object"] = "response"
            event = {"type": "response.completed", "response": response}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(event)}\n\n".encode(),
            )
        else:
            body = _openai_response()
        return httpx.Response(200, json=body)

    mock_upstream.handler = upstream
    messages = [
        {"role": "system", "content": "caller system"},
        {"role": "user", "content": "hello"},
    ]
    with patch("litellm.router_strategy.simple_shuffle.random.choice", side_effect=select):
        for _ in range(5):
            await mixed_router.acompletion(model=MODEL_GROUP, messages=messages, disable_fallbacks=True)

    anthropic_calls = [call for call in calls if "anthropic.com" in call["url"]]
    assert len(anthropic_calls) == 3
    for call, token in zip(anthropic_calls[:2], ("sk-ant-oat-a", "sk-ant-oat-b")):
        assert _header(call["headers"], "authorization") == f"Bearer {token}"
        assert _header(call["headers"], "x-api-key") is None
        assert _system_texts(call["data"]) == [ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT, "caller system"]

    api_key_call = anthropic_calls[2]
    assert _header(api_key_call["headers"], "authorization") is None
    assert _header(api_key_call["headers"], "x-api-key") == "sk-ant-api03-named"
    assert _system_texts(api_key_call["data"]) == ["caller system"]

    chatgpt_call = next(call for call in calls if call["url"].endswith("/responses"))
    assert _header(chatgpt_call["headers"], "authorization") == "Bearer chatgpt-access"
    assert _header(chatgpt_call["headers"], "chatgpt-account-id") == "chatgpt-account"

    openai_call = next(call for call in calls if call["url"].endswith("/chat/completions"))
    assert _header(openai_call["headers"], "authorization") == "Bearer sk-openai-named"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order",
    [(ANTHROPIC_A, ANTHROPIC_B, ANTHROPIC_API), (ANTHROPIC_API, ANTHROPIC_A, ANTHROPIC_B)],
)
async def test_anthropic_failover_rebuilds_oauth_and_api_key_requests(
    mixed_router: litellm.Router,
    order: tuple[str, str, str],
    mock_upstream: _UpstreamMock,
) -> None:
    targets = iter(order)
    attempts: list[dict[str, Any]] = []

    def select(deployments: list[dict]) -> dict:
        return _deployment_by_id(deployments, next(targets))

    def upstream(request: httpx.Request) -> httpx.Response:
        attempts.append(_captured_request(request))
        if len(attempts) < len(order):
            return httpx.Response(429, json={"type": "error", "error": {"message": "subscription throttled"}})
        return httpx.Response(200, json=_anthropic_response())

    mock_upstream.handler = upstream
    with patch("litellm.router_strategy.simple_shuffle.random.choice", side_effect=select):
        response = await mixed_router.acompletion(
            model=MODEL_GROUP,
            messages=[
                {"role": "system", "content": "do not leak this request"},
                {"role": "user", "content": "hello"},
            ],
        )

    tokens = {
        ANTHROPIC_A: "sk-ant-oat-a",
        ANTHROPIC_B: "sk-ant-oat-b",
        ANTHROPIC_API: "sk-ant-api03-named",
    }
    assert response._hidden_params["model_id"] == order[-1]
    assert [_header(call["headers"], "authorization") for call in attempts] == [
        None if deployment == ANTHROPIC_API else f"Bearer {tokens[deployment]}" for deployment in order
    ]
    assert [_header(call["headers"], "x-api-key") for call in attempts] == [
        tokens[deployment] if deployment == ANTHROPIC_API else None for deployment in order
    ]
    assert [_system_texts(call["data"]) for call in attempts] == [
        ["do not leak this request"]
        if deployment == ANTHROPIC_API
        else [ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT, "do not leak this request"]
        for deployment in order
    ]


@pytest.mark.asyncio
async def test_streaming_route_uses_selected_anthropic_account_without_stale_state(
    mixed_router: litellm.Router,
    mock_upstream: _UpstreamMock,
) -> None:
    captured: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(_captured_request(request))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")

    def select(deployments: list[dict]) -> dict:
        return _deployment_by_id(deployments, ANTHROPIC_B)

    mock_upstream.handler = upstream
    with patch("litellm.router_strategy.simple_shuffle.random.choice", side_effect=select):
        stream = await mixed_router.acompletion(
            model=MODEL_GROUP,
            messages=[
                {"role": "system", "content": "stream system"},
                {"role": "user", "content": "hello"},
            ],
            stream=True,
        )
        await stream.aclose()

    assert len(captured) == 1
    call = captured[0]
    assert _header(call["headers"], "authorization") == "Bearer sk-ant-oat-b"
    assert _header(call["headers"], "x-api-key") is None
    assert call["data"]["stream"] is True
    assert _system_texts(call["data"]) == [ANTHROPIC_SUBSCRIPTION_SYSTEM_PROMPT, "stream system"]
