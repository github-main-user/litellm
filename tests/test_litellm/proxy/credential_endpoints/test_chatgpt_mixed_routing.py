import json
import time
from collections.abc import Iterator
from copy import deepcopy
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.models.credentials import CredentialItem
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
    _DatabaseChatGPTCredential,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse

MODEL_GROUP = "shared-model"
CHATGPT_A = "chatgpt-subscription-a"
CHATGPT_B = "chatgpt-subscription-b"
OPENAI_API = "openai-api"


def _credential(name: str, account_id: str, access_token: str, *, expired: bool = False) -> CredentialItem:
    tokens = ChatGPTTokens(
        access_token=access_token,
        refresh_token=f"refresh-{account_id}",
        id_token=f"id-{account_id}",
        expires_at=int(time.time()) - 3600 if expired else int(time.time()) + 3600,
        account_id=account_id,
    )
    return _DatabaseChatGPTCredential(
        credential_name=name,
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def _deployment_by_id(deployments: list[dict], deployment_id: str) -> dict:
    return next(deployment for deployment in deployments if deployment["model_info"]["id"] == deployment_id)


def _response(sequence: int) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id=f"resp-mixed-{sequence}",
        created_at=1741476542,
        status="completed",
        model="gpt-5.4",
        output=[
            {
                "type": "reasoning",
                "id": f"rs-mixed-{sequence}",
                "status": "completed",
                "encrypted_content": f"encrypted-{sequence}",
            }
        ],
        usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    )


def _attempt(call: dict) -> tuple[str, str | None, str | None]:
    params = call["litellm_params"]
    return (
        call["custom_llm_provider"],
        params.api_key,
        params.get("chatgpt_auth_account_id"),
    )


@pytest.fixture
def mixed_router() -> Iterator[litellm.Router]:
    previous_credentials = litellm.credential_list
    previous_callbacks = list(litellm.callbacks)
    litellm.credential_list = [
        _credential("subscription-a", "account-a", "access-a"),
        _credential("subscription-b", "account-b", "access-b"),
    ]
    litellm.callbacks = [ChatGPTOAuthCredentialHook()]
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "subscription-a",
                },
                "model_info": {"id": CHATGPT_A},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "chatgpt/gpt-5.4",
                    "litellm_credential_name": "subscription-b",
                },
                "model_info": {"id": CHATGPT_B},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-5.4",
                    "api_key": "openai-key",
                },
                "model_info": {"id": OPENAI_API},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
        model_group_affinity_config={
            MODEL_GROUP: [
                "responses_api_deployment_check",
                "encrypted_content_affinity",
            ]
        },
    )

    try:
        yield router
    finally:
        router.discard()
        litellm.credential_list = previous_credentials
        litellm.callbacks = previous_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["aresponses", "acompletion"])
async def test_expired_managed_credentials_fail_over_to_third_account(
    api: str,
) -> None:
    previous_credentials = litellm.credential_list
    previous_callbacks = list(litellm.callbacks)
    names = ("subscription-a", "subscription-b", "subscription-c")
    ids = ("account-a-deployment", "account-b-deployment", "account-c-deployment")
    litellm.credential_list = [
        _credential(name, f"account-{letter}", f"access-{letter}", expired=letter != "c")
        for name, letter in zip(names, "abc")
    ]
    hook = ChatGPTOAuthCredentialHook()
    litellm.callbacks = [hook]
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "chatgpt/gpt-5.4", "litellm_credential_name": name},
                "model_info": {"id": deployment_id},
            }
            for name, deployment_id in zip(names, ids)
        ],
        routing_strategy="simple-shuffle",
        num_retries=2,
        max_fallbacks=5,
        enable_weighted_failover=True,
    )
    selections: list[str] = []
    requests: list[httpx.Request] = []

    def select(deployments: list[dict]) -> dict:
        selected = next(item for item in deployments if item["model_info"]["id"] not in selections)
        selections.append(selected["model_info"]["id"])
        return selected

    def upstream(request: httpx.Request, **_kwargs: object) -> httpx.Response:
        assert request.url.host == "chatgpt.com"
        requests.append(request)
        response = _response(3).model_dump()
        response["object"] = "response"
        response["output"] = [
            {
                "type": "message",
                "id": "msg-third",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ]
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=("data: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode(),
        )

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch("litellm.router_strategy.simple_shuffle.random.choice", side_effect=select),
            patch.object(httpx.AsyncClient, "send", new=AsyncMock(side_effect=upstream)),
        ):
            for name in names[:2]:
                with pytest.raises(
                    litellm.AuthenticationError, match=f"ChatGPT credential '{name}' is unavailable"
                ) as error:
                    await hook.async_pre_call_deployment_hook(
                        {"model": "chatgpt/gpt-5.4", "litellm_credential_name": name}, None
                    )
                assert isinstance(error.value.__cause__, RuntimeError)
            if api == "aresponses":
                result = await router.aresponses(model=MODEL_GROUP, input="hello")
            else:
                result = await router.acompletion(model=MODEL_GROUP, messages=[{"role": "user", "content": "hello"}])
        assert result._hidden_params["model_id"] == ids[2]
        assert selections == list(ids)
        assert len(requests) == 1
        assert requests[0].headers["authorization"] == "Bearer access-c"
        assert requests[0].headers["chatgpt-account-id"] == "account-c"
    finally:
        router.discard()
        litellm.credential_list = previous_credentials
        litellm.callbacks = previous_callbacks


@pytest.mark.asyncio
async def test_initial_requests_balance_across_subscriptions_and_api_key(
    mixed_router: litellm.Router,
) -> None:
    targets = iter([CHATGPT_A, CHATGPT_B, OPENAI_API])
    attempts: list[tuple[str, str | None, str | None]] = []

    def select(deployments: list[dict]) -> dict:
        return _deployment_by_id(deployments, next(targets))

    async def respond(**call) -> ResponsesAPIResponse:
        attempts.append(_attempt(call))
        return _response(len(attempts))

    with (
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=select,
        ),
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new=AsyncMock(side_effect=respond),
        ),
    ):
        for _ in range(3):
            await mixed_router.aresponses(model=MODEL_GROUP, input="hello")

    assert attempts == [
        ("chatgpt", "access-a", "account-a"),
        ("chatgpt", "access-b", "account-b"),
        ("openai", "openai-key", None),
    ]


@pytest.mark.asyncio
async def test_initial_subscription_failure_fails_over_to_api_key(
    mixed_router: litellm.Router,
) -> None:
    attempts: list[tuple[str, str | None, str | None]] = []
    first_selection = True

    def select(deployments: list[dict]) -> dict:
        nonlocal first_selection
        deployment_id = CHATGPT_A if first_selection else OPENAI_API
        first_selection = False
        return _deployment_by_id(deployments, deployment_id)

    async def respond(**call) -> ResponsesAPIResponse:
        attempt = _attempt(call)
        attempts.append(attempt)
        if attempt[1] == "access-a":
            raise litellm.RateLimitError(
                message="subscription throttled",
                llm_provider="chatgpt",
                model="gpt-5.4",
                response=httpx.Response(
                    status_code=429,
                    request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
                ),
            )
        return _response(len(attempts))

    with (
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=select,
        ),
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new=AsyncMock(side_effect=respond),
        ),
    ):
        response = await mixed_router.aresponses(model=MODEL_GROUP, input="hello")

    assert response._hidden_params["model_id"] == OPENAI_API
    assert attempts == [
        ("chatgpt", "access-a", "account-a"),
        ("openai", "openai-key", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin_state", ["healthy", "cooled", "quota"])
@pytest.mark.parametrize("alternate", [CHATGPT_B, OPENAI_API])
async def test_encrypted_follow_up_preserves_history_when_origin_fails_over(
    mixed_router: litellm.Router,
    origin_state: str,
    alternate: str,
) -> None:
    encoded_id: Final = ResponsesAPIRequestUtils._build_encrypted_item_id(
        CHATGPT_A,
        "rs-origin",
    )
    readable_history: Final = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "I will check the forecast."},
        {
            "type": "function_call",
            "call_id": "call-weather",
            "name": "weather",
            "arguments": '{"city":"Paris"}',
        },
        {"type": "function_call_output", "call_id": "call-weather", "output": "18 C, sunny"},
        {"role": "user", "content": "Should I bring an umbrella?"},
    ]
    request_input: Final = [
        *readable_history[:1],
        {
            "type": "reasoning",
            "id": encoded_id,
            "encrypted_content": "encrypted-origin",
        },
        *readable_history[1:],
    ]
    attempts: Final = []
    dispatched_inputs: Final = []

    def select(deployments: list[dict]) -> dict:
        assert origin_state != "healthy", "healthy affinity must bypass normal selection"
        return _deployment_by_id(deployments, alternate)

    async def respond(**call) -> ResponsesAPIResponse:
        attempt: Final = _attempt(call)
        attempts.append(attempt)
        dispatched_inputs.append(deepcopy(call["input"]))
        if origin_state == "quota" and attempt[1] == "access-a":
            raise litellm.RateLimitError(
                message="subscription quota exhausted",
                llm_provider="chatgpt",
                model="gpt-5.4",
                response=httpx.Response(
                    status_code=429,
                    request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
                ),
            )
        return _response(len(attempts))

    if origin_state == "cooled":
        mixed_router.cooldown_cache.add_deployment_to_cooldown(
            model_id=CHATGPT_A,
            original_exception=Exception("subscription quota exhausted"),
            exception_status=429,
            cooldown_time=60,
        )

    with (
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=select,
        ),
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new=AsyncMock(side_effect=respond),
        ),
    ):
        response: Final = await mixed_router.aresponses(
            model=MODEL_GROUP,
            input=deepcopy(request_input),
        )

    origin_attempt: Final = ("chatgpt", "access-a", "account-a")
    alternate_attempt: Final = (
        ("chatgpt", "access-b", "account-b") if alternate == CHATGPT_B else ("openai", "openai-key", None)
    )
    expected_attempts: Final = {
        "healthy": [origin_attempt],
        "cooled": [alternate_attempt],
        "quota": [origin_attempt, alternate_attempt],
    }
    origin_input: Final = [
        *readable_history[:1],
        {"type": "reasoning", "id": "rs-origin", "encrypted_content": "encrypted-origin"},
        *readable_history[1:],
    ]
    expected_inputs: Final = {
        "healthy": [origin_input],
        "cooled": [readable_history],
        "quota": [origin_input, readable_history],
    }
    assert response._hidden_params["model_id"] == (CHATGPT_A if origin_state == "healthy" else alternate)
    assert attempts == expected_attempts[origin_state]
    assert dispatched_inputs == expected_inputs[origin_state]


@pytest.mark.asyncio
async def test_streaming_encrypted_follow_up_retries_exhausted_origin_with_readable_history(
    mixed_router: litellm.Router,
) -> None:
    readable_history: Final = [
        {"role": "user", "content": "Check the weather"},
        {"type": "function_call", "call_id": "weather-1", "name": "weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "weather-1", "output": "sunny"},
        {"role": "user", "content": "Do I need an umbrella?"},
    ]
    request_input: Final = [
        *readable_history[:1],
        {
            "type": "reasoning",
            "id": ResponsesAPIRequestUtils._build_encrypted_item_id(CHATGPT_A, "rs-stream-origin"),
            "encrypted_content": "encrypted-origin",
        },
        *readable_history[1:],
    ]
    requests: Final = []

    def select(deployments: list[dict]) -> dict:
        return _deployment_by_id(deployments, CHATGPT_B)

    def upstream(request: httpx.Request, **_kwargs: object) -> httpx.Response:
        requests.append(request)
        if request.headers["authorization"] == "Bearer access-a":
            return httpx.Response(
                429,
                request=request,
                json={"error": {"type": "usage_limit_reached", "message": "subscription quota exhausted"}},
            )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=(
                "data: " + json.dumps({"type": "response.completed", "response": _response(2).model_dump()}) + "\n\n"
            ).encode(),
        )

    injected_client: Final = AsyncHTTPHandler()
    await injected_client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as raw_client:
        injected_client.client = raw_client
        with (
            patch("litellm.router_strategy.simple_shuffle.random.choice", side_effect=select),
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
                return_value=injected_client,
            ),
        ):
            stream: Final = await mixed_router.aresponses(
                model=MODEL_GROUP, input=request_input, stream=True, client=injected_client
            )
            events: Final = [event async for event in stream]

    assert [request.headers["authorization"] for request in requests] == ["Bearer access-a", "Bearer access-b"]
    assert [request.headers["chatgpt-account-id"] for request in requests] == ["account-a", "account-b"]
    assert [json.loads(request.content)["stream"] for request in requests] == [True, True]
    assert json.loads(requests[1].content)["input"] == readable_history
    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert events[0].response.status == "completed"
