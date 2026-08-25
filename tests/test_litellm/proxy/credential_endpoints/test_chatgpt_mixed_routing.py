import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens
from litellm.models.credentials import CredentialItem
from litellm.proxy.credential_endpoints.chatgpt_oauth import (
    CHATGPT_CREDENTIAL_VALUE_KEY,
    ChatGPTOAuthCredentialHook,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse

MODEL_GROUP = "shared-model"
CHATGPT_A = "chatgpt-subscription-a"
CHATGPT_B = "chatgpt-subscription-b"
OPENAI_API = "openai-api"


def _credential(name: str, account_id: str, access_token: str) -> CredentialItem:
    tokens = ChatGPTTokens(
        access_token=access_token,
        refresh_token=f"refresh-{account_id}",
        id_token=f"id-{account_id}",
        expires_at=int(time.time()) + 3600,
        account_id=account_id,
    )
    return CredentialItem(
        credential_name=name,
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def _deployment_by_id(deployments: list[dict], deployment_id: str) -> dict:
    return next(
        deployment
        for deployment in deployments
        if deployment["model_info"]["id"] == deployment_id
    )


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
async def test_encrypted_follow_up_pins_or_fails_without_crossing_accounts(
    mixed_router: litellm.Router,
) -> None:
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(
        CHATGPT_A,
        "rs-origin",
    )
    request_input = [
        {
            "type": "reasoning",
            "id": encoded_id,
            "encrypted_content": "encrypted-origin",
        }
    ]

    with patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=AssertionError("affinity must bypass normal selection"),
    ):
        selected = await mixed_router.async_get_available_deployment(
            model=MODEL_GROUP,
            request_kwargs={"input": request_input, "litellm_metadata": {}},
            input=request_input,
        )

    assert selected["model_info"]["id"] == CHATGPT_A

    mixed_router.cooldown_cache.add_deployment_to_cooldown(
        model_id=CHATGPT_A,
        original_exception=Exception("subscription throttled"),
        exception_status=429,
        cooldown_time=60,
    )

    with (
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=AssertionError("unavailable affinity must fail closed"),
        ),
        pytest.raises(litellm.RateLimitError) as exc_info,
    ):
        await mixed_router.async_get_available_deployment(
            model=MODEL_GROUP,
            request_kwargs={"input": request_input, "litellm_metadata": {}},
            input=request_input,
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.response.headers["retry-after"]
