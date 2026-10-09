"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Final, Literal
from unittest.mock import MagicMock
from uuid import UUID

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens, ManagedChatGPTAccessToken
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.main import responses_api_bridge_check
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.fixture
def managed_credential(monkeypatch: pytest.MonkeyPatch) -> str:
    tokens: Final = ChatGPTTokens("managed-token", "refresh", "id", 9999999999, "managed-account")
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="subscription-a",
        credential_info={"provider": "chatgpt", "auth_type": "oauth"},
        credential_values={"litellm_internal_chatgpt_auth_token": tokens.to_json()},
    )])
    return "subscription-a"


@pytest.mark.parametrize("access_token", [None, "unmanaged-token"])
def test_responses_requires_managed_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, access_token: str | None
) -> None:
    token_dir: Final = tmp_path / ".config" / "litellm" / "chatgpt"
    monkeypatch.setenv("CHATGPT_ALLOW_FILE_AUTH", "true")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(litellm, "credential_list", [])
    config: Final = ChatGPTResponsesAPIConfig()

    with pytest.raises(AuthenticationError, match="managed OAuth credential"):
        config.validate_environment(
            headers={}, model="test-model", litellm_params=GenericLiteLLMParams(api_key=access_token)
        )

    assert not token_dir.exists()


@pytest.mark.parametrize("text", ["Hello", "", "Привет\n世界"])
def test_chatgpt_wraps_string_input_without_changing_openai(text: str) -> None:
    params: Final = {
        "model": "gpt-6-luna",
        "input": text,
        "response_api_optional_request_params": {"instructions": "Keep it brief."},
        "litellm_params": GenericLiteLLMParams(),
        "headers": {},
    }
    chatgpt_request: Final = ChatGPTResponsesAPIConfig().transform_responses_api_request(**params)
    openai_request: Final = OpenAIResponsesAPIConfig().transform_responses_api_request(**params)

    assert chatgpt_request["input"] == [{"role": "user", "content": text}]
    assert chatgpt_request["instructions"] == "Keep it brief."
    assert openai_request["input"] == text


def test_chatgpt_preserves_input_items() -> None:
    items: Final = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        {"type": "function_call_output", "call_id": "call_test", "output": "Done"},
    ]
    request: Final = ChatGPTResponsesAPIConfig().transform_responses_api_request(
        model="gpt-6-luna",
        input=items,
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        {"type": "function_call_output", "call_id": "call_test", "output": "Done"},
    ]


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("cache_key", [None, "stable-prompt-key"], ids=["omitted-key", "explicit-key"])
async def test_chatgpt_prompt_cache_key_reaches_wire(
    api: Literal["responses", "chat"],
    asynchronous: bool,
    cache_key: str | None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    local_model_cost_map: None,
    managed_credential: str,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    model: Final = "gpt-5.6-luna"
    payload: Final = {
        "id": "resp_cache",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_cache",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            }
        ],
    }
    upstream: Final = respx_mock.post("https://chatgpt.test/backend-api/codex/responses").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps({'type': 'response.completed', 'response': payload})}\n\ndata: [DONE]\n\n",
        )
    )
    params: Final = {
        "model": f"chatgpt/{model}",
        "api_key": ManagedChatGPTAccessToken("managed-token"),
        "api_base": "https://chatgpt.test/backend-api/codex",
        "chatgpt_auth_account_id": "managed-account",
        "litellm_credential_name": managed_credential,
        "num_retries": 0,
        **({"prompt_cache_key": cache_key} if cache_key is not None else {}),
    }
    for attempt in range(2):
        request_params: Final = {**params, "litellm_trace_id": f"trace-{attempt}"}
        if api == "responses":
            response = (
                await litellm.aresponses(input="Hello", instructions="Keep it brief.", stream=False, **request_params)
                if asynchronous
                else litellm.responses(input="Hello", instructions="Keep it brief.", stream=False, **request_params)
            )
            assert response.object == "response"
            assert response.status == "completed"
            assert response.output[0].content[0].text == "Hello!"
        else:
            messages: Final = [
                {"role": "system", "content": "Keep it brief."},
                {"role": "user", "content": "Hello"},
            ]
            completion: Final = (
                await litellm.acompletion(messages=messages, **request_params)
                if asynchronous
                else litellm.completion(messages=messages, **request_params)
            )
            assert completion.choices[0].message.content == "Hello!"

    assert upstream.call_count == 2
    session_ids: Final = [call.request.headers["session_id"] for call in upstream.calls]
    assert (session_ids[0] == session_ids[1]) is (cache_key is not None)
    request: Final = upstream.calls[0].request
    assert request.url == httpx.URL("https://chatgpt.test/backend-api/codex/responses")
    assert request.headers["authorization"] == "Bearer managed-token"
    assert request.headers["chatgpt-account-id"] == "managed-account"
    assert json.loads(request.content) == {
        "model": model,
        "input": (
            [{"role": "user", "content": "Hello"}]
            if api == "responses"
            else [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        ),
        "instructions": "Keep it brief.",
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        **({"prompt_cache_key": cache_key} if cache_key is not None else {}),
    }


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.5",
            "chatgpt/gpt-5.6-luna",
            "chatgpt/gpt-5.6-sol",
            "chatgpt/gpt-5.6-terra",
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT


    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
    )
    def test_chatgpt_models_bridge_chat_completions_to_responses(
        self, model_name: str, local_model_cost_map: None
    ) -> None:
        """A chat completions request for these models must take the Responses bridge.

        `gpt-5.6-*` also exists as an openai chat model, so an unregistered
        chatgpt model resolves to mode "chat" here and never reaches the bridge.
        """
        model_info, resolved_model = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider="chatgpt",
        )

        assert model_info["mode"] == "responses"
        assert resolved_model == model_name

    def test_chatgpt_responses_endpoint_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHATGPT_API_BASE", "https://chatgpt.example.com")

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"

    def test_validate_environment_protects_canonical_headers_on_retry(
        self, managed_credential: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHATGPT_ORIGINATOR", "managed-origin")
        monkeypatch.setenv("CHATGPT_USER_AGENT", "managed-client/1")
        config: Final = ChatGPTResponsesAPIConfig()
        metadata: Final = {"session_id": "session-123", "custom": "preserved"}
        litellm_params: Final = GenericLiteLLMParams(
            metadata=metadata, api_key=ManagedChatGPTAccessToken("managed-token"),
            litellm_credential_name=managed_credential, chatgpt_auth_account_id="spoofed-account"
        )
        incoming: Final = {
            **{name: "spoofed" for name in (
                "Authorization", "AUTHORIZATION", "chatgpt-account-id", "ChatGPT-Account-Id",
                "originator", "ORIGINATOR", "user-agent", "User-Agent", "session_id", "SESSION_ID"
            )},
            "X-Custom": "preserved",
        }
        headers: Final = config.validate_environment(incoming, "test-model", litellm_params)
        assert headers == {
            "Authorization": "Bearer managed-token",
            "ChatGPT-Account-Id": "managed-account",
            "originator": "managed-origin",
            "User-Agent": "managed-client/1",
            "session_id": headers["session_id"],
            "content-type": "application/json",
            "accept": "text/event-stream",
            "X-Custom": "preserved",
        }
        assert str(UUID(headers["session_id"])) == headers["session_id"]
        assert headers["session_id"] != "session-123"
        assert config.validate_environment(headers, "test-model", litellm_params) == headers
        assert litellm_params.metadata == metadata
        assert incoming["SESSION_ID"] == "spoofed"

    def test_validate_environment_uses_deployment_credentials(self, managed_credential: str):
        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(
            api_key=ManagedChatGPTAccessToken("managed-token"),
            chatgpt_auth_account_id="managed-account",
            litellm_credential_name=managed_credential,
        )

        headers = config.validate_environment(
            headers={"authorization": "Bearer unmanaged-token", "chatgpt-account-id": "wrong-account"},
            model="gpt-5.4",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer managed-token"
        assert headers["ChatGPT-Account-Id"] == "managed-account"
        assert "authorization" not in headers
        assert "chatgpt-account-id" not in headers

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CHATGPT_DEFAULT_INSTRUCTIONS", raising=False)
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"] == "You are a helpful assistant."

    def test_chatgpt_preserves_client_and_bridged_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler,
        )

        monkeypatch.setenv("CHATGPT_DEFAULT_INSTRUCTIONS", "operator fallback")
        input_items, bridged_instructions = (
            LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(
                [
                    {"role": "system", "content": "Follow the user's format"},
                    {"role": "developer", "content": "Answer briefly"},
                    {"role": "user", "content": "Hello"},
                ]
            )
        )
        config: Final = ChatGPTResponsesAPIConfig()
        request: Final = config.transform_responses_api_request(
            model="gpt-5.3-codex",
            input=input_items,
            response_api_optional_request_params={"instructions": bridged_instructions},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["instructions"] == "Follow the user's format"
        assert any(isinstance(item, dict) and item.get("role") == "developer" for item in request["input"])

        direct_request: Final = config.transform_responses_api_request(
            model="gpt-5.3-codex",
            input="Hello",
            response_api_optional_request_params={"instructions": "Use my own instructions"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert direct_request["instructions"] == "Use my own instructions"

        fallback_request: Final = config.transform_responses_api_request(
            model="gpt-5.3-codex",
            input="Hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert fallback_request["instructions"] == "operator fallback"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 200000}
                ],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                "prompt_cache_retention": "24h",
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request
        assert "prompt_cache_retention" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_recovers_output_items(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from stream!"}],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello from stream!"

    def test_chatgpt_non_stream_sse_recovers_whitespace_padded_chunks(self):
        """Chunks with leading whitespace before `data:` must still parse.

        `_strip_sse_data_from_chunk` only matches the prefix at position 0,
        so without an outer `.strip()` such chunks would fail JSON parsing
        and silently drop the contained event.
        """
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Recovered from padded"}],
        }
        sse_body = "\n".join(
            [
                f"   data:  {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}   ",
                f"\tdata: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Recovered from padded"

    @pytest.mark.parametrize(
        "error_chunk",
        [
            {
                "type": "response.failed",
                "response": {"error": {"message": "ChatGPT upstream failed"}},
            },
            {
                "type": "error",
                "error": {"message": "ChatGPT upstream failed"},
            },
        ],
    )
    def test_chatgpt_non_stream_sse_response_raises_openai_error(self, error_chunk):
        config = ChatGPTResponsesAPIConfig()
        sse_body = "\n".join(
            [
                f"data: {json.dumps(error_chunk)}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            502, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert "ChatGPT upstream failed" in str(exc_info.value)
        assert exc_info.value.status_code == 502


@pytest.mark.parametrize("terminal", [None, "response.failed", "response.incomplete", "response.completed"])
def test_chatgpt_delta_only_recovery_requires_completed(terminal):
    events = [
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": text}
        for text in ("Hel", "lo")
    ]
    if terminal:
        events.append(
            {
                "type": terminal,
                "response": {
                    "id": "resp_delta",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.4",
                    "status": terminal.split(".")[1],
                    "output": [],
                },
            }
        )
    body = "\n".join(f"data: {json.dumps(event)}" for event in events)
    response, error = ChatGPTResponsesAPIConfig()._extract_completed_response_from_sse(body)
    if terminal == "response.completed":
        assert response is not None
        assert response.output_text == "Hello"
        assert len(response.output) == 1
    else:
        assert response is None
    assert error is None


@pytest.mark.parametrize("source", ["terminal", "item_done", "text_done"])
def test_chatgpt_completed_recovery_prefers_full_text_over_deltas(source):
    item = {
        "type": "message",
        "id": "msg_full",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Full text", "annotations": []}],
    }
    events = [{"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Partial"}]
    if source == "item_done":
        events.append({"type": "response.output_item.done", "output_index": 0, "item": item})
    if source == "text_done":
        events.append({"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": "Full text"})
    events.append(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_full",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.4",
                "status": "completed",
                "output": [item] if source == "terminal" else [],
            },
        }
    )
    response, error = ChatGPTResponsesAPIConfig()._extract_completed_response_from_sse(
        "\n".join(f"data: {json.dumps(event)}" for event in events)
    )
    assert response is not None
    assert response.output_text == "Full text"
    assert len(response.output) == 1
    assert error is None


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("spoof_headers", [False, True], ids=["canonical", "spoofed"])
async def test_public_managed_sessions_isolate_owners_and_keys_with_protected_headers(
    api: str,
    asynchronous: bool,
    spoof_headers: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    local_model_cost_map: None,
    managed_credential: str,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("CHATGPT_ORIGINATOR", "managed-origin")
    monkeypatch.setenv("CHATGPT_USER_AGENT", "managed-client/1")
    payload: Final = {
        "id": "resp_session", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.6-luna",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
    }
    upstream: Final = respx_mock.post("https://chatgpt.test/backend-api/codex/responses").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps({'type': 'response.completed', 'response': payload})}\n\ndata: [DONE]\n\n",
        )
    )
    supplied_headers: Final = {
        **{name: "spoofed" for name in (
            "Authorization", "AUTHORIZATION", "chatgpt-account-id", "ChatGPT-Account-Id",
            "originator", "ORIGINATOR", "user-agent", "User-Agent", "session_id", "SESSION_ID"
        )},
        "X-Custom": "preserved",
    }
    owners: Final = (("owner-a", "key-a"), ("owner-a", "key-a"), ("owner-b", "key-a"), ("owner-a", "key-b"))
    for attempt, (owner, key) in enumerate(owners):
        metadata: Final = {
            "session_id": "shared-original-session",
            "user_api_key_auth": UserAPIKeyAuth(user_id=owner, api_key=key),
            "custom": "untouched",
        }
        params: Final = {
            "model": "chatgpt/gpt-5.6-luna", "api_key": ManagedChatGPTAccessToken("managed-token"),
            "api_base": "https://chatgpt.test/backend-api/codex", "litellm_credential_name": managed_credential,
            "chatgpt_auth_account_id": "spoofed-account", "litellm_metadata": metadata,
            "litellm_trace_id": f"attempt-{attempt}",
            "extra_headers": supplied_headers if spoof_headers else {"X-Custom": "preserved"}, "num_retries": 0,
        }
        if api == "responses":
            response: Final = (
                await litellm.aresponses(input="Hello", stream=False, **params)
                if asynchronous else litellm.responses(input="Hello", stream=False, **params)
            )
            assert response.output_text == "ok"
        else:
            completion: Final = (
                await litellm.acompletion(messages=[{"role": "user", "content": "Hello"}], **params)
                if asynchronous else litellm.completion(messages=[{"role": "user", "content": "Hello"}], **params)
            )
            assert completion.choices[0].message.content == "ok"
        assert metadata["session_id"] == "shared-original-session"
        assert metadata["custom"] == "untouched"

    sessions: Final = tuple(call.request.headers["session_id"] for call in upstream.calls)
    assert len(sessions) == 4
    assert sessions[0] == sessions[1]
    assert len(set((sessions[0], sessions[2], sessions[3]))) == 3
    for call in upstream.calls:
        headers: Final = call.request.headers
        assert str(UUID(headers["session_id"])) == headers["session_id"]
        assert headers["session_id"] != "shared-original-session"
        assert headers["x-custom"] == "preserved"
        for name, value in (
            ("authorization", "Bearer managed-token"), ("chatgpt-account-id", "managed-account"),
            ("originator", "managed-origin"), ("user-agent", "managed-client/1"), ("session_id", headers["session_id"]),
        ):
            assert headers.get_list(name) == [value]
