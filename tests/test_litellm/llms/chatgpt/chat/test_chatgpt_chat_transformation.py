from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.oauth_client import ChatGPTTokens, ManagedChatGPTAccessToken
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.parametrize("access_token", [None, "unmanaged-token"])
def test_chat_requires_managed_credentials_even_with_legacy_auth_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, access_token: str | None
) -> None:
    token_dir: Final = tmp_path / "chatgpt"
    token_dir.mkdir()
    auth_file: Final = token_dir / "auth.json"
    contents: Final = ChatGPTTokens("file-token", "refresh", "id", 9999999999, "account").to_json()
    auth_file.write_text(contents)
    monkeypatch.setenv("CHATGPT_ALLOW_FILE_AUTH", "true")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
    monkeypatch.setattr(litellm, "credential_list", [])

    with pytest.raises(AuthenticationError, match="managed OAuth credential"):
        ChatGPTConfig().validate_environment(
            headers={}, model="test-model", messages=[], optional_params={}, litellm_params={}, api_key=access_token
        )

    assert auth_file.read_text() == contents


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["unregistered-subscription-model", "responses/unregistered-subscription-model"])
async def test_chat_rejects_unmanaged_tokens_before_sending(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, asynchronous: bool, model: str
) -> None:
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    params: Final = {
        "model": f"chatgpt/{model}",
        "api_key": "unmanaged-token",
        "messages": [{"role": "user", "content": "hello"}],
        "num_retries": 0,
    }
    with pytest.raises(AuthenticationError, match="managed OAuth credential"):
        if asynchronous:
            await litellm.acompletion(**params)
        else:
            litellm.completion(**params)
    assert len(respx_mock.calls) == 0


def test_provider_resolution_does_not_create_auth_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CHATGPT_API_BASE", "https://chatgpt.example.com")
    result: Final = ChatGPTConfig()._get_openai_compatible_provider_info(
        model="test-model", api_base=None, api_key=None, custom_llm_provider="chatgpt"
    )

    assert result == ("https://chatgpt.example.com", None, "chatgpt")
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_managed_credentials_reach_legacy_chat_transport(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, asynchronous: bool
) -> None:
    tokens: Final = ChatGPTTokens("access-a", "refresh-a", "id-a", 9999999999, "account-a")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="subscription-a",
                credential_info={"provider": "chatgpt", "auth_type": "oauth"},
                credential_values={"litellm_internal_chatgpt_auth_token": tokens.to_json()},
            )
        ],
    )
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("CHATGPT_ORIGINATOR", "managed-origin")
    monkeypatch.setenv("CHATGPT_USER_AGENT", "managed-client/1")
    upstream: Final = respx_mock.post("https://chatgpt.test/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "unregistered-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )
    )
    params: Final = {
        "model": "chatgpt/unregistered-model",
        "api_key": ManagedChatGPTAccessToken(tokens.access_token),
        "chatgpt_auth_account_id": "spoofed-account",
        "api_base": "https://chatgpt.test",
        "litellm_credential_name": "subscription-a",
        "messages": [{"role": "user", "content": "hello"}],
        "num_retries": 0,
        "extra_headers": {
            **{name: "spoofed" for name in (
                "Authorization", "AUTHORIZATION", "chatgpt-account-id", "ChatGPT-Account-Id",
                "originator", "ORIGINATOR", "user-agent", "User-Agent", "session_id", "SESSION_ID"
            )},
            "X-Custom": "preserved",
        },
    }
    for owner, key in (("owner-a", "key-a"), ("owner-a", "key-a"), ("owner-b", "key-a"), ("owner-a", "key-b")):
        metadata: Final = {"session_id": "original", "user_api_key_auth": UserAPIKeyAuth(user_id=owner, api_key=key)}
        request_params: Final = {**params, "metadata": metadata}
        response: Final = (
            await litellm.acompletion(**request_params) if asynchronous else litellm.completion(**request_params)
        )
        assert response.choices[0].message.content == "ok"
        assert metadata["session_id"] == "original"
    sessions: Final = tuple(call.request.headers["session_id"] for call in upstream.calls)
    assert len(sessions) == 4
    assert sessions[0] == sessions[1]
    assert len(set((sessions[0], sessions[2], sessions[3]))) == 3
    for call in upstream.calls:
        for name, value in (
            ("authorization", "Bearer access-a"), ("chatgpt-account-id", "account-a"),
            ("originator", "managed-origin"), ("user-agent", "managed-client/1"),
        ):
            assert call.request.headers.get_list(name) == [value]
        assert call.request.headers["x-custom"] == "preserved"


@pytest.mark.parametrize("access_token", ["access-a", "unmanaged-token"])
def test_chat_uses_only_selected_managed_credentials(monkeypatch: pytest.MonkeyPatch, access_token: str) -> None:
    tokens: Final = ChatGPTTokens("access-a", "refresh-a", "id-a", 9999999999, "account-a")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="subscription-a",
                credential_info={"provider": "chatgpt", "auth_type": "oauth"},
                credential_values={"litellm_internal_chatgpt_auth_token": tokens.to_json()},
            )
        ],
    )
    params: Final = {
        "headers": {},
        "model": "test-model",
        "messages": [],
        "optional_params": {},
        "litellm_params": {"litellm_credential_name": "subscription-a", "chatgpt_auth_account_id": "account-a"},
        "api_key": ManagedChatGPTAccessToken(access_token) if access_token == tokens.access_token else access_token,
    }
    if access_token != tokens.access_token:
        with pytest.raises(AuthenticationError, match="managed OAuth credential"):
            ChatGPTConfig().validate_environment(**params)
        return

    headers: Final = ChatGPTConfig().validate_environment(**params)
    assert headers["Authorization"] == "Bearer access-a"
    assert headers["ChatGPT-Account-Id"] == "account-a"
