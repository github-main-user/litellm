import base64
import json
import time
from pathlib import Path
from typing import Final
from unittest.mock import Mock, mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.oauth_client import ChatGPTOAuthClient


@pytest.mark.parametrize("stored_token", [False, True])
def test_disabled_file_auth_never_uses_local_credentials_or_starts_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_token: bool
) -> None:
    token_dir: Final = tmp_path / "chatgpt"
    auth_file: Final = token_dir / "auth.json"
    auth_data: Final = json.dumps(
        {"access_token": "stored-token", "account_id": "stored-account", "expires_at": time.time() + 3600}
    )
    if stored_token:
        token_dir.mkdir()
        auth_file.write_text(auth_data)
    monkeypatch.setenv("CHATGPT_ALLOW_FILE_AUTH", "false")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    oauth_client: Final = Mock(spec=ChatGPTOAuthClient)
    authenticator: Final = Authenticator(oauth_client=oauth_client)

    with pytest.raises(GetAccessTokenError, match="file authentication is disabled"):
        authenticator.get_access_token()
    with pytest.raises(GetAccessTokenError, match="file authentication is disabled"):
        authenticator.get_account_id()

    assert oauth_client.mock_calls == []
    assert token_dir.exists() is stored_token
    if stored_token:
        assert auth_file.read_text() == auth_data



def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"
