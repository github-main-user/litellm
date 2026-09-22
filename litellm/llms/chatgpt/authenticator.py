import json
import os
import time
from collections.abc import Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger

from .common_utils import (
    CHATGPT_API_BASE,
    CHATGPT_DEVICE_VERIFY_URL,
    GetAccessTokenError,
    RefreshAccessTokenError,
)
from .oauth_client import (
    ChatGPTAuthorizationCode,
    ChatGPTDeviceCode,
    ChatGPTOAuthClient,
)

TOKEN_EXPIRY_SKEW_SECONDS: Final = 60
DEVICE_CODE_TIMEOUT_SECONDS: Final = 15 * 60
DEVICE_CODE_COOLDOWN_SECONDS: Final = 5 * 60
DEVICE_CODE_POLL_SLEEP_SECONDS: Final = 5

OPENAI_AUTH_CLAIM_KEY: Final = "https://api.openai.com/auth"

JsonObject: TypeAlias = Mapping[str, JsonValue]

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(JsonObject)


def _optional_str(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) else None


class Authenticator:
    def __init__(self, oauth_client: ChatGPTOAuthClient | None = None) -> None:
        self.token_dir = os.getenv(
            "CHATGPT_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/chatgpt"),
        )
        self.auth_file = os.path.join(self.token_dir, os.getenv("CHATGPT_AUTH_FILE", "auth.json"))
        self.oauth_client = oauth_client or ChatGPTOAuthClient()
        self.file_auth_enabled = os.getenv("CHATGPT_ALLOW_FILE_AUTH", "true").strip().lower() == "true"
        if self.file_auth_enabled:
            self._ensure_token_dir()

    def get_api_base(self) -> str:
        return os.getenv("CHATGPT_API_BASE") or os.getenv("OPENAI_CHATGPT_API_BASE") or CHATGPT_API_BASE

    def get_access_token(self) -> str:
        auth_data: Final = self._read_auth_file()
        if auth_data:
            access_token: Final = _optional_str(auth_data.get("access_token"))
            if access_token and not self._is_token_expired(auth_data, access_token):
                return access_token
            refresh_token: Final = _optional_str(auth_data.get("refresh_token"))
            if refresh_token:
                try:
                    refreshed: Final = self._refresh_tokens(refresh_token)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning("ChatGPT refresh token failed, re-login required: %s", exc)

        cooldown_remaining: Final = self._get_device_code_cooldown_remaining(auth_data)
        if cooldown_remaining > 0:
            token: Final = self._wait_for_access_token(cooldown_remaining)
            if token:
                return token

        tokens: Final = self._login_device_code()
        return tokens["access_token"]

    def get_account_id(self) -> str | None:
        auth_data: Final = self._read_auth_file()
        if not auth_data:
            return None
        account_id: Final = _optional_str(auth_data.get("account_id"))
        if account_id:
            return account_id
        id_token: Final = auth_data.get("id_token")
        access_token: Final = auth_data.get("access_token")
        derived: Final = self._extract_account_id(_optional_str(id_token or access_token))
        if derived:
            self._write_auth_file({**auth_data, "account_id": derived})
        return derived

    def _ensure_token_dir(self) -> None:
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _require_file_auth(self) -> None:
        if not self.file_auth_enabled:
            raise GetAccessTokenError(
                status_code=401,
                message="ChatGPT file authentication is disabled. Configure a managed ChatGPT credential.",
            )

    def _read_auth_file(self) -> JsonObject | None:
        self._require_file_auth()
        try:
            with open(self.auth_file, "r") as f:
                return _JSON_OBJECT_ADAPTER.validate_python(json.load(f))
        except OSError:
            return None
        except (json.JSONDecodeError, ValidationError) as exc:
            verbose_logger.warning("Invalid ChatGPT auth file: %s", exc)
            return None

    def _write_auth_file(self, data: JsonObject) -> None:
        self._require_file_auth()
        try:
            with open(self.auth_file, "w") as f:
                json.dump(data, f)
        except OSError as exc:
            verbose_logger.error("Failed to write ChatGPT auth file: %s", exc)

    def _is_token_expired(self, auth_data: JsonObject, access_token: str) -> bool:
        stored_expires_at: Final = auth_data.get("expires_at")
        if isinstance(stored_expires_at, (int, float)):
            return time.time() >= float(stored_expires_at) - TOKEN_EXPIRY_SKEW_SECONDS
        derived_expires_at: Final = self._get_expires_at(access_token)
        if derived_expires_at:
            self._write_auth_file({**auth_data, "expires_at": derived_expires_at})
        if derived_expires_at is None:
            return True
        return time.time() >= float(derived_expires_at) - TOKEN_EXPIRY_SKEW_SECONDS

    def _get_expires_at(self, token: str) -> int | None:
        return self.oauth_client.get_expires_at(token)

    def _decode_jwt_claims(self, token: str) -> JsonObject:
        try:
            return _JSON_OBJECT_ADAPTER.validate_python(
                self.oauth_client.decode_jwt_claims(token)
            )
        except ValidationError:
            return {}

    def _extract_account_id(self, token: str | None) -> str | None:
        return self.oauth_client.extract_account_id(token)

    def _login_device_code(self) -> dict[str, str]:
        cooldown_remaining: Final = self._get_device_code_cooldown_remaining(self._read_auth_file())
        if cooldown_remaining > 0:
            token: Final = self._wait_for_access_token(cooldown_remaining)
            if token:
                return {"access_token": token}

        device_code: Final = self.oauth_client.request_device_code()
        self._record_device_code_request()
        print(  # noqa: T201
            "Sign in with ChatGPT using device code:\n"
            f"1) Visit {CHATGPT_DEVICE_VERIFY_URL}\n"
            f"2) Enter code: {device_code.user_code}\n"
            "Device codes are a common phishing target. Never share this code.",
            flush=True,
        )
        auth_code: Final = self._poll_for_authorization_code(device_code)
        tokens: Final = self._exchange_code_for_tokens(auth_code)
        auth_data: Final = self._build_auth_record(tokens)
        self._write_auth_file(auth_data)
        return tokens

    def _request_device_code(self) -> dict[str, str]:
        device_code: Final = self.oauth_client.request_device_code()
        return {
            "device_auth_id": device_code.device_auth_id,
            "user_code": device_code.user_code,
            "interval": str(device_code.interval_seconds),
        }

    def _poll_for_authorization_code(
        self,
        device_code: ChatGPTDeviceCode | dict[str, str],
    ) -> dict[str, str]:
        typed_device_code: Final = (
            device_code
            if isinstance(device_code, ChatGPTDeviceCode)
            else ChatGPTDeviceCode(
                device_auth_id=device_code["device_auth_id"],
                user_code=device_code["user_code"],
                interval_seconds=int(device_code.get("interval", "5")),
            )
        )
        start_time: Final = time.time()
        while time.time() - start_time < DEVICE_CODE_TIMEOUT_SECONDS:
            authorization: Final = self.oauth_client.poll_authorization(typed_device_code)
            if authorization is not None:
                return {
                    "authorization_code": authorization.authorization_code,
                    "code_verifier": authorization.code_verifier,
                }
            time.sleep(max(typed_device_code.interval_seconds, DEVICE_CODE_POLL_SLEEP_SECONDS))

        raise GetAccessTokenError(
            message="Timed out waiting for device authorization",
            status_code=408,
        )

    def _exchange_code_for_tokens(self, code_data: dict[str, str]) -> dict[str, str]:
        tokens: Final = self.oauth_client.exchange_code(
            ChatGPTAuthorizationCode(
                authorization_code=code_data["authorization_code"],
                code_verifier=code_data["code_verifier"],
            )
        )
        return {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
        }

    def _refresh_tokens(self, refresh_token: str) -> dict[str, str]:
        tokens: Final = self.oauth_client.refresh(refresh_token)
        refreshed: Final = {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
        }
        auth_data: Final = self._build_auth_record(refreshed)
        self._write_auth_file(auth_data)
        return refreshed

    def _build_auth_record(self, tokens: dict[str, str]) -> JsonObject:
        access_token: Final = tokens.get("access_token")
        id_token: Final = tokens.get("id_token")
        expires_at: Final = self._get_expires_at(access_token) if access_token else None
        account_id: Final = self._extract_account_id(id_token or access_token)
        return {
            "access_token": access_token,
            "refresh_token": tokens.get("refresh_token"),
            "id_token": id_token,
            "expires_at": expires_at,
            "account_id": account_id,
        }

    def _get_device_code_cooldown_remaining(self, auth_data: JsonObject | None) -> float:
        if not auth_data:
            return 0.0
        requested_at: Final = auth_data.get("device_code_requested_at")
        if not isinstance(requested_at, (int, float, str)):
            return 0.0
        try:
            requested_seconds: Final = float(requested_at)
        except (TypeError, ValueError):
            return 0.0
        elapsed: Final = time.time() - requested_seconds
        remaining: Final = DEVICE_CODE_COOLDOWN_SECONDS - elapsed
        return max(0.0, remaining)

    def _record_device_code_request(self) -> None:
        auth_data: Final = self._read_auth_file() or {}
        self._write_auth_file({**auth_data, "device_code_requested_at": time.time()})

    def _wait_for_access_token(self, timeout_seconds: float) -> str | None:
        deadline: Final = time.time() + timeout_seconds
        while time.time() < deadline:
            auth_data = self._read_auth_file()
            if auth_data:
                access_token = _optional_str(auth_data.get("access_token"))
                if access_token and not self._is_token_expired(auth_data, access_token):
                    return access_token
            sleep_for = min(DEVICE_CODE_POLL_SLEEP_SECONDS, max(0.0, deadline - time.time()))
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        return None
