"""
Constants and helpers for ChatGPT subscription OAuth.
"""

import math
import os
import platform
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# OAuth + API constants (derived from openai/codex)
CHATGPT_AUTH_BASE: Final = "https://auth.openai.com"
CHATGPT_DEVICE_CODE_URL: Final = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL: Final = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token"
CHATGPT_OAUTH_TOKEN_URL: Final = f"{CHATGPT_AUTH_BASE}/oauth/token"
CHATGPT_DEVICE_VERIFY_URL: Final = f"{CHATGPT_AUTH_BASE}/codex/device"
CHATGPT_API_BASE: Final = "https://chatgpt.com/backend-api/codex"
CHATGPT_CLIENT_ID: Final = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_ORIGINATOR: Final = "codex_cli_rs"
CODEX_CLI_VERSION: Final = "0.155.1"
DEFAULT_USER_AGENT: Final = f"{DEFAULT_ORIGINATOR}/{CODEX_CLI_VERSION} (Unknown 0; unknown) unknown"
CHATGPT_DEFAULT_INSTRUCTIONS: Final = "You are a helpful assistant."


def chatgpt_quota_reset_seconds(exception: Exception) -> float | None:
    if getattr(exception, "status_code", None) != 429 or getattr(exception, "llm_provider", None) != "chatgpt":
        return None
    response: Final = getattr(exception, "response", None)
    if response is None:
        return None
    try:
        payload: Final = response.json()
    except (ValueError, AttributeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    error: Final = payload.get("error")
    if not isinstance(error, Mapping) or "usage_limit_reached" not in (error.get("code"), error.get("type")):
        return None
    reset: Final = error.get("resets_at", error.get("reset_at"))
    seconds: Final = error.get("resets_in_seconds")
    if reset is None and isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        try:
            remaining_seconds: Final = float(seconds)
        except OverflowError:
            return None
        return remaining_seconds if math.isfinite(remaining_seconds) and remaining_seconds > 0 else None
    if isinstance(reset, bool):
        return None
    if isinstance(reset, (int, float)):
        try:
            timestamp: Final = float(reset)
        except OverflowError:
            return None
    elif isinstance(reset, str):
        try:
            parsed: Final = datetime.fromisoformat(reset.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        timestamp = parsed.timestamp()
    else:
        return None
    remaining: Final = timestamp - time.time()
    return remaining if math.isfinite(remaining) and remaining > 0 else None


class ChatGPTAuthError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: httpx.Headers | dict | None = None,
        body: dict | None = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetDeviceCodeError(ChatGPTAuthError):
    pass


class GetAccessTokenError(ChatGPTAuthError):
    pass


class RefreshAccessTokenError(ChatGPTAuthError):
    pass


def _safe_header_value(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if 32 <= ord(ch) <= 126 else "_" for ch in value)


def _sanitize_user_agent_token(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if (ch.isalnum() or ch in "-_./") else "_" for ch in value)


def _terminal_user_agent() -> str:
    term_program: Final = os.getenv("TERM_PROGRAM")
    if term_program:
        version: Final = os.getenv("TERM_PROGRAM_VERSION")
        token = f"{term_program}/{version}" if version else term_program
        return _sanitize_user_agent_token(token) or "unknown"

    wezterm_version: Final = os.getenv("WEZTERM_VERSION")
    if wezterm_version is not None:
        token = f"WezTerm/{wezterm_version}" if wezterm_version else "WezTerm"
        return _sanitize_user_agent_token(token) or "WezTerm"

    if os.getenv("ITERM_SESSION_ID") or os.getenv("ITERM_PROFILE") or os.getenv("ITERM_PROFILE_NAME"):
        return "iTerm.app"

    if os.getenv("TERM_SESSION_ID"):
        return "Apple_Terminal"

    if os.getenv("KITTY_WINDOW_ID") or "kitty" in (os.getenv("TERM") or ""):
        return "kitty"

    if os.getenv("ALACRITTY_SOCKET") or os.getenv("TERM") == "alacritty":
        return "Alacritty"

    konsole_version: Final = os.getenv("KONSOLE_VERSION")
    if konsole_version is not None:
        token = f"Konsole/{konsole_version}" if konsole_version else "Konsole"
        return _sanitize_user_agent_token(token) or "Konsole"

    if os.getenv("GNOME_TERMINAL_SCREEN"):
        return "gnome-terminal"

    vte_version: Final = os.getenv("VTE_VERSION")
    if vte_version is not None:
        token = f"VTE/{vte_version}" if vte_version else "VTE"
        return _sanitize_user_agent_token(token) or "VTE"

    if os.getenv("WT_SESSION"):
        return "WindowsTerminal"

    term: Final = os.getenv("TERM")
    if term:
        return _sanitize_user_agent_token(term) or "unknown"

    return "unknown"


def get_chatgpt_originator() -> str:
    originator: Final = os.getenv("CHATGPT_ORIGINATOR") or DEFAULT_ORIGINATOR
    return _safe_header_value(originator) or DEFAULT_ORIGINATOR


def get_chatgpt_user_agent(originator: str) -> str:
    override: Final = os.getenv("CHATGPT_USER_AGENT")
    if override:
        return _safe_header_value(override) or DEFAULT_USER_AGENT
    os_type: Final = platform.system() or "Unknown"
    os_version: Final = platform.release() or "0"
    arch: Final = platform.machine() or "unknown"
    terminal_ua: Final = _terminal_user_agent()
    suffix = os.getenv("CHATGPT_USER_AGENT_SUFFIX", "").strip()
    suffix = f" ({suffix})" if suffix else ""
    candidate: Final = f"{originator}/{CODEX_CLI_VERSION} ({os_type} {os_version}; {arch}) {terminal_ua}{suffix}"
    return _safe_header_value(candidate) or DEFAULT_USER_AGENT


def get_chatgpt_default_headers(
    access_token: str,
    account_id: str | None,
    session_id: str | None = None,
) -> dict:
    originator: Final = get_chatgpt_originator()
    user_agent: Final = get_chatgpt_user_agent(originator)
    headers: Final = {
        "Authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "text/event-stream",
        "originator": originator,
        "user-agent": user_agent,
    }
    if session_id:
        headers["session_id"] = session_id
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def get_chatgpt_default_instructions() -> str:
    return os.getenv("CHATGPT_DEFAULT_INSTRUCTIONS") or CHATGPT_DEFAULT_INSTRUCTIONS


def _normalize_litellm_params(litellm_params: Any | None) -> dict:
    if litellm_params is None:
        return {}
    if isinstance(litellm_params, dict):
        return litellm_params
    if hasattr(litellm_params, "model_dump"):
        try:
            return litellm_params.model_dump()
        except Exception:
            return {}
    if hasattr(litellm_params, "dict"):
        try:
            return litellm_params.dict()
        except Exception:
            return {}
    return {}


def get_chatgpt_session_id(litellm_params: Any | None) -> str | None:
    params: Final = _normalize_litellm_params(litellm_params)
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if value:
            return str(value)
    metadata: Final = params.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("session_id")
        if value:
            return str(value)
    prompt_cache_key: Final = params.get("prompt_cache_key")
    if isinstance(prompt_cache_key, str) and prompt_cache_key:
        return str(uuid5(NAMESPACE_URL, f"litellm:prompt-cache:{prompt_cache_key}"))
    for key in ("litellm_trace_id", "litellm_call_id"):
        value = params.get(key)
        if value:
            return str(value)
    return None


def ensure_chatgpt_session_id(litellm_params: Any | None) -> str:
    return get_chatgpt_session_id(litellm_params) or str(uuid4())
