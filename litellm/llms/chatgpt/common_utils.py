"""
Constants and helpers for ChatGPT subscription OAuth.
"""

import os
import platform
from typing import Any, Final
from uuid import uuid4

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
    for key in ("litellm_trace_id", "litellm_call_id"):
        value = params.get(key)
        if value:
            return str(value)
    return None


def ensure_chatgpt_session_id(litellm_params: Any | None) -> str:
    return get_chatgpt_session_id(litellm_params) or str(uuid4())
