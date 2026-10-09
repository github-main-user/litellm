import json
import re
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger

_PAYLOAD: Final = TypeAdapter(dict[str, object])


def _error_fields(body: str | bytes) -> dict[str, object]:
    if len(body) > 65536:
        return {}
    try:
        payload: Final = _PAYLOAD.validate_json(body, strict=True)
        return _PAYLOAD.validate_python(payload.get("error"), strict=True)
    except ValidationError:
        return {}


_ERROR_TYPES: Final = frozenset(
    {
        "authentication_error",
        "permission_error",
        "rate_limit_error",
        "invalid_request_error",
        "overloaded_error",
        "api_error",
        "not_found_error",
        "billing_error",
    }
)
_ENDPOINTS: Final = frozenset({"/v1/messages", "/v1/messages/count_tokens", "/api/oauth/usage"})


def classify_subscription_failure(status: int, message: str) -> str:
    text: Final = message.lower()
    if "third-party apps" in text and "extra usage" in text:
        return "third_party_billing"
    if "out of extra usage" in text:
        return "extra_usage_exhausted"
    if "only authorized for use with claude code" in text:
        return "client_authorization"
    if "oauth access token has been revoked" in text:
        return "token_revoked"
    if status == 401:
        return "authentication"
    if status == 403:
        return "authorization"
    if status == 429:
        return "quota_or_rate_limit"
    if status >= 500:
        return "upstream_unavailable"
    return "request_rejected"


def log_subscription_failure(
    response: httpx.Response,
    body: str | bytes,
    *,
    proxy_configured: bool | None = None,
    logging_obj: object = None,
    credential_name: object = None,
) -> None:
    try:
        request: Final = response.request
    except RuntimeError:
        return
    if not request.headers.get("authorization", "").startswith("Bearer sk-ant-oat01-"):
        return
    if response.status_code < 400:
        return
    error: Final = _error_fields(body)
    message: Final = error.get("message")
    error_type: Final = error.get("type")
    details: Final = getattr(logging_obj, "model_call_details", None)
    params: Final = details.get("litellm_params") if isinstance(details, Mapping) else None
    name: Final = credential_name or (params.get("litellm_credential_name") if isinstance(params, Mapping) else None)
    request_id: Final = response.headers.get("request-id") or response.headers.get("x-request-id")
    diagnostic: Final = {
        "category": classify_subscription_failure(response.status_code, message if isinstance(message, str) else ""),
        "status": response.status_code,
        "error_type": error_type if isinstance(error_type, str) and error_type in _ERROR_TYPES else "unknown",
        "endpoint": request.url.path if request.url.path in _ENDPOINTS else "custom",
        "credential_name": name
        if isinstance(name, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}", name)
        else None,
        "proxy_configured": proxy_configured,
        "request_id": request_id if request_id and re.fullmatch(r"req_[a-zA-Z0-9]{1,64}", request_id) else None,
    }
    verbose_logger.warning("Anthropic subscription request failed: %s", json.dumps(diagnostic, sort_keys=True))
