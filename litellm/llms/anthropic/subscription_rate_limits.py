from collections.abc import Mapping
from typing import Final, Literal, cast

import httpx

AnthropicSubscriptionRateLimitScope = Literal["request", "model", "account"]


def classify_anthropic_subscription_rate_limit(
    error: Exception,
    litellm_params: Mapping[str, object],
) -> AnthropicSubscriptionRateLimitScope | None:
    if getattr(error, "status_code", None) != 429:
        return None
    credential_name: Final = litellm_params.get("litellm_credential_name")
    provider: Final = litellm_params.get("custom_llm_provider")
    model: Final = litellm_params.get("model")
    if (
        not isinstance(credential_name, str)
        or not credential_name
        or (provider != "anthropic" and not (isinstance(model, str) and model.startswith("anthropic/")))
    ):
        return None

    message: Final = str(error).lower()
    if "fast" in message and ("usage credits" in message or "credits are required" in message):
        return "request"

    headers: Final = _response_headers(error)
    if headers is not None and _shared_subscription_window_rejected(headers):
        return "account"
    return "model"


def anthropic_subscription_identity(credential_name: str) -> str:
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.llms.anthropic.oauth_client import AnthropicOAuthTokens

    values: Final = CredentialAccessor.get_credential_values(credential_name)
    token_bundle: Final = values.get("litellm_internal_anthropic_auth_token") if isinstance(values, Mapping) else None
    if isinstance(token_bundle, str):
        try:
            account_id: Final = AnthropicOAuthTokens.from_json(token_bundle).account_id
            if account_id is not None:
                return account_id
        except (TypeError, ValueError):
            pass
    return credential_name


def _response_headers(error: Exception) -> Mapping[str, str] | None:
    headers: Final = cast(object, getattr(error, "litellm_response_headers", None))
    if isinstance(headers, Mapping) and headers:
        return cast(Mapping[str, str], headers)
    response: Final = cast(object, getattr(error, "response", None))
    response_headers: Final = cast(object, getattr(response, "headers", None))
    return cast(httpx.Headers, response_headers) if isinstance(response_headers, httpx.Headers) else None


def _shared_subscription_window_rejected(headers: Mapping[str, str]) -> bool:
    normalized: Final = {name.lower(): value.strip().lower() for name, value in headers.items()}
    status_5h: Final = normalized.get("anthropic-ratelimit-unified-5h-status", "")
    status_7d: Final = normalized.get("anthropic-ratelimit-unified-7d-status", "")
    if status_5h == "rejected" or status_7d == "rejected":
        return True
    if normalized.get("anthropic-ratelimit-unified-status") != "rejected":
        return False

    overage_rejected: Final = any(
        (
            normalized.get("anthropic-ratelimit-unified-7d_oi-status") == "rejected",
            normalized.get("anthropic-ratelimit-unified-overage-status") == "rejected",
            bool(normalized.get("anthropic-ratelimit-unified-overage-disabled-reason")),
            "overage" in normalized.get("anthropic-ratelimit-unified-representative-claim", ""),
        )
    )
    if not overage_rejected:
        return True
    allowed: Final = {"allowed", "allowed_warning"}
    if status_5h in allowed and status_7d in allowed:
        return False
    if status_7d in allowed and not status_5h:
        return not _healthy_utilization(normalized.get("anthropic-ratelimit-unified-5h-utilization"))
    if status_5h in allowed and not status_7d:
        return not _healthy_utilization(normalized.get("anthropic-ratelimit-unified-7d-utilization"))
    return True


def _healthy_utilization(value: str | None) -> bool:
    try:
        utilization: Final = float(value) if value is not None else -1
    except ValueError:
        return False
    return 0 <= utilization < 1
