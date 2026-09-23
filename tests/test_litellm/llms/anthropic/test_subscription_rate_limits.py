import httpx
import pytest

from litellm.exceptions import RateLimitError
from litellm.llms.anthropic.subscription_rate_limits import classify_anthropic_subscription_rate_limit


def _error(message: str, headers: dict[str, str] | None = None) -> RateLimitError:
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return RateLimitError(message=message, llm_provider="anthropic", model="claude", response=response)


@pytest.mark.parametrize(
    ("message", "headers", "expected"),
    [
        ("Usage credits are required for fast mode.", None, "request"),
        ("rate limit", {"anthropic-ratelimit-unified-5h-status": "rejected"}, "account"),
        ("rate limit", {"anthropic-ratelimit-unified-7d-status": "rejected"}, "account"),
        (
            "Fable usage window rejected.",
            {
                "anthropic-ratelimit-unified-status": "rejected",
                "anthropic-ratelimit-unified-overage-status": "rejected",
                "anthropic-ratelimit-unified-5h-status": "allowed",
                "anthropic-ratelimit-unified-7d-status": "allowed",
            },
            "model",
        ),
        ("rate limit", None, "model"),
    ],
)
def test_classifies_subscription_rate_limit_scope(
    message: str,
    headers: dict[str, str] | None,
    expected: str,
) -> None:
    assert (
        classify_anthropic_subscription_rate_limit(
            _error(message, headers),
            {"custom_llm_provider": "anthropic", "litellm_credential_name": "subscription"},
        )
        == expected
    )


def test_ignores_unmanaged_anthropic_rate_limit() -> None:
    assert (
        classify_anthropic_subscription_rate_limit(
            _error("rate limit"),
            {"custom_llm_provider": "anthropic"},
        )
        is None
    )
