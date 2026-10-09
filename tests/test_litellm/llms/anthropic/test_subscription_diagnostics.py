import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from litellm.llms.anthropic.subscription_diagnostics import (
    classify_subscription_failure,
    log_subscription_failure,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


@pytest.mark.parametrize(
    "status,message,category",
    [
        (403, "Third-party apps now draw from extra usage, not plan limits.", "third_party_billing"),
        (
            400,
            "You're out of extra usage. Add more at claude.ai/settings/usage and keep going.",
            "extra_usage_exhausted",
        ),
        (403, "This credential is only authorized for use with Claude Code.", "client_authorization"),
        (401, "OAuth access token has been revoked.", "token_revoked"),
        (401, "Unknown", "authentication"),
        (403, "Unknown", "authorization"),
        (429, "Rate limited. Please try again later.", "quota_or_rate_limit"),
        (529, "Overloaded", "upstream_unavailable"),
        (400, "Invalid JSON", "request_rejected"),
    ],
)
def test_failure_categories_do_not_infer_bans(status, message, category):
    assert classify_subscription_failure(status, message) == category


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["/v1/messages", "/v1/messages/count_tokens"])
async def test_transport_logs_safe_subscription_context_without_changing_error(caplog, asynchronous, stream, endpoint):
    caplog.set_level(logging.WARNING, logger="LiteLLM")
    body = {"error": {"type": "permission_error", "message": "You're out of extra usage. secret-response-body"}}
    handler = AsyncHTTPHandler() if asynchronous else HTTPHandler()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            403,
            json=body,
            headers={"request-id": "req_test123", "secret-header": "private"},
        )
    )
    if asynchronous:
        await handler.client.aclose()
        handler.client = httpx.AsyncClient(transport=transport)
    else:
        handler.client.close()
        handler.client = httpx.Client(transport=transport)
    handler.proxy_url = "http://secret-user:secret-password@proxy.invalid:8080"
    context = SimpleNamespace(model_call_details={"litellm_params": {"litellm_credential_name": "test-subscription"}})
    params = {
        "url": "https://api.anthropic.com" + endpoint + "?secret-query=value",
        "headers": {"authorization": "Bearer sk-ant-oat01-secret-token"},
        "json": {"messages": [{"role": "user", "content": "secret-conversation"}]},
        "logging_obj": context,
        "stream": stream,
    }
    try:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            if asynchronous:
                await handler.post(**params)
            else:
                handler.post(**params)
        assert raised.value.response.status_code == 403
        assert json.loads(raised.value.response.content) == body
        records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("Anthropic subscription request failed: ")
        ]
        assert len(records) == 1
        diagnostic = json.loads(records[0].getMessage().split(": ", 1)[1])
        assert diagnostic == {
            "category": "extra_usage_exhausted",
            "status": 403,
            "error_type": "permission_error",
            "endpoint": endpoint,
            "credential_name": "test-subscription",
            "proxy_configured": True,
            "request_id": "req_test123",
        }
        assert "secret" not in records[0].getMessage()
    finally:
        if asynchronous:
            await handler.close()
        else:
            handler.close()


@pytest.mark.parametrize(
    "body",
    [b"not JSON secret", b"[]", b'{"error":"secret"}', b"x" * 65537],
    ids=["text", "array", "bad-error", "oversized"],
)
def test_malformed_body_and_untrusted_fields_never_escape(caplog, body):
    caplog.set_level(logging.WARNING, logger="LiteLLM")
    response = httpx.Response(
        403,
        request=httpx.Request(
            "POST",
            "https://custom.invalid/private-path?secret=true",
            headers={"authorization": "Bearer sk-ant-oat01-synthetic"},
        ),
        headers={"request-id": "secret-id"},
    )
    log_subscription_failure(response, body, credential_name="unsafe\nname", proxy_configured=False)
    diagnostic = json.loads(caplog.records[-1].getMessage().split(": ", 1)[1])
    assert diagnostic == {
        "category": "authorization",
        "status": 403,
        "error_type": "unknown",
        "endpoint": "custom",
        "credential_name": None,
        "proxy_configured": False,
        "request_id": None,
    }


@pytest.mark.parametrize("token,status", [("sk-ant-api03-synthetic", 403), ("sk-ant-oat01-synthetic", 200)])
def test_api_keys_and_successes_do_not_emit_subscription_failures(caplog, token, status):
    response = httpx.Response(
        status,
        request=httpx.Request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={"authorization": f"Bearer {token}"},
        ),
    )
    log_subscription_failure(response, '{"error":{"message":"secret"}}')
    assert not [
        record for record in caplog.records if record.getMessage().startswith("Anthropic subscription request failed: ")
    ]
