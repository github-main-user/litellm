import asyncio
import json
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import HTTPException

from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.credential_endpoints.subscription_usage import (
    SubscriptionUsageAuth,
    SubscriptionUsageService,
    get_credential_subscription_usage,
    reset_credential_subscription_usage,
    parse_anthropic_usage,
    parse_chatgpt_reset_credits,
    parse_chatgpt_usage,
)


def _oauth_credential(provider: str = "anthropic") -> CredentialItem:
    return CredentialItem(
        credential_name="subscription",
        credential_info={"provider": provider, "auth_type": "oauth"},
        credential_values={},
    )


def test_anthropic_parser_prefers_modern_limits_and_validates_percentages() -> None:
    payload = {
        "limits": [
            {"kind": "session", "percent": 12.5, "resets_at": "2026-01-02T03:04:05Z"},
            {
                "kind": "weekly_scoped",
                "percent": 40,
                "resets_at": "2026-01-03T03:04:05+00:00",
                "scope": {"model": {"display_name": "Sonnet"}},
            },
            {"kind": "weekly_all", "percent": float("nan")},
            {"kind": "weekly_all", "percent": True},
            {"kind": "weekly_all", "percent": 101},
        ],
        "five_hour": {"utilization": 99, "resets_at": "2026-01-04T03:04:05Z"},
    }

    windows = parse_anthropic_usage(payload)

    assert [window.model_dump(mode="json") for window in windows] == [
        {
            "key": "session",
            "label": "5h",
            "used_percent": 12.5,
            "resets_at": "2026-01-02T03:04:05Z",
        },
        {
            "key": "weekly_scoped:Sonnet",
            "label": "7d Sonnet",
            "used_percent": 40.0,
            "resets_at": "2026-01-03T03:04:05Z",
        },
    ]


def test_anthropic_parser_supports_legacy_fields_without_inventing_zeroes() -> None:
    windows = parse_anthropic_usage(
        {
            "limits": [],
            "five_hour": {"utilization": 0, "resets_at": "not-a-date"},
            "seven_day": None,
            "seven_day_opus": {"utilization": False},
        }
    )

    assert [window.model_dump(mode="json") for window in windows] == [
        {"key": "five_hour", "label": "5h", "used_percent": 0.0, "resets_at": None}
    ]


def test_chatgpt_parser_uses_window_durations_and_rejects_invalid_numbers() -> None:
    plan, windows = parse_chatgpt_usage(
        {
            "plan_type": " plus ",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25,
                    "limit_window_seconds": 18_000,
                    "reset_at": 1_800_000_000,
                },
                "secondary_window": {"used_percent": float("inf"), "limit_window_seconds": 604_800},
            },
        }
    )

    assert plan == "plus"
    assert parse_chatgpt_reset_credits({"rate_limit_reset_credits": {"available_count": 0}}) == 0
    assert parse_chatgpt_reset_credits({"rate_limit_reset_credits": {"available_count": True}}) is None
    assert [window.model_dump(mode="json") for window in windows] == [
        {
            "key": "primary_window",
            "label": "5h",
            "used_percent": 25.0,
            "resets_at": datetime.fromtimestamp(1_800_000_000, timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    ]


@pytest.mark.asyncio
async def test_service_coalesces_requests_caches_results_and_passes_proxy() -> None:
    calls: list[tuple[str | None, str | None]] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def find_credential(name: str) -> CredentialItem | None:
        return _oauth_credential()

    async def resolve_auth(provider: str, name: str) -> SubscriptionUsageAuth:
        return SubscriptionUsageAuth("anthropic", "secret-token", "http://proxy.example:8080", "account-a")

    async def request(provider: str, token: str, proxy: str | None, account: str | None) -> Mapping[str, object]:
        calls.append((proxy, account))
        started.set()
        await release.wait()
        return {"five_hour": {"utilization": 10}}

    service = SubscriptionUsageService(request, find_credential, resolve_auth)
    first = asyncio.create_task(service.get("subscription"))
    await started.wait()
    second = asyncio.create_task(service.get("subscription"))
    release.set()
    first_response, second_response = await asyncio.gather(first, second)
    cached_response = await service.get("subscription")

    assert calls == [("http://proxy.example:8080", "account-a")]
    assert first_response == second_response == cached_response
    assert first_response.status == "ok"


@pytest.mark.asyncio
async def test_service_returns_sanitized_unavailable_without_fake_windows() -> None:
    async def find_credential(name: str) -> CredentialItem | None:
        return _oauth_credential("chatgpt")

    async def resolve_auth(provider: str, name: str) -> SubscriptionUsageAuth:
        return SubscriptionUsageAuth("chatgpt", "private-token", None, "private-account")

    async def request(provider: str, token: str, proxy: str | None, account: str | None) -> Mapping[str, object]:
        raise RuntimeError(f"failed with {token} and {account}")

    response = await SubscriptionUsageService(request, find_credential, resolve_auth).get("subscription")

    assert response.model_dump(mode="json") == {
        "credential_name": "subscription",
        "provider": "chatgpt",
        "status": "unavailable",
        "fetched_at": None,
        "plan_type": None,
        "windows": [],
        "error": "Usage provider request failed",
        "reset_credits_available": None,
    }


@pytest.mark.asyncio
async def test_service_distinguishes_missing_and_non_oauth_credentials() -> None:
    async def missing(name: str) -> CredentialItem | None:
        return None

    async def api_key(name: str) -> CredentialItem | None:
        return CredentialItem(
            credential_name=name,
            credential_info={"provider": "anthropic", "auth_type": "api_key"},
            credential_values={"api_key": "secret"},
        )

    with pytest.raises(HTTPException) as missing_error:
        await SubscriptionUsageService(credential_finder=missing).get("missing")
    with pytest.raises(HTTPException) as unsupported_error:
        await SubscriptionUsageService(credential_finder=api_key).get("api-key")

    assert missing_error.value.status_code == 404
    assert unsupported_error.value.status_code == 409


@pytest.mark.asyncio
async def test_endpoint_is_admin_only() -> None:
    non_admin = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    with pytest.raises(HTTPException) as denied:
        await get_credential_subscription_usage("subscription", non_admin, SubscriptionUsageService())

    assert denied.value.status_code == 403
    with pytest.raises(HTTPException) as reset_denied:
        await reset_credential_subscription_usage("subscription", non_admin, SubscriptionUsageService())
    assert reset_denied.value.status_code == 403


@contextmanager
def _local_proxy(body):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, dict(self.headers)))
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append((self.path, dict(self.headers), json.loads(self.rfile.read(length))))
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://proxy-user:proxy-password@127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_real_proxy_routing_isolates_accounts_and_ignores_no_proxy(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    monkeypatch.setattr(subscription_usage, "ANTHROPIC_USAGE_URL", "http://usage.invalid/anthropic")
    monkeypatch.setattr(subscription_usage, "CHATGPT_USAGE_URL", "http://usage.invalid/chatgpt")
    monkeypatch.setenv("NO_PROXY", "*")
    with _local_proxy({"five_hour": {"utilization": 12}}) as (first_proxy, first), _local_proxy(
        {"rate_limit": {"primary_window": {"used_percent": 34}}}
    ) as (second_proxy, second):
        one = await SubscriptionUsageService._request_payload("anthropic", "first-token", first_proxy, None)
        two = await SubscriptionUsageService._request_payload("chatgpt", "second-token", second_proxy, "account-two")

    assert parse_anthropic_usage(one)[0].used_percent == 12
    assert parse_chatgpt_usage(two)[1][0].used_percent == 34
    assert first[0][0] == "http://usage.invalid/anthropic"
    assert second[0][0] == "http://usage.invalid/chatgpt"
    assert first[0][1]["Authorization"] == "Bearer first-token"
    assert second[0][1]["Authorization"] == "Bearer second-token"
    assert second[0][1]["ChatGPT-Account-Id"] == "account-two"
    assert first[0][1]["Proxy-Authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_chatgpt_reset_uses_credential_proxy_and_invalidates_cached_usage(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    monkeypatch.setattr(subscription_usage, "CHATGPT_RESET_URL", "http://reset.invalid/consume")
    monkeypatch.setenv("NO_PROXY", "*")
    calls = []

    async def find(name):
        return _oauth_credential("chatgpt")

    async def request(*args):
        calls.append(True)
        return {
            "rate_limit": {"primary_window": {"used_percent": len(calls)}},
            "rate_limit_reset_credits": {"available_count": 1 if len(calls) == 1 else 0},
        }

    with _local_proxy({"code": "reset", "windows_reset": 2}) as (proxy, received):
        async def auth(provider, name):
            return SubscriptionUsageAuth("chatgpt", "secret-token", proxy, "account-id")

        service = SubscriptionUsageService(request, find, auth)
        assert (await service.get("subscription")).reset_credits_available == 1
        result = await service.reset_chatgpt("subscription")
        assert result.windows_reset == 2
        assert (await service.get("subscription")).reset_credits_available == 0

    assert len(calls) == 2
    assert received[0][0] == "http://reset.invalid/consume"
    assert received[0][1]["Authorization"] == "Bearer secret-token"
    assert received[0][1]["ChatGPT-Account-Id"] == "account-id"
    assert received[0][1]["Proxy-Authorization"].startswith("Basic ")
    assert received[0][2]["redeem_request_id"]


@pytest.mark.asyncio
async def test_reset_without_credit_does_not_invalidate_usage(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    monkeypatch.setattr(subscription_usage, "CHATGPT_RESET_URL", "http://reset.invalid/consume")

    async def find(name):
        return _oauth_credential("chatgpt")

    with _local_proxy({"code": "no_credit", "windows_reset": 0}) as (proxy, _):
        async def auth(provider, name):
            return SubscriptionUsageAuth("chatgpt", "secret-token", proxy, None)

        with pytest.raises(HTTPException) as error:
            await SubscriptionUsageService(credential_finder=find, auth_resolver=auth).reset_chatgpt("subscription")
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_reset_with_dead_proxy_never_reaches_direct_endpoint(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    with _local_proxy({"code": "reset", "windows_reset": 2}) as (origin, received):
        dead_server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        dead_proxy = f"http://127.0.0.1:{dead_server.server_port}"
        dead_server.server_close()
        monkeypatch.setattr(subscription_usage, "CHATGPT_RESET_URL", f"http://{origin.rsplit('@', 1)[1]}/reset")
        monkeypatch.setenv("NO_PROXY", "*")

        async def find(name):
            return _oauth_credential("chatgpt")

        async def auth(provider, name):
            return SubscriptionUsageAuth("chatgpt", "secret", dead_proxy, None)

        with pytest.raises(HTTPException) as error:
            await SubscriptionUsageService(credential_finder=find, auth_resolver=auth).reset_chatgpt("subscription")
        assert error.value.status_code == 502
        assert not received


@pytest.mark.asyncio
async def test_dead_proxy_never_falls_back_to_reachable_direct_endpoint(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    with _local_proxy({"five_hour": {"utilization": 99}}) as (origin_url, origin_requests):
        dead_server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        dead_proxy = f"http://127.0.0.1:{dead_server.server_port}"
        dead_server.server_close()
        origin_address = origin_url.rsplit("@", 1)[1]
        monkeypatch.setattr(subscription_usage, "ANTHROPIC_USAGE_URL", f"http://{origin_address}/usage")
        monkeypatch.setenv("NO_PROXY", "*")

        async def find(name):
            return _oauth_credential()

        async def auth(provider, name):
            return SubscriptionUsageAuth("anthropic", "private-token", dead_proxy, None)

        response = await SubscriptionUsageService(credential_finder=find, auth_resolver=auth).get("subscription")
        assert response.status == "unavailable"
        assert response.windows == ()
        assert not origin_requests


@pytest.mark.asyncio
async def test_rotating_proxy_token_or_account_invalidates_cache():
    current = [SubscriptionUsageAuth("anthropic", "token-a", "http://proxy-a", "account-a")]
    calls = []

    async def find(name):
        return _oauth_credential()

    async def auth(provider, name):
        return current[0]

    async def request(provider, token, proxy, account):
        calls.append((token, proxy, account))
        return {"five_hour": {"utilization": len(calls)}}

    service = SubscriptionUsageService(request, find, auth)
    assert (await service.get("subscription")).windows[0].used_percent == 1
    for changed in (
        SubscriptionUsageAuth("anthropic", "token-b", "http://proxy-a", "account-a"),
        SubscriptionUsageAuth("anthropic", "token-b", "http://proxy-b", "account-a"),
        SubscriptionUsageAuth("anthropic", "token-b", "http://proxy-b", "account-b"),
    ):
        current[0] = changed
        await service.get("subscription")
        await service.get("subscription")
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_cached_errors_expire_and_retry_without_exposing_provider_body(monkeypatch):
    from litellm.proxy.credential_endpoints import subscription_usage

    calls = []

    async def find(name):
        return _oauth_credential()

    async def auth(provider, name):
        return SubscriptionUsageAuth("anthropic", "secret", None, None)

    async def request(*args):
        calls.append(True)
        if len(calls) == 1:
            raise httpx.ConnectError("secret proxy password")
        return {"five_hour": {"utilization": 0}}

    monkeypatch.setattr(subscription_usage, "_ERROR_TTL_SECONDS", 0.02)
    service = SubscriptionUsageService(request, find, auth)
    first = await service.get("subscription")
    assert first.status == "unavailable"
    assert "password" not in first.model_dump_json()
    assert await service.get("subscription") == first
    assert len(calls) == 1
    await asyncio.sleep(0.03)
    assert (await service.get("subscription")).windows[0].used_percent == 0
    assert len(calls) == 2
