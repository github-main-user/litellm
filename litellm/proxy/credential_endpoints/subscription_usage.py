import asyncio
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Final, Literal, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.anthropic_oauth import get_anthropic_oauth_credential_hook
from litellm.proxy.credential_endpoints.chatgpt_oauth import get_chatgpt_oauth_credential_hook
from litellm.repositories.credentials_repository import CredentialsRepository

Provider = Literal["anthropic", "chatgpt"]
UsageRequester = Callable[[Provider, str, str | None, str | None], Awaitable[Mapping[str, object]]]
CredentialFinder = Callable[[str], Awaitable[CredentialItem | None]]

ANTHROPIC_USAGE_URL: Final = "https://api.anthropic.com/api/oauth/usage"
CHATGPT_USAGE_URL: Final = "https://chatgpt.com/backend-api/wham/usage"
_SUCCESS_TTL_SECONDS: Final = 60.0
_ERROR_TTL_SECONDS: Final = 15.0
_MAX_CACHE_ENTRIES: Final = 256
_PAYLOAD_ADAPTER: Final = TypeAdapter(dict[str, object])
router: Final = APIRouter()


class SubscriptionUsageWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    used_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    resets_at: datetime | None


class SubscriptionUsageResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    credential_name: str
    provider: Provider
    status: Literal["ok", "unavailable"]
    fetched_at: datetime | None
    plan_type: str | None
    windows: tuple[SubscriptionUsageWindow, ...]
    error: str | None


@dataclass(frozen=True, slots=True, repr=False)
class SubscriptionUsageAuth:
    provider: Provider
    access_token: str
    proxy_url: str | None
    account_id: str | None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    response: SubscriptionUsageResponse


class _UsageFetchError(RuntimeError):
    pass


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _PAYLOAD_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _finite_number(value: object, *, minimum: float = 0, maximum: float | None = None) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number: Final = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        return None
    return number


def _iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _epoch_datetime(value: object) -> datetime | None:
    epoch: Final = _finite_number(value, minimum=0)
    if epoch is None or epoch == 0:
        return None
    try:
        return datetime.fromtimestamp(epoch, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _duration_label(seconds_value: object, fallback: str) -> str:
    seconds: Final = _finite_number(seconds_value, minimum=1)
    if seconds is None:
        return fallback
    minutes: Final = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m"
    hours: Final = minutes / 60
    if hours < 24:
        return f"{int(hours)}h"
    return f"{int(hours / 24)}d"


def _anthropic_limit_window(entry: Mapping[str, object]) -> SubscriptionUsageWindow | None:
    percent: Final = _finite_number(entry.get("percent"), maximum=100)
    kind_value: Final = entry.get("kind")
    kind: Final = kind_value if isinstance(kind_value, str) else ""
    if percent is None:
        return None
    if kind == "session":
        return SubscriptionUsageWindow(
            key=kind,
            label="5h",
            used_percent=percent,
            resets_at=_iso_datetime(entry.get("resets_at")),
        )
    if kind == "weekly_all":
        return SubscriptionUsageWindow(
            key=kind,
            label="7d",
            used_percent=percent,
            resets_at=_iso_datetime(entry.get("resets_at")),
        )
    if kind != "weekly_scoped":
        return None
    scope: Final = _mapping(entry.get("scope"))
    model: Final = _mapping(scope.get("model")) if scope is not None else None
    display_name: Final = model.get("display_name") if model is not None else None
    if not isinstance(display_name, str) or not display_name.strip() or len(display_name) > 128:
        return None
    return SubscriptionUsageWindow(
        key=f"{kind}:{display_name.strip()}",
        label=f"7d {display_name.strip()}",
        used_percent=percent,
        resets_at=_iso_datetime(entry.get("resets_at")),
    )


def _anthropic_legacy_label(key: str) -> str | None:
    if len(key) > 255:
        return None
    for prefix, label in (("five_hour", "5h"), ("seven_day", "7d")):
        if key == prefix:
            return label
        if key.startswith(f"{prefix}_"):
            return f"{label} {key[len(prefix) + 1 :].replace('_', ' ')}"
    return None


def parse_anthropic_usage(payload: Mapping[str, object]) -> tuple[SubscriptionUsageWindow, ...]:
    limits: Final = payload.get("limits")
    modern: Final = tuple(
        window
        for raw_entry in (cast(list[object], limits) if isinstance(limits, list) else ())
        for entry in (_mapping(raw_entry),)
        if entry is not None
        for window in (_anthropic_limit_window(entry),)
        if window is not None
    )
    if modern:
        return tuple(sorted(modern, key=lambda window: (len(window.label), window.label)))
    legacy: Final = tuple(
        SubscriptionUsageWindow(
            key=key,
            label=label,
            used_percent=percent,
            resets_at=_iso_datetime(value.get("resets_at")),
        )
        for key, raw_value in payload.items()
        for value in (_mapping(raw_value),)
        if value is not None
        for label in (_anthropic_legacy_label(key),)
        for percent in (_finite_number(value.get("utilization"), maximum=100),)
        if label is not None and percent is not None
    )
    return tuple(sorted(legacy, key=lambda window: (len(window.label), window.label)))


def _chatgpt_window(
    key: str,
    fallback: str,
    value: object,
) -> SubscriptionUsageWindow | None:
    window: Final = _mapping(value)
    if window is None:
        return None
    percent: Final = _finite_number(window.get("used_percent"), maximum=100)
    if percent is None:
        return None
    return SubscriptionUsageWindow(
        key=key,
        label=_duration_label(window.get("limit_window_seconds"), fallback),
        used_percent=percent,
        resets_at=_epoch_datetime(window.get("reset_at")),
    )


def parse_chatgpt_usage(payload: Mapping[str, object]) -> tuple[str | None, tuple[SubscriptionUsageWindow, ...]]:
    rate_limit: Final = _mapping(payload.get("rate_limit"))
    windows: Final = tuple(
        window
        for key, fallback in (("primary_window", "5h"), ("secondary_window", "7d"))
        for window in (_chatgpt_window(key, fallback, rate_limit.get(key) if rate_limit is not None else None),)
        if window is not None
    )
    plan_value: Final = payload.get("plan_type")
    plan: Final = (
        plan_value.strip()
        if isinstance(plan_value, str) and plan_value.strip() and len(plan_value.strip()) <= 255
        else None
    )
    return plan, windows


UsageAuthResolver = Callable[[Provider, str], Awaitable[SubscriptionUsageAuth]]


class SubscriptionUsageService:
    def __init__(
        self,
        requester: UsageRequester | None = None,
        credential_finder: CredentialFinder | None = None,
        auth_resolver: UsageAuthResolver | None = None,
    ) -> None:
        self._requester: Final = requester or self._request_payload
        self._credential_finder: Final = credential_finder or self._find_credential
        self._auth_resolver: Final = auth_resolver or self._resolve_auth
        self._cache: Final[OrderedDict[str, _CacheEntry]] = OrderedDict()
        self._inflight: Final[dict[str, asyncio.Task[SubscriptionUsageResponse]]] = {}
        self._lock: Final = asyncio.Lock()

    async def get(self, credential_name: str) -> SubscriptionUsageResponse:
        credential: Final = await self._credential_finder(credential_name)
        if credential is None:
            raise HTTPException(status_code=404, detail="Credential not found")
        provider: Final = self._supported_provider(credential)
        if provider is None:
            raise HTTPException(status_code=409, detail="Credential is not a supported OAuth subscription")
        try:
            auth: Final = await self._auth_resolver(provider, credential_name)
        except Exception:  # noqa: BLE001  # OAuth and storage failures must not expose credential details
            return self._unavailable(credential_name, provider, "Credential authentication is unavailable")
        cache_key: Final = self._cache_key(credential_name, auth)
        async with self._lock:
            now: Final = time.monotonic()
            self._remove_expired(now)
            cached: Final = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached.response
            existing: Final = self._inflight.get(cache_key)
            task: Final = existing or asyncio.create_task(self._fetch_and_cache(cache_key, credential_name, auth))
            if existing is None:
                self._inflight[cache_key] = task
        return await asyncio.shield(task)

    async def _find_credential(self, credential_name: str) -> CredentialItem | None:
        cached: Final = CredentialAccessor.find_credential(credential_name)
        if cached is not None:
            return cached
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return None
        try:
            return await CredentialsRepository(prisma_client).find_by_name(credential_name)
        except Exception:  # noqa: BLE001  # Database adapter errors are untyped and must be sanitized
            raise HTTPException(status_code=503, detail="Credential store is unavailable") from None

    @staticmethod
    def _supported_provider(credential: CredentialItem) -> Provider | None:
        metadata: Final = _PAYLOAD_ADAPTER.validate_python(credential.model_dump(include={"credential_info"}))
        info: Final = _mapping(metadata.get("credential_info"))
        if info is None or info.get("auth_type") != "oauth":
            return None
        if info.get("provider") == "anthropic":
            return "anthropic"
        if info.get("provider") == "chatgpt":
            return "chatgpt"
        return None

    @staticmethod
    async def _resolve_auth(provider: Provider, credential_name: str) -> SubscriptionUsageAuth:
        if provider == "anthropic":
            (
                access_token,
                proxy_url,
                account_id,
            ) = await get_anthropic_oauth_credential_hook().get_subscription_usage_auth(credential_name)
        else:
            access_token, proxy_url, account_id = await get_chatgpt_oauth_credential_hook().get_subscription_usage_auth(
                credential_name
            )
        return SubscriptionUsageAuth(provider, access_token, proxy_url, account_id)

    @staticmethod
    def _cache_key(credential_name: str, auth: SubscriptionUsageAuth) -> str:
        source: Final = "\0".join(
            (credential_name, auth.provider, auth.access_token, auth.proxy_url or "", auth.account_id or "")
        )
        return hashlib.sha256(source.encode()).hexdigest()

    async def _fetch_and_cache(
        self,
        cache_key: str,
        credential_name: str,
        auth: SubscriptionUsageAuth,
    ) -> SubscriptionUsageResponse:
        try:
            response: Final = await self._fetch(credential_name, auth)
            ttl: Final = _SUCCESS_TTL_SECONDS if response.status == "ok" else _ERROR_TTL_SECONDS
            async with self._lock:
                self._cache[cache_key] = _CacheEntry(time.monotonic() + ttl, response)
                self._cache.move_to_end(cache_key)
                while len(self._cache) > _MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
            return response
        finally:
            async with self._lock:
                if self._inflight.get(cache_key) is asyncio.current_task():
                    self._inflight.pop(cache_key, None)

    async def _fetch(
        self,
        credential_name: str,
        auth: SubscriptionUsageAuth,
    ) -> SubscriptionUsageResponse:
        try:
            payload: Final = await self._requester(auth.provider, auth.access_token, auth.proxy_url, auth.account_id)
            plan, windows = (
                (None, parse_anthropic_usage(payload)) if auth.provider == "anthropic" else parse_chatgpt_usage(payload)
            )
            if not windows:
                raise _UsageFetchError("Usage provider returned an invalid response")
            return SubscriptionUsageResponse(
                credential_name=credential_name,
                provider=auth.provider,
                status="ok",
                fetched_at=datetime.now(timezone.utc),
                plan_type=plan,
                windows=windows,
                error=None,
            )
        except _UsageFetchError as error:
            return self._unavailable(credential_name, auth.provider, str(error))
        except Exception:  # noqa: BLE001  # Requester failures must not expose tokens or proxy credentials
            return self._unavailable(credential_name, auth.provider, "Usage provider request failed")

    def _remove_expired(self, now: float) -> None:
        for key in tuple(key for key, entry in self._cache.items() if entry.expires_at <= now):
            self._cache.pop(key, None)

    @staticmethod
    def _unavailable(credential_name: str, provider: Provider, error: str) -> SubscriptionUsageResponse:
        return SubscriptionUsageResponse(
            credential_name=credential_name,
            provider=provider,
            status="unavailable",
            fetched_at=None,
            plan_type=None,
            windows=(),
            error=error,
        )

    @staticmethod
    async def _request_payload(
        provider: Provider,
        access_token: str,
        proxy_url: str | None,
        account_id: str | None,
    ) -> Mapping[str, object]:
        url, headers = (
            (
                ANTHROPIC_USAGE_URL,
                {
                    "Authorization": f"Bearer {access_token}",
                    "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": "LiteLLM",
                    "Accept": "application/json",
                },
            )
            if provider == "anthropic"
            else (
                CHATGPT_USAGE_URL,
                {
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                    **({"ChatGPT-Account-Id": account_id} if account_id is not None else {}),
                },
            )
        )
        timeout: Final = httpx.Timeout(10.0, connect=5.0)
        try:
            async with httpx.AsyncClient(
                proxy=proxy_url, timeout=timeout, trust_env=proxy_url is None, follow_redirects=False
            ) as client:
                response: Final = await client.get(url, headers=headers)
        except httpx.TimeoutException:
            raise _UsageFetchError("Usage provider request timed out") from None
        except httpx.HTTPError:
            raise _UsageFetchError("Usage provider request failed") from None
        if response.status_code != 200:
            raise _UsageFetchError(f"Usage provider returned HTTP {response.status_code}")
        try:
            return _PAYLOAD_ADAPTER.validate_json(response.content, strict=True)
        except (ValueError, ValidationError):
            raise _UsageFetchError("Usage provider returned an invalid response") from None


_usage_service: Final = SubscriptionUsageService()


def get_subscription_usage_service() -> SubscriptionUsageService:
    return _usage_service


@router.get(
    "/credentials/{credential_name:path}/usage",
    response_model=SubscriptionUsageResponse,
    tags=["credential management"],
)
async def get_credential_subscription_usage(
    credential_name: Annotated[str, Path(description="The credential name, percent-decoded; may contain slashes")],
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    service: Annotated[SubscriptionUsageService, Depends(get_subscription_usage_service)],
) -> SubscriptionUsageResponse:
    if user.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value):
        raise HTTPException(status_code=403, detail="Only proxy administrators may view credential usage")
    return await service.get(credential_name)
