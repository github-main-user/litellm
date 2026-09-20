import asyncio
import hashlib
import json
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from typing import Annotated, Final, Literal, Protocol, cast
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import AuthenticationError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.anthropic.oauth_client import (
    ANTHROPIC_OAUTH_REDIRECT_URI,
    AnthropicOAuthClient,
    AnthropicOAuthError,
    AnthropicOAuthTokens,
)
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.endpoints import CredentialHelperUtils
from litellm.proxy.utils import jsonify_object  # pyright: ignore[reportUnknownVariableType]  # legacy untyped boundary
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import CallTypes

ANTHROPIC_CREDENTIAL_VALUE_KEY: Final = "litellm_internal_anthropic_auth_token"
ANTHROPIC_CREDENTIAL_PROVIDER: Final = "anthropic"
ANTHROPIC_CREDENTIAL_AUTH_TYPE: Final = "oauth"
ANTHROPIC_ATTEMPT_LIFETIME_SECONDS: Final = 10 * 60
ANTHROPIC_EXPIRY_SKEW_SECONDS: Final = 5 * 60
ANTHROPIC_REFRESH_TRANSACTION_TIMEOUT: Final = timedelta(minutes=2)
_CREDENTIAL_NAME_PATTERN: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")

router: Final = APIRouter()


class _CredentialRow(Protocol):
    @property
    def credential_info(self) -> object: ...

    @property
    def credential_values(self) -> object: ...


class _CredentialTable(Protocol):
    async def find_unique(self, *, where: dict[str, str]) -> _CredentialRow | None: ...

    async def create(self, *, data: dict[str, object]) -> object: ...

    async def update(self, *, where: dict[str, str], data: dict[str, object]) -> object: ...


class _Transaction(Protocol):
    litellm_credentialstable: _CredentialTable

    async def execute_raw(self, query: str, *parameters: object) -> object: ...


class _TransactionManager(Protocol):
    async def __aenter__(self) -> _Transaction: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class _TransactionalDatabase(Protocol):
    def tx(self, *, timeout: timedelta) -> _TransactionManager: ...


class _PrismaClient(Protocol):
    db: _TransactionalDatabase


_STRING_OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])
_OBJECT_MAPPING: Final = TypeAdapter(dict[object, object])


class AnthropicOAuthStartRequest(BaseModel):
    credential_name: str = Field(min_length=1, max_length=255)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class AnthropicOAuthCompleteRequest(BaseModel):
    attempt_token: str = Field(min_length=1, max_length=8192, repr=False)
    authorization_code: str = Field(min_length=1, max_length=8192, repr=False)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class AnthropicOAuthStartResponse(BaseModel):
    authorization_url: str
    attempt_token: str
    expires_in_seconds: int


class AnthropicOAuthCompleteResponse(BaseModel):
    status: Literal["connected"]
    credential_name: str


class AnthropicOAuthAttempt(BaseModel):
    credential_name: str
    actor: str
    state: str = Field(repr=False)
    code_verifier: str = Field(repr=False)
    expires_at: float


class ParsedAuthorizationCode(BaseModel):
    code: str = Field(repr=False)
    state: str = Field(repr=False)


def _require_admin(user: UserAPIKeyAuth) -> None:
    if user.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value):
        raise HTTPException(status_code=403, detail="Only proxy administrators may manage Anthropic OAuth credentials")


def _actor(user: UserAPIKeyAuth) -> str:
    return str(user.user_id or "litellm-proxy-admin")


def _encode_attempt(attempt: AnthropicOAuthAttempt) -> str:
    encrypted: Final = encrypt_value_helper(attempt.model_dump_json())
    if not isinstance(encrypted, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime crypto boundary
        raise HTTPException(status_code=500, detail="Unable to protect Anthropic authentication attempt")
    return encrypted


def _decode_attempt(value: str, actor: str) -> AnthropicOAuthAttempt:
    decrypted: Final = decrypt_value_helper(value, ANTHROPIC_CREDENTIAL_VALUE_KEY, exception_type="debug")
    if not isinstance(decrypted, str):
        raise HTTPException(status_code=400, detail="Invalid Anthropic authentication attempt")
    try:
        attempt: Final = AnthropicOAuthAttempt.model_validate_json(decrypted)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid Anthropic authentication attempt") from error
    if attempt.actor != actor:
        raise HTTPException(status_code=400, detail="Invalid Anthropic authentication attempt")
    if time.time() >= attempt.expires_at:
        raise HTTPException(status_code=400, detail="Anthropic authentication attempt expired")
    return attempt


def _authorization_response(value: str) -> ParsedAuthorizationCode:
    text: Final = value.strip()
    if text.startswith(("http://", "https://")):
        try:
            parsed: Final = urlparse(text)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Anthropic authorization response") from None
        expected: Final = urlparse(ANTHROPIC_OAUTH_REDIRECT_URI)
        if parsed.params or (parsed.scheme, parsed.netloc, parsed.path) != (
            expected.scheme, expected.netloc, expected.path
        ):
            raise HTTPException(status_code=400, detail="Invalid Anthropic authorization response")
        parameters: Final = parse_qs(f"{parsed.query}&{parsed.fragment}", keep_blank_values=True)
        code_values: Final = parameters.get("code", [])
        state_values: Final = parameters.get("state", [])
        if len(code_values) != 1 or len(state_values) != 1 or not code_values[0] or not state_values[0]:
            raise HTTPException(status_code=400, detail="Invalid Anthropic authorization response")
        return ParsedAuthorizationCode(code=code_values[0], state=state_values[0])
    parts: Final = text.split("#")
    if len(parts) != 2 or not all(parts):
        raise HTTPException(status_code=400, detail="Invalid Anthropic authorization response")
    return ParsedAuthorizationCode(code=parts[0], state=parts[1])


def _parse_authorization_code(value: str, expected_state: str) -> ParsedAuthorizationCode:
    response: Final = _authorization_response(value)
    if not response.state.isascii() or not secrets.compare_digest(response.state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid Anthropic authorization response")
    return response


def _string_object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        return _STRING_OBJECT_MAPPING.validate_python(value, strict=True)
    except ValidationError:
        return None


def _object_mapping(value: object) -> dict[object, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return None


def _is_managed_credential(credential: CredentialItem) -> bool:
    info: Final = _string_object_mapping(cast(object, credential.credential_info))
    return info is not None and (
        info.get("provider") == ANTHROPIC_CREDENTIAL_PROVIDER
        and info.get("auth_type") == ANTHROPIC_CREDENTIAL_AUTH_TYPE
    )


def _tokens_from_plaintext(credential: CredentialItem) -> AnthropicOAuthTokens | None:
    values: Final = _string_object_mapping(cast(object, credential.credential_values))
    value: Final = values.get(ANTHROPIC_CREDENTIAL_VALUE_KEY) if values is not None else None
    if not isinstance(value, str):
        return None
    try:
        return AnthropicOAuthTokens.from_json(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _tokens_from_row(row: _CredentialRow) -> AnthropicOAuthTokens:
    values: Final = _string_object_mapping(row.credential_values)
    if values is None:
        raise TypeError("Credential values are invalid")
    encrypted: Final = values.get(ANTHROPIC_CREDENTIAL_VALUE_KEY)
    decrypted: Final = (
        decrypt_value_helper(encrypted, ANTHROPIC_CREDENTIAL_VALUE_KEY, exception_type="debug")
        if isinstance(encrypted, str)
        else None
    )
    if not isinstance(decrypted, str):
        raise TypeError("Credential token bundle cannot be decrypted")
    return AnthropicOAuthTokens.from_json(decrypted)


def _credential_info_from_row(row: _CredentialRow) -> dict[str, object]:
    return _string_object_mapping(row.credential_info) or {}


def _lock_key(identity: str, namespace: str = "credential") -> int:
    return int.from_bytes(
        hashlib.blake2b(f"anthropic-oauth:{namespace}:{identity}".encode(), digest_size=8).digest(), "big", signed=True
    )


def _find_cached(credential_name: str) -> CredentialItem | None:
    return next((item for item in litellm.credential_list if item.credential_name == credential_name), None)


def _plaintext_credential(credential_name: str, tokens: AnthropicOAuthTokens) -> CredentialItem:
    return CredentialItem(
        credential_name=credential_name,
        credential_info={"provider": ANTHROPIC_CREDENTIAL_PROVIDER, "auth_type": ANTHROPIC_CREDENTIAL_AUTH_TYPE},
        credential_values={ANTHROPIC_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )


def _database_data(data: dict[str, object]) -> dict[str, object]:
    return _STRING_OBJECT_MAPPING.validate_python(jsonify_object(data), strict=True)


async def _complete_and_store(
    attempt: AnthropicOAuthAttempt,
    code: ParsedAuthorizationCode,
    oauth_client: AnthropicOAuthClient,
) -> AnthropicOAuthTokens:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    database: Final = cast(_PrismaClient, prisma_client).db
    async with database.tx(timeout=ANTHROPIC_REFRESH_TRANSACTION_TIMEOUT) as transaction:
        await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", _lock_key(attempt.credential_name))
        row: Final = await transaction.litellm_credentialstable.find_unique(
            where={"credential_name": attempt.credential_name}
        )
        existing_info: Final = _credential_info_from_row(row) if row is not None else {}
        if row is not None and (
            existing_info.get("provider") != ANTHROPIC_CREDENTIAL_PROVIDER
            or existing_info.get("auth_type") != ANTHROPIC_CREDENTIAL_AUTH_TYPE
        ):
            raise HTTPException(
                status_code=409, detail="Credential name is already used by another authentication type"
            )
        existing_tokens: Final = _tokens_from_row(row) if row is not None else None
        if existing_tokens is not None and existing_tokens.account_id is not None:
            await transaction.execute_raw(
                "SELECT pg_advisory_xact_lock($1::bigint)", _lock_key(existing_tokens.account_id, "account")
            )
        try:
            tokens: Final = await oauth_client.exchange_code(code.code, code.state, attempt.code_verifier)
        except AnthropicOAuthError as error:
            if error.retryable:
                raise HTTPException(
                    status_code=error.status_code,
                    detail="Unable to complete Anthropic authentication",
                    headers=dict(error.response_headers) if error.response_headers is not None else None,
                ) from None
            raise HTTPException(status_code=400, detail="Unable to complete Anthropic authentication") from None
        if (
            existing_tokens is not None
            and existing_tokens.account_id is not None
            and existing_tokens.account_id != tokens.account_id
        ):
            raise HTTPException(status_code=409, detail="Credential name belongs to another Anthropic account")
        stored_tokens: Final = (
            replace(tokens, device_id=existing_tokens.device_id) if existing_tokens is not None else tokens
        )
        plaintext: Final = _plaintext_credential(attempt.credential_name, stored_tokens)
        encrypted: Final = CredentialHelperUtils.encrypt_credential_values(plaintext)
        dumped: Final = _STRING_OBJECT_MAPPING.validate_python(encrypted.model_dump(), strict=True)
        data: Final = _database_data(dumped)
        if row is None:
            await transaction.litellm_credentialstable.create(
                data={**data, "created_by": attempt.actor, "updated_by": attempt.actor}
            )
        else:
            await transaction.litellm_credentialstable.update(
                where={"credential_name": attempt.credential_name},
                data={**data, "updated_by": attempt.actor},
            )
    CredentialAccessor.upsert_credentials([plaintext])
    await publish_config_change_for_object_type("litellm_credentialstable")
    return stored_tokens


@router.post(
    "/credentials/anthropic/oauth/start",
    response_model=AnthropicOAuthStartResponse,
    tags=["credential management"],
)
async def start_anthropic_oauth(
    payload: AnthropicOAuthStartRequest,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AnthropicOAuthStartResponse:
    _require_admin(user)
    if not _CREDENTIAL_NAME_PATTERN.fullmatch(payload.credential_name):
        raise HTTPException(
            status_code=422,
            detail="Credential name may contain only letters, numbers, dots, underscores, and hyphens",
        )
    authorization: Final = AnthropicOAuthClient().begin_authorization()
    attempt: Final = AnthropicOAuthAttempt(
        credential_name=payload.credential_name,
        actor=_actor(user),
        state=authorization.state,
        code_verifier=authorization.code_verifier,
        expires_at=time.time() + ANTHROPIC_ATTEMPT_LIFETIME_SECONDS,
    )
    return AnthropicOAuthStartResponse(
        authorization_url=authorization.authorization_url,
        attempt_token=_encode_attempt(attempt),
        expires_in_seconds=ANTHROPIC_ATTEMPT_LIFETIME_SECONDS,
    )


@router.post(
    "/credentials/anthropic/oauth/complete",
    response_model=AnthropicOAuthCompleteResponse,
    tags=["credential management"],
)
async def complete_anthropic_oauth(
    payload: AnthropicOAuthCompleteRequest,
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AnthropicOAuthCompleteResponse:
    _require_admin(user)
    attempt: Final = _decode_attempt(payload.attempt_token, _actor(user))
    code: Final = _parse_authorization_code(payload.authorization_code, attempt.state)
    await _complete_and_store(attempt, code, AnthropicOAuthClient())
    return AnthropicOAuthCompleteResponse(status="connected", credential_name=attempt.credential_name)


class AnthropicOAuthCredentialHook(CustomLogger):
    def __init__(self, oauth_client: AnthropicOAuthClient | None = None) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # inherited variadic kwargs are untyped
        self._oauth_client = oauth_client or AnthropicOAuthClient()
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, object],
        call_type: CallTypes | None,
    ) -> dict[str, object] | None:
        model: Final = kwargs.get("model")
        provider: Final = kwargs.get("custom_llm_provider")
        is_anthropic_model: Final = (
            provider == ANTHROPIC_CREDENTIAL_PROVIDER
            if provider is not None
            else isinstance(model, str) and model.startswith(("anthropic/", "claude-"))
        )
        credential_name: Final = kwargs.get("litellm_credential_name")
        if not isinstance(credential_name, str) or not credential_name:
            return None
        try:
            credential: Final = await self._find_or_load(credential_name)
        except Exception:  # noqa: BLE001  # credential loading errors must not disclose token bundles
            raise self._authentication_error(model, credential_name) from None
        if credential is None:
            if is_anthropic_model:
                raise self._authentication_error(model, credential_name)
            return None
        if not _is_managed_credential(credential):
            credential_values: Final = _string_object_mapping(cast(object, credential.credential_values)) or {}
            credential_info: Final = _string_object_mapping(cast(object, credential.credential_info)) or {}
            if ANTHROPIC_CREDENTIAL_VALUE_KEY in credential_values or (
                is_anthropic_model and credential_info.get("auth_type") == "oauth"
            ):
                raise self._authentication_error(model, credential_name)
            return None
        if not is_anthropic_model:
            raise self._authentication_error(model, credential_name)
        api_base: Final = kwargs.get("api_base")
        if isinstance(api_base, str) and api_base.rstrip("/") != "https://api.anthropic.com":
            raise self._authentication_error(model, credential_name)
        scoped: Final = kwargs.get("provider_specific_header")
        scoped_entries: Final = (
            (_object_mapping(cast(object, scoped)),)
            if isinstance(scoped, Mapping)
            else tuple(_object_mapping(entry) for entry in cast(list[object] | tuple[object, ...], scoped))
            if isinstance(scoped, (list, tuple))
            else ()
        )
        scoped_headers: Final = tuple(
            entry.get("extra_headers")
            for entry in scoped_entries
            if entry is not None
            and isinstance(entry.get("custom_llm_provider"), str)
            and ANTHROPIC_CREDENTIAL_PROVIDER in cast(str, entry["custom_llm_provider"]).replace(" ", "").split(",")
        )
        header_values: Final = (
            *(kwargs.get(name) for name in ("extra_headers", "headers", "additional_headers")),
            *scoped_headers,
        )
        header_mappings: Final = tuple(_object_mapping(headers) for headers in header_values)
        if any(
            headers is not None and any(str(name).lower() in {"authorization", "x-api-key", "host"} for name in headers)
            for headers in header_mappings
        ):
            raise self._authentication_error(model, credential_name)
        try:
            tokens: Final = await self._get_tokens(credential_name)
        except AnthropicOAuthError as error:
            if error.retryable:
                raise
            raise self._authentication_error(model, credential_name) from None
        except Exception:  # noqa: BLE001  # refresh failures must not disclose token bundles
            raise self._authentication_error(model, credential_name) from None
        return {**kwargs, "api_key": tokens.access_token, "api_base": "https://api.anthropic.com"}

    async def recover_rejected_token(self, credential_name: str, rejected_access_token: str) -> str | None:
        if not rejected_access_token.startswith("sk-ant-oat"):
            return None
        credential: Final = await self._find_or_load(credential_name)
        if credential is None or not _is_managed_credential(credential):
            return None
        cached: Final = _tokens_from_plaintext(credential)
        if cached is None:
            raise ValueError("Credential token bundle is invalid")
        identity: Final = cached.account_id or credential_name
        lock: Final = self._refresh_locks.setdefault(identity, asyncio.Lock())
        async with lock:
            recovered: Final = await self._refresh_tokens(
                credential_name,
                identity,
                rejected_access_token=rejected_access_token,
            )
        return recovered.access_token

    @staticmethod
    def _authentication_error(model: object, credential_name: str) -> AuthenticationError:
        return AuthenticationError(
            model=model if isinstance(model, str) else "anthropic",
            llm_provider=ANTHROPIC_CREDENTIAL_PROVIDER,
            message=f"Anthropic credential '{credential_name}' is unavailable",
        )

    async def _find_or_load(self, credential_name: str) -> CredentialItem | None:
        cached: Final = _find_cached(credential_name)
        if cached is not None:
            return cached
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return None
        row: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
        if row is None or not _is_managed_credential(row):
            return row
        tokens: Final = _tokens_from_row(row)
        plaintext: Final = _plaintext_credential(credential_name, tokens)
        CredentialAccessor.upsert_credentials([plaintext])
        return plaintext

    async def _get_tokens(self, credential_name: str) -> AnthropicOAuthTokens:
        cached: Final = _find_cached(credential_name)
        tokens: Final = _tokens_from_plaintext(cached) if cached is not None else None
        if tokens is None:
            raise ValueError("Credential token bundle is invalid")
        if self._is_fresh(tokens):
            return tokens
        identity: Final = tokens.account_id or credential_name
        lock: Final = self._refresh_locks.setdefault(identity, asyncio.Lock())
        async with lock:
            rechecked_credential: Final = _find_cached(credential_name)
            rechecked: Final = (
                _tokens_from_plaintext(rechecked_credential) if rechecked_credential is not None else None
            )
            if rechecked is not None and self._is_fresh(rechecked):
                return rechecked
            return await self._refresh_tokens(credential_name, identity)

    @staticmethod
    def _is_fresh(tokens: AnthropicOAuthTokens) -> bool:
        return time.time() < tokens.expires_at - ANTHROPIC_EXPIRY_SKEW_SECONDS

    async def _refresh_tokens(
        self,
        credential_name: str,
        identity: str,
        *,
        rejected_access_token: str | None = None,
    ) -> AnthropicOAuthTokens:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise RuntimeError("Database not connected")
        database: Final = cast(_PrismaClient, prisma_client).db
        async with database.tx(timeout=ANTHROPIC_REFRESH_TRANSACTION_TIMEOUT) as transaction:
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", _lock_key(credential_name))
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", _lock_key(identity, "account"))
            row: Final = await transaction.litellm_credentialstable.find_unique(
                where={"credential_name": credential_name}
            )
            if row is None:
                raise ValueError("Credential not found")
            info: Final = _credential_info_from_row(row)
            if (
                info.get("provider") != ANTHROPIC_CREDENTIAL_PROVIDER
                or info.get("auth_type") != ANTHROPIC_CREDENTIAL_AUTH_TYPE
            ):
                raise ValueError("Credential authentication type changed")
            stored: Final = _tokens_from_row(row)
            if (stored.account_id or credential_name) != identity:
                raise ValueError("Anthropic account identity changed before refresh")
            token_was_rotated: Final = (
                rejected_access_token is not None and stored.access_token != rejected_access_token
            )
            should_refresh: Final = not self._is_fresh(stored) or (
                rejected_access_token is not None and not token_was_rotated
            )
            refreshed: Final = await self._oauth_client.refresh(stored) if should_refresh else stored
            if should_refresh:
                if stored.account_id is not None and refreshed.account_id != stored.account_id:
                    raise ValueError("Anthropic account identity changed during refresh")
                encrypted: Final = encrypt_value_helper(refreshed.to_json())
                if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime crypto boundary
                    encrypted, str
                ):
                    raise ValueError("Credential token bundle cannot be encrypted")
                values: Final = _string_object_mapping(row.credential_values)
                if values is None:
                    raise TypeError("Credential values are invalid")
                await transaction.litellm_credentialstable.update(
                    where={"credential_name": credential_name},
                    data=_database_data(
                        {
                            "credential_values": {**values, ANTHROPIC_CREDENTIAL_VALUE_KEY: encrypted},
                            "credential_info": {
                                **info,
                                "provider": ANTHROPIC_CREDENTIAL_PROVIDER,
                                "auth_type": ANTHROPIC_CREDENTIAL_AUTH_TYPE,
                            },
                            "updated_by": "litellm-anthropic-oauth",
                        }
                    ),
                )
        CredentialAccessor.upsert_credentials([_plaintext_credential(credential_name, refreshed)])
        await publish_config_change_for_object_type("litellm_credentialstable")
        return refreshed


def register_anthropic_oauth_credential_hook() -> None:
    callbacks: Final = cast(list[object], litellm.callbacks)
    if any(isinstance(callback, AnthropicOAuthCredentialHook) for callback in callbacks):
        return
    callbacks.append(AnthropicOAuthCredentialHook())
