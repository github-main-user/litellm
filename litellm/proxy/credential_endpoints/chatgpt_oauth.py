import asyncio
import hashlib
import json
import re
import time
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm.exceptions import AuthenticationError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_DEVICE_VERIFY_URL,
    ChatGPTAuthError,
)
from litellm.llms.chatgpt.oauth_client import (
    ChatGPTDeviceCode,
    ChatGPTOAuthClient,
    ChatGPTTokens,
)
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.credential_endpoints.endpoints import CredentialHelperUtils
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import CallTypes

CHATGPT_CREDENTIAL_VALUE_KEY: Final = "litellm_internal_chatgpt_auth_token"
CHATGPT_CREDENTIAL_PROVIDER: Final = "chatgpt"
CHATGPT_CREDENTIAL_AUTH_TYPE: Final = "oauth"
CHATGPT_TOKEN_EXPIRY_SKEW_SECONDS: Final = 60
CHATGPT_DEVICE_CODE_LIFETIME_SECONDS: Final = 15 * 60
_CREDENTIAL_NAME_PATTERN: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

router: Final = APIRouter()


class ChatGPTOAuthStartRequest(BaseModel):
    credential_name: str | None = Field(default=None, min_length=1, max_length=255)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class ChatGPTOAuthPollRequest(BaseModel):
    attempt_token: str = Field(min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class ChatGPTOAuthStartResponse(BaseModel):
    verification_url: str
    user_code: str
    attempt_token: str
    interval_seconds: int
    expires_in_seconds: int


class ChatGPTOAuthPollResponse(BaseModel):
    status: Literal["pending", "connected"]
    credential_name: str | None = None
    interval_seconds: int | None = None


class ChatGPTOAuthAttempt(BaseModel):
    credential_name: str | None
    device_auth_id: str
    user_code: str
    interval_seconds: int
    expires_at: float


def _validate_credential_name(credential_name: str | None) -> str | None:
    if credential_name is None:
        return None
    if not _CREDENTIAL_NAME_PATTERN.fullmatch(credential_name):
        raise HTTPException(
            status_code=422,
            detail="Credential name may contain only letters, numbers, dots, underscores, and hyphens",
        )
    return credential_name


def _encode_attempt(attempt: ChatGPTOAuthAttempt) -> str:
    encrypted: Final = encrypt_value_helper(attempt.model_dump_json())
    if not isinstance(encrypted, str):
        raise HTTPException(status_code=500, detail="Unable to protect ChatGPT authentication attempt")
    return encrypted


def _decode_attempt(attempt_token: str) -> ChatGPTOAuthAttempt:
    decrypted: Final = decrypt_value_helper(
        attempt_token,
        CHATGPT_CREDENTIAL_VALUE_KEY,
        exception_type="debug",
    )
    if not isinstance(decrypted, str):
        raise HTTPException(status_code=400, detail="Invalid ChatGPT authentication attempt")
    try:
        attempt: Final = ChatGPTOAuthAttempt.model_validate_json(decrypted)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid ChatGPT authentication attempt") from error
    if time.time() >= attempt.expires_at:
        raise HTTPException(status_code=400, detail="ChatGPT authentication attempt expired")
    return attempt


def _tokens_from_credential(credential: CredentialItem) -> ChatGPTTokens | None:
    value: Final = credential.credential_values.get(CHATGPT_CREDENTIAL_VALUE_KEY)
    if not isinstance(value, str):
        return None
    try:
        return ChatGPTTokens.from_json(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _tokens_from_encrypted_credential(credential: CredentialItem) -> ChatGPTTokens | None:
    encrypted_value: Final = credential.credential_values.get(CHATGPT_CREDENTIAL_VALUE_KEY)
    if not isinstance(encrypted_value, str):
        return None
    decrypted_value: Final = decrypt_value_helper(
        encrypted_value,
        CHATGPT_CREDENTIAL_VALUE_KEY,
        exception_type="debug",
    )
    if not isinstance(decrypted_value, str):
        return None
    try:
        return ChatGPTTokens.from_json(decrypted_value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _is_chatgpt_oauth_credential(credential: CredentialItem) -> bool:
    return (
        credential.credential_info.get("provider") == CHATGPT_CREDENTIAL_PROVIDER
        and credential.credential_info.get("auth_type") == CHATGPT_CREDENTIAL_AUTH_TYPE
    )


def _decrypt_credential(credential: CredentialItem) -> CredentialItem:
    return CredentialItem(
        credential_name=credential.credential_name,
        credential_info=credential.credential_info,
        credential_values={
            key: decrypt_value_helper(value=value, key=key) or value
            for key, value in credential.credential_values.items()
        },
    )


def _find_credential(credential_name: str) -> CredentialItem | None:
    return next(
        (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
        None,
    )


def _find_credential_name_for_account(account_id: str) -> str | None:
    for credential in litellm.credential_list:
        if credential.credential_info.get("provider") != CHATGPT_CREDENTIAL_PROVIDER:
            continue
        tokens: Final = _tokens_from_credential(credential)
        if tokens is not None and tokens.account_id == account_id:
            return credential.credential_name
    return None


def _default_credential_name(account_id: str) -> str:
    normalized: Final = re.sub(r"[^a-zA-Z0-9._-]+", "-", account_id).strip("-._")
    if not normalized:
        raise HTTPException(status_code=400, detail="ChatGPT account ID cannot be used as a credential name")
    return f"chatgpt-{normalized}"[:255]


def _resolved_credential_name(requested_name: str | None, tokens: ChatGPTTokens) -> str:
    if requested_name is not None:
        return requested_name
    if tokens.account_id is None:
        raise HTTPException(status_code=400, detail="ChatGPT account ID is missing; provide a credential name")
    existing_name: Final = _find_credential_name_for_account(tokens.account_id)
    return existing_name or _default_credential_name(tokens.account_id)


async def _store_tokens(
    credential_name: str,
    tokens: ChatGPTTokens,
    actor: str,
) -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    existing: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
    if existing is not None and existing.credential_info.get("provider") != CHATGPT_CREDENTIAL_PROVIDER:
        raise HTTPException(status_code=409, detail="Credential name is already used by another provider")
    if existing is not None:
        in_memory_existing: Final = _find_credential(credential_name)
        existing_tokens: Final = (
            _tokens_from_credential(in_memory_existing)
            if in_memory_existing is not None
            else _tokens_from_encrypted_credential(existing)
        )
        if (
            existing_tokens is not None
            and existing_tokens.account_id is not None
            and tokens.account_id is not None
            and existing_tokens.account_id != tokens.account_id
        ):
            raise HTTPException(status_code=409, detail="Credential name belongs to another ChatGPT account")
    plaintext: Final = CredentialItem(
        credential_name=credential_name,
        credential_info={
            "provider": CHATGPT_CREDENTIAL_PROVIDER,
            "auth_type": CHATGPT_CREDENTIAL_AUTH_TYPE,
        },
        credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: tokens.to_json()},
    )
    encrypted: Final = CredentialHelperUtils.encrypt_credential_values(plaintext)
    data: Final = encrypted.model_dump()
    repository: Final = CredentialsRepository(prisma_client)
    if existing is None:
        await repository.create(data={**data, "created_by": actor, "updated_by": actor})
    else:
        await repository.update_by_name(
            credential_name,
            data={**data, "updated_by": actor},
        )
    CredentialAccessor.upsert_credentials([plaintext])


@router.post(
    "/credentials/chatgpt/oauth/start",
    dependencies=[Depends(user_api_key_auth)],
    response_model=ChatGPTOAuthStartResponse,
    tags=["credential management"],
)
async def start_chatgpt_oauth(
    payload: ChatGPTOAuthStartRequest,
) -> ChatGPTOAuthStartResponse:
    credential_name: Final = _validate_credential_name(payload.credential_name)
    try:
        device_code: Final = await asyncio.to_thread(ChatGPTOAuthClient().request_device_code)
    except ChatGPTAuthError as error:
        raise HTTPException(status_code=error.status_code or 400, detail=str(error)) from error
    attempt: Final = ChatGPTOAuthAttempt(
        credential_name=credential_name,
        device_auth_id=device_code.device_auth_id,
        user_code=device_code.user_code,
        interval_seconds=device_code.interval_seconds,
        expires_at=time.time() + CHATGPT_DEVICE_CODE_LIFETIME_SECONDS,
    )
    return ChatGPTOAuthStartResponse(
        verification_url=CHATGPT_DEVICE_VERIFY_URL,
        user_code=device_code.user_code,
        attempt_token=_encode_attempt(attempt),
        interval_seconds=device_code.interval_seconds,
        expires_in_seconds=CHATGPT_DEVICE_CODE_LIFETIME_SECONDS,
    )


@router.post(
    "/credentials/chatgpt/oauth/poll",
    dependencies=[Depends(user_api_key_auth)],
    response_model=ChatGPTOAuthPollResponse,
    tags=["credential management"],
)
async def poll_chatgpt_oauth(
    payload: ChatGPTOAuthPollRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ChatGPTOAuthPollResponse:
    attempt: Final = _decode_attempt(payload.attempt_token)
    device_code: Final = ChatGPTDeviceCode(
        device_auth_id=attempt.device_auth_id,
        user_code=attempt.user_code,
        interval_seconds=attempt.interval_seconds,
    )
    oauth_client: Final = ChatGPTOAuthClient()
    try:
        authorization: Final = await asyncio.to_thread(oauth_client.poll_authorization, device_code)
        if authorization is None:
            return ChatGPTOAuthPollResponse(
                status="pending",
                interval_seconds=attempt.interval_seconds,
            )
        tokens: Final = await asyncio.to_thread(oauth_client.exchange_code, authorization)
    except ChatGPTAuthError as error:
        raise HTTPException(status_code=error.status_code or 400, detail=str(error)) from error
    credential_name: Final = _resolved_credential_name(attempt.credential_name, tokens)
    actor: Final = str(user_api_key_dict.user_id or "litellm-proxy-admin")
    await _store_tokens(credential_name=credential_name, tokens=tokens, actor=actor)
    return ChatGPTOAuthPollResponse(status="connected", credential_name=credential_name)


class ChatGPTOAuthCredentialHook(CustomLogger):
    def __init__(self, oauth_client: ChatGPTOAuthClient | None = None) -> None:
        super().__init__()
        self._oauth_client = oauth_client or ChatGPTOAuthClient()
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, Any],
        call_type: CallTypes | None,
    ) -> dict[str, Any] | None:
        model: Final = kwargs.get("model")
        provider: Final = kwargs.get("custom_llm_provider")
        if provider != CHATGPT_CREDENTIAL_PROVIDER and not (
            isinstance(model, str) and model.startswith(f"{CHATGPT_CREDENTIAL_PROVIDER}/")
        ):
            return None
        credential_name: Final = kwargs.get("litellm_credential_name")
        if not isinstance(credential_name, str) or not credential_name:
            return None
        credential: Final = await self._find_or_load_credential(credential_name)
        if credential is None or not _is_chatgpt_oauth_credential(credential):
            return None
        try:
            tokens: Final = await self._get_tokens(credential_name)
        except Exception as error:
            raise AuthenticationError(
                model=model if isinstance(model, str) else "chatgpt",
                llm_provider=CHATGPT_CREDENTIAL_PROVIDER,
                message=f"ChatGPT credential '{credential_name}' is unavailable",
            ) from error
        return {
            **kwargs,
            "api_key": tokens.access_token,
            "chatgpt_auth_account_id": tokens.account_id,
        }

    async def _find_or_load_credential(self, credential_name: str) -> CredentialItem | None:
        cached: Final = _find_credential(credential_name)
        if cached is not None:
            return cached
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return None
        stored: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
        if stored is None:
            return None
        credential: Final = _decrypt_credential(stored)
        CredentialAccessor.upsert_credentials([credential])
        return credential

    async def _get_tokens(self, credential_name: str) -> ChatGPTTokens:
        cached: Final = self._read_cached_tokens(credential_name)
        if self._is_fresh(cached):
            return cached
        lock: Final = self._refresh_locks.setdefault(credential_name, asyncio.Lock())
        async with lock:
            rechecked: Final = self._read_cached_tokens(credential_name)
            if self._is_fresh(rechecked):
                return rechecked
            return await self._refresh_tokens(credential_name)

    def _read_cached_tokens(self, credential_name: str) -> ChatGPTTokens:
        credential: Final = _find_credential(credential_name)
        if credential is None:
            raise ValueError("Credential not found")
        tokens: Final = _tokens_from_credential(credential)
        if tokens is None:
            raise ValueError("Credential does not contain ChatGPT OAuth tokens")
        return tokens

    @staticmethod
    def _is_fresh(tokens: ChatGPTTokens) -> bool:
        return tokens.expires_at is not None and time.time() < tokens.expires_at - CHATGPT_TOKEN_EXPIRY_SKEW_SECONDS

    async def _refresh_tokens(self, credential_name: str) -> ChatGPTTokens:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise RuntimeError("Database not connected")
        lock_key: Final = int.from_bytes(
            hashlib.blake2b(credential_name.encode(), digest_size=8).digest(),
            "big",
            signed=True,
        )
        refreshed: ChatGPTTokens
        credential_info: dict[str, Any]
        async with prisma_client.db.tx() as transaction:
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
            row: Final = await transaction.litellm_credentialstable.find_unique(
                where={"credential_name": credential_name}
            )
            if row is None:
                raise ValueError("Credential not found")
            encrypted_value: Final = row.credential_values.get(CHATGPT_CREDENTIAL_VALUE_KEY)
            decrypted_value: Final = (
                decrypt_value_helper(encrypted_value, CHATGPT_CREDENTIAL_VALUE_KEY)
                if isinstance(encrypted_value, str)
                else None
            )
            if not isinstance(decrypted_value, str):
                raise TypeError("Credential token bundle cannot be decrypted")
            stored: Final = ChatGPTTokens.from_json(decrypted_value)
            if self._is_fresh(stored):
                refreshed = stored
            else:
                refreshed = await asyncio.to_thread(self._oauth_client.refresh, stored.refresh_token)
                encrypted_bundle: Final = encrypt_value_helper(refreshed.to_json())
                if not isinstance(encrypted_bundle, str):
                    raise ValueError("Credential token bundle cannot be encrypted")
                await transaction.litellm_credentialstable.update(
                    where={"credential_name": credential_name},
                    data={
                        "credential_values": {
                            **row.credential_values,
                            CHATGPT_CREDENTIAL_VALUE_KEY: encrypted_bundle,
                        },
                        "credential_info": {
                            **(row.credential_info or {}),
                            "provider": CHATGPT_CREDENTIAL_PROVIDER,
                            "auth_type": CHATGPT_CREDENTIAL_AUTH_TYPE,
                        },
                        "updated_by": "litellm-chatgpt-oauth",
                    },
                )
            credential_info = {
                **(row.credential_info or {}),
                "provider": CHATGPT_CREDENTIAL_PROVIDER,
                "auth_type": CHATGPT_CREDENTIAL_AUTH_TYPE,
            }
        await publish_config_change_for_object_type("litellm_credentialstable")
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name=credential_name,
                    credential_info=credential_info,
                    credential_values={CHATGPT_CREDENTIAL_VALUE_KEY: refreshed.to_json()},
                )
            ]
        )
        return refreshed


def register_chatgpt_oauth_credential_hook() -> None:
    if any(isinstance(callback, ChatGPTOAuthCredentialHook) for callback in litellm.callbacks):
        return
    litellm.callbacks.append(ChatGPTOAuthCredentialHook())
