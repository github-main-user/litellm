import os
import secrets
import time
from datetime import timedelta
from typing import Annotated, Final, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, SecretStr, ValidationError

from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.kling import oauth
from litellm.llms.kling.videos.client import KlingVideoClient
from litellm.models.credentials import CredentialItem
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.management_endpoints.canonical_model_endpoints import _admin, _client
from litellm.proxy.utils import jsonify_object

router: Final = APIRouter(prefix="/credentials/kling", tags=["credential management"])
Admin = Annotated[UserAPIKeyAuth, Depends(_admin)]
KEY: Final = "litellm_internal_kling_oauth"


class Start(BaseModel):
    reconnect: bool = False
    credential_name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")


class Attempt(Start):
    previous_id: str | None = None
    client_id: str
    redirect_uri: str
    verifier: str = Field(repr=False)
    state: str = Field(repr=False)
    deadline: float
    actor: str


class Started(BaseModel):
    authorization_url: str
    attempt_token: str = Field(repr=False)


class Complete(BaseModel):
    attempt_token: SecretStr = Field(max_length=65536)
    code: SecretStr = Field(min_length=1, max_length=8192)
    state: SecretStr = Field(min_length=1, max_length=512)


class Connected(BaseModel):
    status: Literal["connected"] = "connected"
    credential_name: str


def _callback() -> str:
    uri: Final = os.environ.get("KLING_OAUTH_REDIRECT_URI", "https://admin.void-api.tech/providers")
    parts: Final = urlsplit(uri)
    if (
        not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or not (parts.scheme == "https" or parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1"))
    ):
        raise HTTPException(503, "Invalid Kling OAuth callback configuration")
    return uri


@router.post("/oauth/start")
async def start(body: Start, user: Admin) -> Started:
    db: Final = _client().db
    existing = await db.litellm_credentialstable.find_unique(where={"credential_name": body.credential_name})
    if body.reconnect and (
        not existing
        or (existing.credential_info or {}).get("provider") != "kling"
        or KEY not in existing.credential_values
    ):
        raise HTTPException(409, "Cannot reconnect this credential")
    if existing and not body.reconnect:
        raise HTTPException(409, "Provider connection name is already in use")
    uri: Final = _callback()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            client_id: Final = await oauth.register(client, uri)
    except (ValueError, httpx.HTTPError):
        raise HTTPException(502, "Kling client registration failed") from None
    flow: Final = oauth.authorize(client_id, uri)
    attempt: Final = Attempt(
        credential_name=body.credential_name,
        reconnect=body.reconnect,
        previous_id=existing.credential_id if existing else None,
        client_id=client_id,
        redirect_uri=uri,
        verifier=flow.verifier,
        state=flow.state,
        deadline=time.time() + 600,
        actor=str(user.user_id),
    )
    return Started(authorization_url=flow.url, attempt_token=encrypt_value_helper(attempt.model_dump_json()))


@router.post("/oauth/complete")
async def complete(body: Complete, user: Admin) -> Connected:
    try:
        attempt: Final = Attempt.model_validate_json(
            decrypt_value_helper(body.attempt_token.get_secret_value(), KEY, exception_type="debug") or ""
        )
    except (ValidationError, ValueError):
        raise HTTPException(400, "Invalid Kling authorization attempt") from None
    if (
        attempt.deadline <= time.time()
        or attempt.actor != str(user.user_id)
        or not secrets.compare_digest(attempt.state.encode(), body.state.get_secret_value().encode())
    ):
        raise HTTPException(400, "Invalid or expired Kling authorization attempt")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            tokens: Final = await oauth.exchange(
                client, attempt.client_id, body.code.get_secret_value(), attempt.verifier, attempt.redirect_uri
            )
    except (ValueError, httpx.HTTPError):
        raise HTTPException(502, "Kling authorization failed; restart sign-in") from None
    db: Final = _client().db
    async with db.tx() as tx:
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))::text AS lock", attempt.credential_name
        )
        row = await tx.litellm_credentialstable.find_unique(where={"credential_name": attempt.credential_name})
        if attempt.reconnect:
            if not row or row.credential_id != attempt.previous_id or KEY not in row.credential_values:
                raise HTTPException(409, "Connection changed during authorization")
            await tx.litellm_credentialstable.update(
                where={"credential_name": attempt.credential_name},
                data=jsonify_object(
                    {
                        "credential_values": {KEY: encrypt_value_helper(tokens.model_dump_json())},
                        "updated_by": attempt.actor,
                    }
                ),
            )
        elif row:
            raise HTTPException(409, "Provider connection name is already in use")
        else:
            await tx.litellm_credentialstable.create(
                data=jsonify_object(
                    {
                        "credential_name": attempt.credential_name,
                        "credential_info": {"provider": "kling", "auth_type": "oauth"},
                        "credential_values": {KEY: encrypt_value_helper(tokens.model_dump_json())},
                        "created_by": attempt.actor,
                        "updated_by": attempt.actor,
                    }
                )
            )
    from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type

    CredentialAccessor.upsert_credentials(
        [
            CredentialItem(
                credential_name=attempt.credential_name,
                credential_info={"provider": "kling", "auth_type": "oauth"},
                credential_values={},
            )
        ]
    )
    await publish_config_change_for_object_type("litellm_credentialstable")
    return Connected(credential_name=attempt.credential_name)


async def video_client(name: str) -> KlingVideoClient:
    async with _client().db.tx(timeout=timedelta(seconds=25)) as tx:
        rows: Final = await tx.query_raw(
            "SELECT credential_values FROM \"LiteLLM_CredentialsTable\" WHERE credential_name=$1 AND credential_info->>'provider'='kling' AND credential_info->>'auth_type'='oauth' FOR UPDATE",
            name,
        )
        if not rows:
            raise HTTPException(404, "Kling connection not found")
        bundle: Final = oauth.TokenBundle.model_validate_json(
            decrypt_value_helper(rows[0]["credential_values"][KEY], KEY, exception_type="debug") or ""
        )
        if bundle.expires_at > time.time() + 60:
            return KlingVideoClient(SecretStr(bundle.access_token))
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                updated: Final = await oauth.refresh(client, bundle)
        except (ValueError, httpx.HTTPError):
            raise HTTPException(502, "Kling token refresh failed") from None
        await tx.litellm_credentialstable.update(
            where={"credential_name": name},
            data=jsonify_object({"credential_values": {KEY: encrypt_value_helper(updated.model_dump_json())}}),
        )
        return KlingVideoClient(SecretStr(updated.access_token))


@router.get("/{name}/models")
async def list_models(name: str, user: Admin) -> list[str]:
    client: Final = await video_client(name)
    capabilities: Final = await client.capabilities()
    return sorted({model.model for model in capabilities.text_to_video()})
