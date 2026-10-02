import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import litellm
from litellm._uuid import uuid
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type
from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient
from litellm.repositories.base_repository import is_unique_violation
from litellm.repositories.model_repository import ModelRepository

from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.proxy.management_endpoints.model_resolution import ModelResolution, ResolutionRequest, resolve_identity

router: Final = APIRouter(tags=["canonical models"])


class ConnectionWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str | None = Field(default=None, min_length=1, max_length=255)
    provider_connection: str | None = Field(default=None, min_length=1, max_length=255)
    provider_model: str = Field(min_length=1, max_length=255)


class CanonicalWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=255, pattern=r"^[^*]+$")
    group: str | None = Field(default=None, min_length=1, max_length=100)
    base_model: str = Field(
        min_length=3, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    input_price_per_million_tokens: Decimal = Field(gt=0, max_digits=24, decimal_places=12)
    output_price_per_million_tokens: Decimal = Field(gt=0, max_digits=24, decimal_places=12)
    cache_read_price_per_million_tokens: Decimal | None = Field(default=None, gt=0, max_digits=24, decimal_places=12)
    cache_write_price_per_million_tokens: Decimal | None = Field(default=None, gt=0, max_digits=24, decimal_places=12)
    connections: list[ConnectionWrite]

    @model_validator(mode="after")
    def unique_connections(self) -> "CanonicalWrite":
        ids: Final = [connection.id for connection in self.connections if connection.id is not None]
        bindings: Final = [
            (connection.provider_connection, connection.provider_model) for connection in self.connections
        ]
        if len(ids) != len(set(ids)) or len(bindings) != len(set(bindings)):
            raise ValueError("Model connections must be unique")
        return self


class ConnectionRead(BaseModel):
    id: str
    provider_connection: str | None
    provider_model: str
    mode: Literal["chat", "responses"]


class CanonicalRead(BaseModel):
    id: str
    name: str
    group: str | None
    base_model: str
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    cache_read_price_per_million_tokens: Decimal
    cache_write_price_per_million_tokens: Decimal
    connections: list[ConnectionRead]


@dataclass(frozen=True, slots=True)
class _TxClient:
    db: object


async def reject_canonical_deployment(client: object, deployment_id: str) -> None:
    connection: Final = await WriterPinnedClient(client.db).db.litellm_canonicalmodelconnection.find_unique(
        where={"deployment_id": deployment_id}
    )
    if connection is not None:
        raise _error(409, "canonical_managed", "Manage this deployment through /canonical/models")


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _client():
    from litellm.proxy.proxy_server import prisma_client, store_model_in_db

    if prisma_client is None or not store_model_in_db:
        raise _error(503, "database_unavailable", "Database-backed models are not enabled")
    return prisma_client


def _admin(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    if user.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise _error(403, "forbidden", "Only proxy admins can manage canonical models")
    return user


def _read(row: object) -> CanonicalRead:
    return CanonicalRead(
        id=row.id,
        name=row.name,
        group=row.group,
        base_model=row.base_model,
        input_price_per_million_tokens=row.input_price_per_million_tokens,
        output_price_per_million_tokens=row.output_price_per_million_tokens,
        cache_read_price_per_million_tokens=row.cache_read_price_per_million_tokens,
        cache_write_price_per_million_tokens=row.cache_write_price_per_million_tokens,
        connections=[
            ConnectionRead(
                id=connection.id,
                provider_connection=connection.provider_connection,
                provider_model=connection.provider_model,
                mode=connection.mode,
            )
            for connection in sorted(row.connections, key=lambda connection: (connection.position, connection.id))
        ],
    )


def _parse_body(raw: object) -> CanonicalWrite:
    try:
        return CanonicalWrite.model_validate(raw)
    except ValidationError as exc:
        raise _error(422, "invalid_request", str(exc)) from exc


def _fields(body: CanonicalWrite, actor: str) -> dict[str, object]:
    return {
        "name": body.name,
        "group": body.group,
        "base_model": body.base_model,
        "input_price_per_million_tokens": body.input_price_per_million_tokens,
        "output_price_per_million_tokens": body.output_price_per_million_tokens,
        "cache_read_price_per_million_tokens": body.cache_read_price_per_million_tokens
        or body.input_price_per_million_tokens,
        "cache_write_price_per_million_tokens": body.cache_write_price_per_million_tokens
        or body.input_price_per_million_tokens,
        "updated_by": actor,
    }


def _mode(provider_model: str) -> Literal["chat", "responses"]:
    if provider_model.startswith("chatgpt/"):
        return "responses"
    direct: Final = litellm.model_cost.get(provider_model)
    provider, _, model_name = provider_model.partition("/")
    candidate: Final = litellm.model_cost.get(model_name) if model_name else None
    metadata: Final = direct or (
        candidate if candidate is not None and candidate.get("litellm_provider") == provider else None
    )
    mode: Final = metadata.get("mode") if metadata is not None else None
    if mode is None:
        return "chat"
    if mode in ("chat", "responses"):
        return mode
    raise _error(400, "llm_model_mode_unsupported", "Provider model does not support chat or responses")


def _model_info(body: CanonicalWrite, deployment_id: str, mode: str) -> dict[str, object]:
    return {
        "id": deployment_id,
        "mode": mode,
        "input_cost_per_token": float(body.input_price_per_million_tokens / 1_000_000),
        "output_cost_per_token": float(body.output_price_per_million_tokens / 1_000_000),
        "cache_read_input_token_cost": float(
            (body.cache_read_price_per_million_tokens or body.input_price_per_million_tokens) / 1_000_000
        ),
        "cache_creation_input_token_cost": float(
            (body.cache_write_price_per_million_tokens or body.input_price_per_million_tokens) / 1_000_000
        ),
        "cache_creation_input_token_cost_above_1hr": float(
            (body.cache_write_price_per_million_tokens or body.input_price_per_million_tokens) / 1_000_000
        ),
        "cache_creation_input_token_cost_above_200k_tokens": float(
            (body.cache_write_price_per_million_tokens or body.input_price_per_million_tokens) / 1_000_000
        ),
        "cache_read_input_token_cost_above_200k_tokens": float(
            (body.cache_read_price_per_million_tokens or body.input_price_per_million_tokens) / 1_000_000
        ),
        "group": body.group,
    }


async def _validate_credential(tx: object, connection: ConnectionWrite) -> None:
    if connection.provider_connection is None:
        return
    locked: Final = await tx.query_raw(
        'SELECT credential_id FROM "LiteLLM_CredentialsTable" WHERE credential_name = $1 FOR SHARE',
        connection.provider_connection,
    )
    if not locked:
        raise _error(404, "provider_connection_not_found", "Provider connection not found")
    credential: Final = await tx.litellm_credentialstable.find_unique(
        where={"credential_name": connection.provider_connection}
    )
    if credential is None:
        raise _error(404, "provider_connection_not_found", "Provider connection not found")
    info: Final = credential.credential_info
    provider: Final = info.get("provider") if isinstance(info, dict) else None
    model_provider: Final = connection.provider_model.partition("/")[0]
    is_anthropic: Final = model_provider == "anthropic" or (
        "/" not in connection.provider_model and connection.provider_model.startswith("claude-")
    )
    expected_provider: Final = litellm.model_cost.get(connection.provider_model, {}).get("litellm_provider")
    if (
        (provider == "anthropic" and not is_anthropic)
        or (provider == "chatgpt" and model_provider != "chatgpt")
        or (provider not in (None, "anthropic", "chatgpt") and provider != (expected_provider or model_provider))
    ):
        raise _error(400, "provider_connection_provider_mismatch", "Provider connection does not match model")


def _deployment_params(body: CanonicalWrite, connection: ConnectionWrite) -> dict[str, object]:
    million: Final = Decimal(1_000_000)
    read_cost: Final = float(
        (body.cache_read_price_per_million_tokens or body.input_price_per_million_tokens) / million
    )
    write_cost: Final = float(
        (body.cache_write_price_per_million_tokens or body.input_price_per_million_tokens) / million
    )
    params: Final[dict[str, object]] = {
        "model": connection.provider_model,
        "input_cost_per_token": float(body.input_price_per_million_tokens / million),
        "output_cost_per_token": float(body.output_price_per_million_tokens / million),
        "cache_read_input_token_cost": read_cost,
        "cache_creation_input_token_cost": write_cost,
        "cache_creation_input_token_cost_above_1hr": write_cost,
        "cache_creation_input_token_cost_above_200k_tokens": write_cost,
        "cache_read_input_token_cost_above_200k_tokens": read_cost,
    }
    if connection.provider_connection is not None:
        params["litellm_credential_name"] = connection.provider_connection
    return params


async def _connection_create(
    tx: object,
    repo: ModelRepository,
    canonical_id: str,
    body: CanonicalWrite,
    connection: ConnectionWrite,
    actor: str,
    position: int,
) -> None:
    await _validate_credential(tx, connection)
    deployment_id: Final = str(uuid.uuid4())
    mode: Final = _mode(connection.provider_model)
    await repo.create_model(
        model_id=deployment_id,
        model_name=body.name,
        litellm_params=_deployment_params(body, connection),
        model_info=_model_info(body, deployment_id, mode),
        created_by=actor,
    )
    await tx.litellm_canonicalmodelconnection.create(
        data={
            "id": connection.id or str(uuid.uuid4()),
            "canonical_model_id": canonical_id,
            "provider_connection": connection.provider_connection,
            "provider_model": connection.provider_model,
            "mode": mode,
            "deployment_id": deployment_id,
            "position": position,
        }
    )


async def _connection_update(
    tx: object,
    repo: ModelRepository,
    row: object,
    body: CanonicalWrite,
    connection: ConnectionWrite,
    actor: str,
    position: int,
) -> None:
    await _validate_credential(tx, connection)
    mode: Final = (
        row.mode
        if (row.provider_connection, row.provider_model) == (connection.provider_connection, connection.provider_model)
        else _mode(connection.provider_model)
    )
    await repo.update_model(
        model_id=row.deployment_id,
        updated_by=actor,
        model_name=body.name,
        litellm_params=_deployment_params(body, connection),
        model_info=_model_info(body, row.deployment_id, mode),
    )
    await tx.litellm_canonicalmodelconnection.update(
        where={"id": row.id},
        data={
            "provider_connection": connection.provider_connection,
            "provider_model": connection.provider_model,
            "mode": mode,
            "position": position,
        },
    )


async def _refresh() -> None:
    from litellm.proxy.management_endpoints.model_management_endpoints import clear_cache
    from litellm.proxy.proxy_server import llm_router

    await publish_config_change_for_object_type("litellm_proxymodeltable")
    outcome: Final = await clear_cache()
    if llm_router is not None and outcome.still_desired is None:
        raise _error(503, "reload_failed", "Saved to the database, but the local model router did not reload")


async def _load(client: object, model_id: str) -> CanonicalRead:
    row: Final = await WriterPinnedClient(client.db).db.litellm_canonicalmodel.find_unique(
        where={"id": model_id}, include={"connections": True}
    )
    if row is None:
        raise _error(404, "not_found", "Canonical model not found")
    return _read(row)


@router.get("/canonical/models", response_model=list[CanonicalRead])
async def list_canonical_models(user: Annotated[UserAPIKeyAuth, Depends(_admin)]) -> list[CanonicalRead]:
    client: Final = _client()
    rows: Final = await WriterPinnedClient(client.db).db.litellm_canonicalmodel.find_many(include={"connections": True})
    return [_read(row) for row in sorted(rows, key=lambda row: row.name)]


@router.get("/canonical/models/{model_id}", response_model=CanonicalRead)
async def get_canonical_model(model_id: str, user: Annotated[UserAPIKeyAuth, Depends(_admin)]) -> CanonicalRead:
    return await _load(_client(), model_id)


@router.post("/canonical/models", response_model=CanonicalRead, status_code=201)
async def create_canonical_model(
    user: Annotated[UserAPIKeyAuth, Depends(_admin)], raw: Annotated[object, Body()] = None
) -> CanonicalRead:
    body: Final = _parse_body(raw)
    client: Final = _client()
    model_id: Final = str(uuid.uuid4())
    actor: Final = user.user_id or "proxy-admin"
    try:
        async with client.db.tx() as tx:
            collision: Final = await tx.litellm_proxymodeltable.find_first(where={"model_name": body.name})
            if collision is not None:
                raise _error(409, "conflict", "A deployment already uses this name")
            await tx.litellm_canonicalmodel.create(data={**_fields(body, actor), "id": model_id, "created_by": actor})
            repo: Final = ModelRepository(_TxClient(tx), publish_on_write=False)
            for position, connection in enumerate(body.connections):
                await _connection_create(tx, repo, model_id, body, connection, actor, position)
    except HTTPException:
        raise
    except Exception as exc:
        _write_error(exc)
        raise
    if body.connections:
        await _refresh()
    return await _load(client, model_id)


@router.put("/canonical/models/{model_id}", response_model=CanonicalRead)
async def update_canonical_model(
    model_id: str, user: Annotated[UserAPIKeyAuth, Depends(_admin)], raw: Annotated[object, Body()] = None
) -> CanonicalRead:
    body: Final = _parse_body(raw)
    client: Final = _client()
    actor: Final = user.user_id or "proxy-admin"
    try:
        async with client.db.tx() as tx:
            locked: Final = await tx.query_raw(
                'SELECT id FROM "LiteLLM_CanonicalModel" WHERE id = $1 FOR UPDATE', model_id
            )
            if not locked:
                raise _error(404, "not_found", "Canonical model not found")
            existing: Final = await tx.litellm_canonicalmodelconnection.find_many(
                where={"canonical_model_id": model_id}
            )
            collision: Final = await tx.litellm_proxymodeltable.find_first(
                where={"model_name": body.name, "model_id": {"notIn": [row.deployment_id for row in existing]}}
            )
            if collision is not None:
                raise _error(409, "conflict", "A deployment already uses this name")
            by_id: Final = {row.id: row for row in existing}
            by_binding: Final = {(row.provider_connection, row.provider_model): row.id for row in existing}
            resolved: Final = tuple(
                connection
                if connection.id is not None
                else connection.model_copy(
                    update={"id": by_binding.get((connection.provider_connection, connection.provider_model))}
                )
                for connection in body.connections
            )
            resolved_ids: Final = [connection.id for connection in resolved if connection.id is not None]
            if len(resolved_ids) != len(set(resolved_ids)):
                raise _error(400, "invalid_connection", "Connections resolve to the same ID")
            unknown: Final = set(resolved_ids) - by_id.keys()
            if unknown:
                raise _error(400, "invalid_connection", "Connection does not belong to this canonical model")
            repo: Final = ModelRepository(_TxClient(tx), publish_on_write=False)
            kept: Final = {connection.id for connection in resolved if connection.id is not None}
            for row in existing:
                if row.id not in kept:
                    await tx.litellm_canonicalmodelconnection.delete(where={"id": row.id})
                    await repo.delete_model(row.deployment_id)
            await tx.litellm_canonicalmodel.update(where={"id": model_id}, data=_fields(body, actor))
            for position, connection in enumerate(resolved):
                if connection.id is None:
                    await _connection_create(tx, repo, model_id, body, connection, actor, position)
                else:
                    await _connection_update(tx, repo, by_id[connection.id], body, connection, actor, position)
    except HTTPException:
        raise
    except Exception as exc:
        _write_error(exc)
        raise
    await _refresh()
    return await _load(client, model_id)


@router.delete("/canonical/models/{model_id}")
async def delete_canonical_model(model_id: str, user: Annotated[UserAPIKeyAuth, Depends(_admin)]) -> dict[str, bool]:
    client: Final = _client()
    try:
        async with client.db.tx() as tx:
            locked: Final = await tx.query_raw(
                'SELECT id FROM "LiteLLM_CanonicalModel" WHERE id = $1 FOR UPDATE', model_id
            )
            if not locked:
                raise _error(404, "not_found", "Canonical model not found")
            connections: Final = await tx.litellm_canonicalmodelconnection.find_many(
                where={"canonical_model_id": model_id}
            )
            repo: Final = ModelRepository(_TxClient(tx), publish_on_write=False)
            for connection in connections:
                await tx.litellm_canonicalmodelconnection.delete(where={"id": connection.id})
                await repo.delete_model(connection.deployment_id)
            await tx.litellm_canonicalmodel.delete(where={"id": model_id})
    except HTTPException:
        raise
    except Exception as exc:
        _write_error(exc)
        raise
    if connections:
        await _refresh()
    return {"success": True}


def _write_error(exc: Exception) -> None:
    if is_unique_violation(exc):
        raise _error(409, "conflict", "Canonical model name or connection ID already exists") from exc
    if getattr(exc, "code", None) in ("P2034", "40P01", "40001"):
        raise _error(409, "concurrent_write", "Concurrent model write; retry the request") from exc
    if getattr(exc, "code", None) == "P2003":
        raise _error(400, "invalid_credential", "Provider connection does not exist") from exc
    raise _error(500, "internal_error", "Canonical model write failed") from exc


@router.post("/canonical/resolve", response_model=list[ModelResolution])
async def resolve_models(body: ResolutionRequest, user: Annotated[UserAPIKeyAuth, Depends(_admin)]) -> list[ModelResolution]:
    client: Final = _client()
    rows: Final = await WriterPinnedClient(client.db).db.litellm_canonicalmodel.find_many(where={"base_model": body.base_model}, include={"connections": True})
    names: Final = tuple(dict.fromkeys(name for name in (*body.provider_connections, *(connection.provider_connection for row in rows for connection in row.connections)) if name is not None))
    credentials: Final = await asyncio.gather(*(CredentialsRepository(client).find_by_name(name) for name in names))
    providers: Final = {name: credential.credential_info.get("provider") if credential is not None else None for name, credential in zip(names, credentials)}
    known: Final = tuple(connection.provider_model for row in rows for connection in row.connections if providers.get(connection.provider_connection) == connection.provider_model.partition("/")[0])
    return [resolve_identity(body.base_model, name, providers.get(name), known) for name in body.provider_connections]
