"""PostgreSQL integration tests for the public inference ID mapping.

Set TEST_PUBLIC_INFERENCE_DATABASE_URL to a disposable PostgreSQL database to run.
Each test creates and drops its own schema; no application tables are touched.
"""

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
import pytest_asyncio

from litellm.proxy.db.public_inference_ids import PublicInferenceIdDb, PublicInferenceIdStore


@pytest_asyncio.fixture(loop_scope="function")
async def databases() -> AsyncIterator[tuple[PublicInferenceIdDb, PublicInferenceIdDb]]:
    database_url = os.environ.get("TEST_PUBLIC_INFERENCE_DATABASE_URL")
    if not database_url:
        pytest.skip("set TEST_PUBLIC_INFERENCE_DATABASE_URL to a disposable PostgreSQL database")

    import psycopg
    from prisma import Prisma
    from psycopg import sql

    parsed = urlsplit(database_url)
    schema = f"public_ids_{uuid.uuid4().hex}"
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params.pop("schema", None)  # psycopg does not accept Prisma's schema URL parameter
    admin_url = urlunsplit(parsed._replace(query=urlencode(params)))
    params["schema"] = schema
    prisma_url = urlunsplit(parsed._replace(query=urlencode(params)))
    migration = (
        Path(__file__).resolve().parents[4]
        / "litellm-proxy-extras/litellm_proxy_extras/migrations"
        / "20260928000000_add_public_inference_ids/migration.sql"
    )
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        clients = [Prisma(use_dotenv=False, datasource={"url": prisma_url}) for _ in range(2)]
        try:
            admin.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            for statement in migration.read_text().split(";"):
                if statement.strip():
                    admin.execute(statement)
            await asyncio.gather(*(client.connect() for client in clients))
            yield clients[0], clients[1]
        finally:
            await asyncio.gather(*(client.disconnect() for client in clients if client.is_connected()))
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio
async def test_concurrent_publish_is_atomic_and_prefixed(
    databases: tuple[PublicInferenceIdDb, PublicInferenceIdDb],
) -> None:
    first, second = (PublicInferenceIdStore(db) for db in databases)
    secret = "sensitive' upstream_id"
    ids = await asyncio.gather(*(store.publish("owner", "response", secret) for store in (first, second) * 12))
    assert len(set(ids)) == 1
    assert re.fullmatch(r"resp_[0-9a-f]{32}", ids[0])
    assert secret not in ids[0]
    assert await second.resolve("owner", "response", ids[0]) == secret
    assert len(await databases[0].query_raw('SELECT public_id FROM "LiteLLM_PublicInferenceId"')) == 1


@pytest.mark.asyncio
async def test_prefixes_and_owner_kind_value_isolation(
    databases: tuple[PublicInferenceIdDb, PublicInferenceIdDb],
) -> None:
    store = PublicInferenceIdStore(databases[0])
    for kind, prefix in (
        ("response", "resp_"),
        ("reasoning", "enc_"),
        ("item", "item_"),
        ("container", "cntr_"),
        ("file", "file_"),
        ("batch", "batch_"),
        ("video", "video_"),
        ("object", "obj_"),
    ):
        public_id = await store.publish("owner-a", kind, "native-id")
        assert re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", public_id)
        assert await store.resolve("owner-a", kind, public_id) == "native-id"
        assert await store.resolve("owner-b", kind, public_id) is None
        assert await store.resolve("owner-a", "not-a-kind", public_id) is None

    public_id = await store.publish("owner-a", "response", "native-id")
    other_owner = await store.publish("owner-b", "response", "native-id")
    other_value = await store.publish("owner-a", "response", "different-id")
    assert len({public_id, other_owner, other_value}) == 3
    assert await store.resolve("owner-a", "item", public_id) is None
    assert await store.resolve("owner-a", "response", "resp_missing") is None
    assert await store.resolve("owner-b", "response", other_owner) == "native-id"
    assert await store.resolve("owner-a", "response", other_value) == "different-id"


@pytest.mark.asyncio
async def test_expiry_cleanup_and_republish(databases: tuple[PublicInferenceIdDb, PublicInferenceIdDb]) -> None:
    db = databases[0]
    store = PublicInferenceIdStore(db)
    expired = await store.publish("owner", "file", "old-native")
    await db.execute_raw(
        "UPDATE \"LiteLLM_PublicInferenceId\" SET expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE public_id = $1",
        expired,
    )
    assert await store.resolve("owner", "file", expired) is None
    assert await store.publish("owner", "file", "old-native") == expired
    assert await store.resolve("owner", "file", expired) == "old-native"
    assert (
        len(
            await db.query_raw(
                "SELECT public_id FROM \"LiteLLM_PublicInferenceId\" WHERE public_id = $1 AND expires_at > CURRENT_TIMESTAMP + INTERVAL '29 days'",
                expired,
            )
        )
        == 1
    )

    await db.execute_raw(
        "UPDATE \"LiteLLM_PublicInferenceId\" SET expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE public_id = $1",
        expired,
    )
    live = await store.publish("owner", "file", "live-native")
    await store.cleanup_expired()
    assert await store.resolve("owner", "file", expired) is None
    assert await store.resolve("owner", "file", live) == "live-native"
    assert len(await db.query_raw('SELECT public_id FROM "LiteLLM_PublicInferenceId"')) == 1
    assert await store.publish("owner", "file", "old-native") != expired


@pytest.mark.asyncio
async def test_wrapped_payload_upgrades_without_downgrading(
    databases: tuple[PublicInferenceIdDb, PublicInferenceIdDb],
) -> None:
    first, second = (PublicInferenceIdStore(db) for db in databases)
    native = "provider-secret-item-id"
    wrapped = '{"id":"provider-secret-item-id","type":"function_call"}'
    public_id = await first.publish("owner", "item", native)
    assert await second.publish("owner", "item", wrapped, identity=native, replace=True) == public_id
    assert await first.resolve("owner", "item", public_id) == wrapped
    assert await first.publish("owner", "item", native) == public_id
    assert await second.resolve("owner", "item", public_id) == wrapped
    assert native not in public_id
    with pytest.raises(KeyError) as error:
        await first.publish("owner", "unsupported-kind", native)
    assert native not in str(error.value)


@pytest.mark.asyncio
async def test_database_failures_do_not_expose_native_ids(
    databases: tuple[PublicInferenceIdDb, PublicInferenceIdDb],
) -> None:
    db = databases[0]
    await db.execute_raw('DROP TABLE "LiteLLM_PublicInferenceId"')
    native = "private-upstream-id-should-never-appear-in-error"
    with pytest.raises(Exception) as error:
        await PublicInferenceIdStore(db).publish("owner", "response", native)
    assert native not in str(error.value)
