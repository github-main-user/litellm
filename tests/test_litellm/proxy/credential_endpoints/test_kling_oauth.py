import asyncio
import json
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import httpx2
import pytest
from prisma import Prisma

from litellm.experimental_mcp_client.client import MCPClient, verbose_logger
from litellm.llms.kling.oauth import TokenBundle
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints import kling_oauth as kling


@pytest.mark.asyncio
async def test_concurrent_refresh_rotates_once_and_survives_reopen(monkeypatch, caplog):
    url = os.environ.get("KLING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires disposable PostgreSQL with the proxy credential schema")
    name = "rotation-" + uuid4().hex
    monkeypatch.setenv("LITELLM_SALT_KEY", uuid4().hex)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger=verbose_logger.name)
    monkeypatch.setattr(verbose_logger, "propagate", True)
    debug = Mock(wraps=verbose_logger.debug)
    monkeypatch.setattr(verbose_logger, "debug", debug)
    previous_access, previous_refresh, new_access, new_refresh = (uuid4().hex for _ in range(4))
    expired = TokenBundle(
        client_id=name, access_token=previous_access, refresh_token=previous_refresh, expires_at=time.time() - 60
    )
    exchanges = []
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def token(request):
        assert str(request.url) == "https://kling.ai/auth/token"
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["refresh_token"] and form["client_id"] == [name]
        exchanges.extend(form["refresh_token"])
        assert exchanges == [previous_refresh]
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"access_token": new_access, "refresh_token": new_refresh, "expires_in": 3600})

    def mcp(request):
        assert str(request.url) == "https://kling.ai/mcp"
        assert request.headers["Authorization"] == f"Bearer {new_access}"
        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx2.Response(202)
        if message["method"] == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1"},
            }
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "who_am_i", "inputSchema": {"type": "object"}}]}
        else:
            assert message["method"] == "tools/call"
            calls.append(message["params"]["name"])
            result = {"content": [], "structuredContent": {"availableModels": {"text_to_video": []}}}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    monkeypatch.setattr(
        kling,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(token), **kw),
            HTTPError=httpx.HTTPError,
        ),
    )
    monkeypatch.setattr(
        MCPClient,
        "_create_httpx_client_factory",
        lambda self: lambda **kw: httpx2.AsyncClient(transport=httpx2.MockTransport(mcp), **kw),
    )

    def database():
        return Prisma(datasource={"url": url})

    db = database()
    await db.connect()
    monkeypatch.setattr(kling, "_client", lambda: SimpleNamespace(db=db))
    try:
        await db.execute_raw(
            """INSERT INTO "LiteLLM_CredentialsTable"(credential_id,credential_name,credential_values,credential_info,created_by,updated_by)
            VALUES($1,$1,$2::jsonb,'{"provider":"kling","auth_type":"oauth"}','test','test')""",
            name,
            json.dumps({kling.KEY: encrypt_value_helper(expired.model_dump_json())}),
        )
        async with asyncio.timeout(10), asyncio.TaskGroup() as group:
            first = group.create_task(kling.video_client(name))
            await entered.wait()
            second = group.create_task(kling.video_client(name))
            while not await db.query_raw("""SELECT 1 AS blocked FROM pg_stat_activity WHERE datname=current_database()
                AND wait_event_type='Lock' AND query LIKE 'SELECT credential_values FROM %' """):
                await asyncio.sleep(0.01)
            release.set()
        for client in (first.result(), second.result()):
            await client.capabilities()
        await db.disconnect()
        db = database()
        await db.connect()
        row = await db.litellm_credentialstable.find_unique(where={"credential_name": name})
        stored = TokenBundle.model_validate_json(decrypt_value_helper(row.credential_values[kling.KEY], kling.KEY))
        assert stored.refresh_token == new_refresh and stored.expires_at > time.time() + 60
        await (await kling.video_client(name)).capabilities()
        assert calls == ["who_am_i"] * 3 and exchanges == [previous_refresh]
        assert (verbose_logger.name, logging.DEBUG, "MCP client sending tool call to session") in caplog.record_tuples
        for secret in (previous_access, previous_refresh, new_access, new_refresh):
            assert secret not in caplog.text
            assert all(secret not in repr(call.args) for call in debug.call_args_list)
    finally:
        await db.execute_raw('DELETE FROM "LiteLLM_CredentialsTable" WHERE credential_name=$1', name)
        await db.disconnect()
