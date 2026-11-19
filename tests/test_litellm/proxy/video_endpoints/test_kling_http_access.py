import os
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from mcp.types import CallToolResult

from litellm import Router
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy import proxy_server as proxy
from litellm.proxy._types import hash_token
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.credential_endpoints.kling_oauth import KEY
from litellm.proxy.utils import PrismaClient
from litellm.proxy.video_endpoints.kling_jobs import reconcile_jobs


@pytest.mark.asyncio
async def test_virtual_key_video_access(monkeypatch):
    url = os.environ.get("KLING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires a migrated disposable PostgreSQL database")
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("LITELLM_SALT_KEY", uuid4().hex)
    db = PrismaClient(database_url=url, proxy_logging_obj=proxy.proxy_logging_obj)
    await db.connect()
    uid = uuid4().hex
    model = "public-" + uid
    keys = ["sk-" + uuid4().hex for _ in range(2)]
    tokens = [hash_token(k) for k in keys]
    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": "kling/demo"},
                "model_info": {"id": uid, "mode": "video_generation"},
            }
        ]
    )
    monkeypatch.setattr(proxy, "master_key", "sk-" + uuid4().hex)
    monkeypatch.setattr(proxy, "prisma_client", db)
    monkeypatch.setattr(proxy, "store_model_in_db", True)
    monkeypatch.setattr(proxy, "llm_router", router)
    tool_names = []

    async def mcp(params, **kwargs):
        tool_names.append(params.name)
        result = {
            "who_am_i": {
                "availableModels": {"text_to_video": [{"model": "demo", "arguments": [{"name": "duration"}]}]}
            },
            "text_to_video": {"generationId": uid, "status": "submitted"},
            "query_tasks": {
                "generationId": uid,
                "status": "COMPLETED",
                "works": [{"contentType": "video", "url": "https://media.test/file.mp4"}],
            },
        }[params.name]
        return CallToolResult(content=[], structured_content=result)

    monkeypatch.setattr(MCPClient, "call_tool", AsyncMock(side_effect=mcp))
    media = b"fixture video bytes"

    def download(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, content=media, headers={"content-type": "video/mp4"})

    try:
        import json

        bundle = encrypt_value_helper(
            json.dumps(
                {
                    "client_id": "fixture",
                    "access_token": "fixture",
                    "refresh_token": "fixture",
                    "expires_at": 4102444800,
                }
            )
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_CredentialsTable"(credential_id,credential_name,credential_values,credential_info,created_by,updated_by) VALUES($1,$1,$2::jsonb,'{"provider":"kling","auth_type":"oauth"}','test','test')""",
            uid,
            json.dumps({KEY: bundle}),
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_ProxyModelTable"(model_id,model_name,litellm_params,created_by,updated_by) VALUES($1,$2,'{}','test','test')""",
            uid,
            model,
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_CanonicalModel"(id,name,base_model,video_price_per_second,created_by,updated_by) VALUES($1,$2,'kling/demo',0.123,'test','test')""",
            uid,
            model,
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_CanonicalModelConnection"(id,canonical_model_id,provider_connection,provider_model,mode,deployment_id) VALUES($1,$1,$1,'kling/demo','video_generation',$1)""",
            uid,
        )
        for token, allowed in zip(tokens, [model, "other-" + uid]):
            await db.db.execute_raw(
                """INSERT INTO "LiteLLM_VerificationToken"(token,models) VALUES($1,ARRAY[$2]::text[])""", token, allowed
            )
        async with httpx.AsyncClient(
            base_url="http://proxy.test",
            transport=httpx.ASGITransport(app=proxy.app),
            mounts={"https://media.test": httpx.MockTransport(download)},
        ) as client:
            owner, denied = [{"Authorization": "Bearer " + k} for k in keys]
            payload = {"model": model, "seconds": "5", "prompt": "a tree"}
            forbidden = await client.post("/v1/videos", headers=denied, json=payload)
            assert forbidden.status_code == 403, forbidden.text
            assert not tool_names
            created = await client.post("/v1/videos", headers=owner, json=payload)
            assert created.status_code == 200, created.text
            path = "/v1/videos/" + created.json()["id"]
            await reconcile_jobs()
            status = await client.get(path, headers=owner)
            assert status.json()["status"] == "completed", status.text
            for endpoint in (path, path + "/content"):
                hidden = await client.get(endpoint, headers=denied)
                assert hidden.status_code == 404 and "location" not in hidden.headers, hidden.text
            video = await client.get(path + "/content", headers=owner, follow_redirects=True)
            assert video.status_code == 200 and video.content == media
            assert [r.status_code for r in video.history] == [303]
            assert tool_names == ["who_am_i", "text_to_video", "query_tasks"]
    finally:
        for table, column in [
            ("LiteLLM_SpendLogs", "model_id"),
            ("LiteLLM_VideoGenerationJob", "model_id"),
            ("LiteLLM_CanonicalModelConnection", "id"),
            ("LiteLLM_CanonicalModel", "id"),
            ("LiteLLM_ProxyModelTable", "model_id"),
            ("LiteLLM_CredentialsTable", "credential_id"),
        ]:
            await db.db.execute_raw(f'DELETE FROM "{table}" WHERE {column}=$1', uid)
        await db.db.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token IN ($1,$2)', *tokens)
        await db.disconnect()
