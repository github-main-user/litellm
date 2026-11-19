import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psycopg
import pytest
from psycopg.rows import dict_row

from litellm.llms.kling.videos.protocol import Capabilities, Submission
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import kling_jobs as jobs


@pytest.mark.asyncio
async def test_lost_submission_receipt_survives_worker_restart(monkeypatch):
    url = os.environ.get("KLING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires a migrated disposable PostgreSQL database")
    async with await psycopg.AsyncConnection.connect(
        url, autocommit=True, row_factory=dict_row, cursor_factory=psycopg.AsyncRawCursor
    ) as conn:
        await conn.execute(
            'CREATE TEMP TABLE "LiteLLM_VideoGenerationJob" (LIKE public."LiteLLM_VideoGenerationJob" INCLUDING ALL)'
        )
        await conn.execute('CREATE TEMP TABLE "LiteLLM_SpendLogs" (LIKE public."LiteLLM_SpendLogs" INCLUDING ALL)')
        await conn.execute("""CREATE FUNCTION pg_temp.lose_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'injected receipt loss'; END $$""")
        await conn.execute("""CREATE TRIGGER lose_receipt BEFORE UPDATE OF provider_id ON "LiteLLM_VideoGenerationJob"
            FOR EACH ROW EXECUTE FUNCTION pg_temp.lose_receipt()""")

        async def query(sql, *params):
            return await (await conn.execute(sql, params)).fetchall()

        async def execute(sql, *params):
            await conn.execute(sql, params)

        binding = {"provider_connection": "fixture", "provider_model": "kling/demo", "rate": "0.123"}
        monkeypatch.setattr(
            jobs,
            "_client",
            lambda: SimpleNamespace(
                db=SimpleNamespace(query_raw=AsyncMock(return_value=[binding]), execute_raw=execute)
            ),
        )
        provider = AsyncMock()
        provider.capabilities.return_value = Capabilities.model_validate(
            {"availableModels": {"text_to_video": [{"model": "demo", "arguments": [{"name": "duration"}]}]}}
        )
        provider.submit.return_value = Submission(generationId="accepted-by-kling", status="submitted")
        monkeypatch.setattr(jobs, "video_client", AsyncMock(return_value=provider))
        router = AsyncMock()
        router.async_get_available_deployment.return_value = {"model_info": {"id": "deployment"}}
        with pytest.raises(psycopg.errors.RaiseException, match="injected receipt loss"):
            await jobs.submit(
                {"model": "public", "prompt": "a tree", "seconds": 5}, UserAPIKeyAuth(api_key="fixture-key"), router
            )

        monkeypatch.setattr(
            jobs, "_client", lambda: SimpleNamespace(db=SimpleNamespace(query_raw=query, execute_raw=execute))
        )
        await jobs.reconcile_jobs()
        assert await query('SELECT status, provider_id FROM "LiteLLM_VideoGenerationJob"') == [
            {"status": "submitting", "provider_id": None}
        ]
        await execute("""UPDATE "LiteLLM_VideoGenerationJob" SET created_at=now()-interval '1 hour' """)
        await jobs.reconcile_jobs()
        await jobs.reconcile_jobs()
        assert await query('SELECT status, provider_id FROM "LiteLLM_VideoGenerationJob"') == [
            {"status": "manual_review", "provider_id": None}
        ]
        assert not await query('SELECT request_id FROM "LiteLLM_SpendLogs"')
        provider.submit.assert_awaited_once()
        provider.query.assert_not_awaited()
