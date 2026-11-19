import os
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml

from litellm.caching.dual_cache import DualCache
from litellm.proxy import proxy_server as proxy
from litellm.proxy.utils import PrismaClient
from litellm.proxy.video_endpoints.kling_jobs import _settle


@pytest.mark.asyncio
async def test_key_budget_recovers_after_committed_video_settlement(monkeypatch):
    url = os.environ.get("KLING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires a migrated disposable PostgreSQL database")
    uid = uuid4().hex
    counter = f"spend:key:{uid}"
    config = Path(__file__).resolve().parents[4] / "deploy/void/litellm-config.yaml"
    monkeypatch.setattr(proxy, "general_settings", yaml.safe_load(config.read_text())["general_settings"])
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(proxy, "store_model_in_db", True)

    def connection():
        return PrismaClient(database_url=url, proxy_logging_obj=proxy.proxy_logging_obj)

    db = connection()
    await db.connect()
    monkeypatch.setattr(proxy, "prisma_client", db)
    shared = {counter: 0.25}

    def set_max(key, value):
        shared[key] = max(shared[key], value)
        return shared[key]

    redis = AsyncMock()
    redis.async_get_cache.side_effect = lambda key: shared.get(key)
    redis.async_set_max.side_effect = set_max
    monkeypatch.setattr(proxy, "spend_counter_cache", DualCache(redis_cache=redis))
    try:
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_CredentialsTable"(credential_id,credential_name,credential_values,created_by,updated_by) VALUES($1,$1,'{}','test','test')""",
            uid,
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_VerificationToken"(token,spend,total_spend,max_budget) VALUES($1,0.25,0.25,1)""",
            uid,
        )
        await db.db.execute_raw(
            """INSERT INTO "LiteLLM_VideoGenerationJob"(id,credential,api_key,model_name,model_id,provider_model,requested_seconds,usd_per_second,status) VALUES($1,$1,$1,'public','model','fixture',5,0.125,'in_progress')""",
            uid,
        )
        assert await proxy.get_current_spend(counter, 0.25, max_budget=1) == 0.25
        with monkeypatch.context() as crash:
            crash.setattr(proxy, "increment_spend_counters", AsyncMock(side_effect=SystemExit("after commit")))
            with pytest.raises(SystemExit, match="after commit"):
                await _settle(uid, "https://media.test/video")
        assert shared[counter] == 0.25
        await db.disconnect()
        db = connection()
        await db.connect()
        monkeypatch.setattr(proxy, "prisma_client", db)
        monkeypatch.setattr(proxy, "spend_counter_cache", DualCache(redis_cache=redis))
        await _settle(uid, "https://media.test/video")
        spend = await proxy.get_current_spend(counter, 0.25, max_budget=1)
        assert 1 - spend == 0.125
        assert shared[counter] == 0.875
        assert await db.db.query_raw("""SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=$1""", uid) == [
            {"spend": 0.625}
        ]
        assert await db.db.query_raw("""SELECT status FROM "LiteLLM_VideoGenerationJob" WHERE id=$1""", uid) == [
            {"status": "completed"}
        ]
    finally:
        for table, column in [
            ("LiteLLM_SpendLogs", "request_id"),
            ("LiteLLM_VideoGenerationJob", "id"),
            ("LiteLLM_VerificationToken", "token"),
            ("LiteLLM_CredentialsTable", "credential_id"),
        ]:
            await db.db.execute_raw(f'DELETE FROM "{table}" WHERE {column}=$1', uid)
        await db.disconnect()
