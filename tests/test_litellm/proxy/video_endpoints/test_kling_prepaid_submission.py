from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from litellm.llms.kling.videos.protocol import Capabilities, Submission
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import kling_jobs as jobs


@pytest.mark.asyncio
async def test_prepaid_submit_debits_before_provider_and_repeated_id_does_not_generate_again(monkeypatch):
    operations = []
    saved = {}

    async def query(sql, *args):
        if "SELECT c.provider_connection" in sql:
            return [{"provider_connection": "test", "provider_model": "kling/demo", "rate": "1"}]
        if "SELECT max_budget,spend" in sql:
            return [{"max_budget": 10, "spend": 0}]
        if "SELECT status,model_name" in sql:
            return [saved] if saved else []
        return [{"token": "key"}]

    async def execute(sql, *args):
        operations.append(sql)
        if "request_hash=$2" in sql:
            saved.update(status="queued", model_name="Kling", requested_seconds=5, request_hash=args[1])

    @asynccontextmanager
    async def transaction(**kwargs):
        yield SimpleNamespace(query_raw=query, execute_raw=execute)

    db = SimpleNamespace(query_raw=query, execute_raw=execute, tx=transaction)
    monkeypatch.setattr(jobs, "_client", lambda: SimpleNamespace(db=db))
    provider = SimpleNamespace(
        capabilities=AsyncMock(
            return_value=Capabilities.model_validate(
                {
                    "availableModels": {
                        "text_to_video": [
                            {"model": "demo", "arguments": [{"name": "duration", "allowedValues": [5]}]},
                        ]
                    }
                }
            )
        ),
    )

    async def generate(*args):
        assert any("SET spend=spend+$2" in sql for sql in operations)
        return Submission(generationId="provider-task", status="submitted")

    provider.submit = AsyncMock(side_effect=generate)
    monkeypatch.setattr(jobs, "video_client", AsyncMock(return_value=provider))
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "increment_spend_counters", AsyncMock())
    router = SimpleNamespace(
        async_get_available_deployment=AsyncMock(return_value={"model_info": {"id": "deployment"}})
    )
    payload = {"model": "Kling", "prompt": "Landscape", "seconds": 5, "request_id": str(uuid4()), "amount_usd": "5"}
    auth = UserAPIKeyAuth(api_key="key", user_id="user")
    first = await jobs.submit(payload, auth, router)
    second = await jobs.submit(payload, auth, router)
    assert first.id == second.id
    assert provider.submit.await_count == 1
    with pytest.raises(HTTPException) as failure:
        await jobs.submit(payload | {"prompt": "Other prompt"}, auth, router)
    assert failure.value.status_code == 409
    assert provider.submit.await_count == 1


@pytest.mark.asyncio
async def test_insufficient_video_budget_never_contacts_provider(monkeypatch):
    async def query(sql, *args):
        if "SELECT c.provider_connection" in sql:
            return [{"provider_connection": "test", "provider_model": "kling/demo", "rate": "1"}]
        if "SELECT max_budget,spend" in sql:
            return [{"max_budget": 1, "spend": 0}]
        return []

    execute = AsyncMock()

    @asynccontextmanager
    async def transaction(**kwargs):
        yield SimpleNamespace(query_raw=query, execute_raw=execute)

    monkeypatch.setattr(jobs, "_client", lambda: SimpleNamespace(db=SimpleNamespace(query_raw=query, tx=transaction)))
    provider = SimpleNamespace(
        capabilities=AsyncMock(
            return_value=Capabilities.model_validate(
                {
                    "availableModels": {
                        "text_to_video": [{"model": "demo", "arguments": [{"name": "duration", "allowedValues": [5]}]}]
                    },
                }
            )
        ),
        submit=AsyncMock(),
    )
    monkeypatch.setattr(jobs, "video_client", AsyncMock(return_value=provider))
    router = SimpleNamespace(
        async_get_available_deployment=AsyncMock(return_value={"model_info": {"id": "deployment"}})
    )
    with pytest.raises(HTTPException) as failure:
        await jobs.submit(
            {"model": "Kling", "prompt": "Landscape", "seconds": 5},
            UserAPIKeyAuth(api_key="key", user_id="user"),
            router,
        )
    assert failure.value.status_code == 402
    provider.submit.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_prepaid_job_refunds_once(monkeypatch):
    refunded = False

    async def query(sql, *args):
        nonlocal refunded
        if refunded:
            return []
        refunded = True
        return [{"api_key": "key", "user_id": "user", "prepaid": True, "cost": "5"}]

    execute = AsyncMock()

    @asynccontextmanager
    async def transaction(**kwargs):
        yield SimpleNamespace(query_raw=query, execute_raw=execute)

    monkeypatch.setattr(jobs, "_client", lambda: SimpleNamespace(db=SimpleNamespace(tx=transaction)))
    from litellm.proxy import proxy_server

    counter = AsyncMock()
    monkeypatch.setattr(proxy_server, "increment_spend_counters", counter)
    await jobs._refund_failed("job")
    await jobs._refund_failed("job")
    assert execute.await_count == 2
    counter.assert_awaited_once_with(token="key", user_id="user", team_id=None, response_cost=-5.0)
