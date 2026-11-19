import secrets
from datetime import timedelta
from decimal import Decimal
from typing import Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import RedirectResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.credential_endpoints.kling_oauth import video_client
from litellm.proxy.management_endpoints.canonical_model_endpoints import _client
from litellm.router import Router
from litellm.types.videos.main import VideoObject


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1)
    seconds: int = Field(gt=0, le=86400)


def is_configured(router: Router, model: object) -> bool:
    if not isinstance(model, str):
        return False
    return any(d["litellm_params"]["model"].startswith("kling/") for d in router.get_model_list(model_name=model) or ())


def _owner(auth: UserAPIKeyAuth) -> str:
    from litellm.proxy._types import hash_token

    if not auth.api_key:
        raise HTTPException(403, "A Void API key is required")
    return hash_token(auth.api_key) if auth.api_key.startswith("sk-") else auth.api_key


async def submit(data: dict, auth: UserAPIKeyAuth, router: Router) -> VideoObject:
    try:
        body: Final = RequestBody.model_validate(data)
    except ValidationError:
        raise HTTPException(422, "Kling text-to-video requires model, prompt and seconds") from None
    key: Final = _owner(auth)
    deployment: Final = await router.async_get_available_deployment(
        model=body.model, request_kwargs={"metadata": {"user_api_key_team_id": auth.team_id}}
    )
    dep_id: Final = deployment["model_info"]["id"]
    db: Final = _client().db
    bindings: Final = await db.query_raw(
        """SELECT c.provider_connection, c.provider_model, m.video_price_per_second::text AS rate
        FROM "LiteLLM_CanonicalModelConnection" c JOIN "LiteLLM_CanonicalModel" m ON m.id=c.canonical_model_id
        WHERE c.deployment_id=$1 AND m.name=$2 AND c.mode='video_generation' """,
        dep_id,
        body.model,
    )
    if not bindings:
        raise HTTPException(409, "Video model is not configured")
    binding: Final = bindings[0]
    name: Final = binding["provider_connection"]
    model: Final = binding["provider_model"].removeprefix("kling/")
    client: Final = await video_client(name)
    caps: Final = await client.capabilities()
    spec: Final = next((entry for entry in caps.text_to_video() if entry.model == model), None)
    if spec is None:
        raise HTTPException(422, "Model is unavailable on this Kling connection")
    duration: Final = next((arg for arg in spec.arguments if arg.name == "duration"), None)
    if (
        duration is None
        or duration.allowedValues is not None
        and str(body.seconds) not in {str(v) for v in duration.allowedValues}
    ):
        raise HTTPException(422, "Duration is not supported by this model")
    if any(arg.required and arg.name not in ("prompt", "duration") for arg in spec.arguments) or any(
        slot.required for slot in spec.inputs
    ):
        raise HTTPException(422, "This model requires additional inputs")
    job: Final = "video_kg_" + secrets.token_hex(16)
    await db.execute_raw(
        """INSERT INTO "LiteLLM_VideoGenerationJob"(id,credential,api_key,user_id,team_id,model_name,model_id,provider_model,requested_seconds,usd_per_second)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::text::numeric)""",
        job,
        name,
        key,
        auth.user_id,
        auth.team_id,
        body.model,
        dep_id,
        model,
        body.seconds,
        binding["rate"],
    )
    try:
        result: Final = await client.submit(model, {"prompt": body.prompt, "duration": str(body.seconds)})
    except Exception:
        await db.execute_raw(
            "UPDATE \"LiteLLM_VideoGenerationJob\" SET status='manual_review' WHERE id=$1 AND provider_id IS NULL AND status='submitting'",
            job,
        )
        verbose_proxy_logger.warning("Kling submission needs manual reconciliation: %s", job)
        raise HTTPException(502, f"Submission unconfirmed; do not resubmit. Contact support with job {job}") from None
    await db.execute_raw(
        "UPDATE \"LiteLLM_VideoGenerationJob\" SET provider_id=$2,status='in_progress' WHERE id=$1 AND provider_id IS NULL AND status IN ('submitting','manual_review')",
        job,
        result.generationId,
    )
    return VideoObject(id=job, object="video", status="queued", seconds=str(body.seconds), model=body.model)


async def _owned(job: str, auth: UserAPIKeyAuth) -> dict:
    rows: Final = await _client().db.query_raw(
        'SELECT status,model_name,requested_seconds,output_url FROM "LiteLLM_VideoGenerationJob" WHERE id=$1 AND api_key=$2',
        job,
        _owner(auth),
    )
    if not rows:
        raise HTTPException(404, "Video not found")
    return rows[0]


async def status(job: str, auth: UserAPIKeyAuth) -> VideoObject:
    row: Final = await _owned(job, auth)
    return VideoObject(
        id=job, object="video", status=row["status"], model=row["model_name"], seconds=str(row["requested_seconds"])
    )


async def content(job: str, auth: UserAPIKeyAuth):
    row: Final = await _owned(job, auth)
    if row["status"] != "completed" or not row["output_url"]:
        raise HTTPException(409, "Video is not ready")
    return RedirectResponse(
        row["output_url"], status_code=303, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    )


async def reconcile_jobs() -> None:
    db: Final = _client().db
    uncertain: Final = await db.query_raw("""UPDATE "LiteLLM_VideoGenerationJob" SET status='manual_review'
        WHERE provider_id IS NULL AND (status='submission_unknown' OR
        (status='submitting' AND created_at < now()-interval '15 minutes')) RETURNING id""")
    for row in uncertain:
        verbose_proxy_logger.warning("Kling submission needs manual reconciliation: %s", row["id"])
    jobs: Final = await db.query_raw("""UPDATE "LiteLLM_VideoGenerationJob" SET next_poll=now()+interval '30 seconds'
        WHERE id IN (SELECT id FROM "LiteLLM_VideoGenerationJob" WHERE status='in_progress' AND next_poll<=now()
        ORDER BY next_poll LIMIT 4 FOR UPDATE SKIP LOCKED) RETURNING id,credential,provider_id""")
    for job in jobs:
        try:
            client: Final = await video_client(job["credential"])
            task: Final = await client.query(job["provider_id"])
            outcome: Final = task.billing_outcome()
            if outcome == "failed":
                await db.execute_raw(
                    "UPDATE \"LiteLLM_VideoGenerationJob\" SET status='failed' WHERE id=$1 AND status='in_progress'",
                    job["id"],
                )
            elif outcome == "billable":
                url: Final = next(
                    str(w.url)
                    for w in task.works
                    if w.contentType == "video"
                    and w.url
                    and w.url.scheme == "https"
                    and (w.status or task.status).casefold() in ("succeed", "completed")
                )
                await _settle(job["id"], url)
        except Exception:
            verbose_proxy_logger.warning("Unable to reconcile video job %s; will retry", job["id"])


async def _settle(job: str, url: str) -> None:
    from litellm.proxy.proxy_server import increment_spend_counters

    async with _client().db.tx(timeout=timedelta(seconds=10)) as tx:
        rows: Final = await tx.query_raw(
            """UPDATE "LiteLLM_VideoGenerationJob" SET status='completed',output_url=$2 WHERE id=$1 AND status='in_progress'
            RETURNING id,api_key,user_id,team_id,model_name,model_id,provider_model,created_at,(requested_seconds*usd_per_second)::text AS cost""",
            job,
            url,
        )
        if not rows:
            return
        r: Final = rows[0]
        cost: Final = float(Decimal(r["cost"]))
        await tx.execute_raw(
            """INSERT INTO "LiteLLM_SpendLogs"(request_id, call_type, api_key, spend, "startTime", "endTime",model,model_id,model_group,custom_llm_provider,"user",team_id,status,updated_at)
            SELECT id,'acreate_video',api_key,requested_seconds*usd_per_second,created_at,now(),provider_model,model_id,model_name,'kling',user_id,team_id,'success',now() FROM "LiteLLM_VideoGenerationJob" WHERE id=$1""",
            job,
        )
        await tx.execute_raw(
            'UPDATE "LiteLLM_VerificationToken" SET spend=spend+$2,total_spend=total_spend+$2 WHERE token=$1',
            r["api_key"],
            cost,
        )
        if r["user_id"]:
            await tx.execute_raw('UPDATE "LiteLLM_UserTable" SET spend=spend+$2 WHERE user_id=$1', r["user_id"], cost)
        if r["team_id"]:
            await tx.execute_raw('UPDATE "LiteLLM_TeamTable" SET spend=spend+$2 WHERE team_id=$1', r["team_id"], cost)
    await increment_spend_counters(token=r["api_key"], team_id=r["team_id"], user_id=r["user_id"], response_cost=cost)
