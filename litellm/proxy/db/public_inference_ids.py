import hashlib
import secrets
from collections.abc import Mapping, Sequence
from typing import Final, Protocol


class PublicInferenceIdDb(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


PUBLIC_ID_PREFIXES: Final = {
    "response": "resp_",
    "reasoning": "enc_",
    "item": "item_",
    "container": "cntr_",
    "file": "file_",
    "batch": "batch_",
    "video": "video_",
    "object": "obj_",
}

_PUBLISH_SQL: Final = """
INSERT INTO "LiteLLM_PublicInferenceId" (public_id, owner, kind, fingerprint, value, created_at, updated_at, expires_at)
VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP + INTERVAL '30 days')
ON CONFLICT (owner, kind, fingerprint) DO UPDATE SET
    expires_at = CURRENT_TIMESTAMP + INTERVAL '30 days',
    updated_at = CURRENT_TIMESTAMP,
    value = CASE WHEN $6 THEN EXCLUDED.value ELSE "LiteLLM_PublicInferenceId".value END
RETURNING public_id
"""

_RESOLVE_SQL: Final = """
SELECT value FROM "LiteLLM_PublicInferenceId"
WHERE owner = $1 AND kind = $2 AND public_id = $3 AND expires_at > CURRENT_TIMESTAMP
"""

_CLEANUP_SQL: Final = """
DELETE FROM "LiteLLM_PublicInferenceId" WHERE expires_at <= CURRENT_TIMESTAMP
"""


class PublicInferenceIdStore:
    def __init__(self, db: PublicInferenceIdDb) -> None:
        self.db: Final = db

    async def publish(
        self, owner: str, kind: str, value: str, *, identity: str | None = None, replace: bool = False
    ) -> str:
        prefix: Final = PUBLIC_ID_PREFIXES[kind]
        public_id: Final = prefix + secrets.token_hex(16)
        fingerprint: Final = hashlib.sha256((identity or value).encode("utf-8")).hexdigest()
        rows: Final = await self.db.query_raw(_PUBLISH_SQL, public_id, owner, kind, fingerprint, value, replace)
        return str(rows[0]["public_id"])

    async def resolve(self, owner: str, kind: str, public_id: str) -> str | None:
        if kind not in PUBLIC_ID_PREFIXES:
            return None
        rows: Final = await self.db.query_raw(_RESOLVE_SQL, owner, kind, public_id)
        return str(rows[0]["value"]) if rows else None

    async def cleanup_expired(self) -> None:
        await self.db.execute_raw(_CLEANUP_SQL)
