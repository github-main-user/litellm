import base64
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, Protocol, cast
from urllib.parse import urlencode

from fastapi import HTTPException, Request
from starlette.datastructures import QueryParams

from litellm.proxy.db.public_inference_ids import PUBLIC_ID_PREFIXES

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


class InferenceIdStore(Protocol):
    async def publish(
        self, owner: str, kind: str, value: str, *, identity: str | None = None, replace: bool = False
    ) -> str: ...

    async def resolve(self, owner: str, kind: str, public_id: str) -> str | None: ...


STATE_KEY: Final = "public_inference_ids"
_FIELDS: Final = {
    "response_id": "response",
    "previous_response_id": "response",
    "encrypted_content": "reasoning",
    "item_id": "item",
    "container_id": "container",
    "file_id": "file",
    "input_file_id": "file",
    "output_file_id": "file",
    "error_file_id": "file",
    "training_file": "file",
    "validation_file": "file",
    "batch_id": "batch",
    "video_id": "video",
    "fine_tuning_job_id": "object",
    "vector_store_id": "object",
    "character_id": "object",
}
_CHILDREN: Final = frozenset(
    {
        "response",
        "input",
        "output",
        "item",
        "content",
        "annotations",
        "annotation",
        "part",
        "data",
        "tools",
        "container",
        "code_interpreter_call",
        "code_interpreter_results",
        "messages",
        "message",
        "choices",
        "delta",
        "content_block",
        "provider_specific_fields",
        "thinking_blocks",
        "reasoning_items",
    }
)
_PRIVATE_FIELDS: Final = frozenset(
    {
        "_hidden_params",
        "litellm_metadata",
        "litellm_params",
        "model_info",
        "model_id",
        "custom_llm_provider",
        "api_base",
        "api_key",
        "litellm_model_name",
        "vertex_ai_grounding_metadata",
        "vertex_ai_url_context_metadata",
        "vertex_ai_safety_results",
        "vertex_ai_citation_metadata",
    }
)
_CACHE_USAGE_EXTENSIONS: Final = frozenset(
    {
        "cache_creation_token_details",
        "cache_creation_tokens",
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }
)
_USAGE_DETAILS: Final = frozenset(
    {"prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details"}
)


def _public_usage(value: dict[str, object]) -> dict[str, object]:
    return {
        key: {name: item for name, item in detail.items() if name not in _CACHE_USAGE_EXTENSIONS}
        if key in _USAGE_DETAILS and isinstance(detail, dict)
        else detail
        for key, detail in value.items()
        if key not in _CACHE_USAGE_EXTENSIONS
    }


_PUBLIC_ID: Final = re.compile(r"(?:resp|enc|item|cntr|file|batch|video|obj)_[a-f0-9]{32}\Z")


def get_public_inference_id_store() -> InferenceIdStore:
    from litellm.proxy.db.public_inference_ids import PublicInferenceIdStore
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    return PublicInferenceIdStore(prisma_client.db)


def _managed(value: str) -> bool:
    if value.startswith(("litellm", "encitem_")):
        return True
    stripped: Final = next(
        (
            value[len(prefix) :]
            for prefix in ("resp_", "cntr_", "file-", "batch_", "video_", "char_", "character_", "vs_", "ftjob_")
            if value.startswith(prefix)
        ),
        value,
    )
    try:
        prefix: Final = stripped[:64]
        return base64.urlsafe_b64decode(prefix + "=" * (-len(prefix) % 4)).startswith(b"litellm")
    except (ValueError, UnicodeError):
        return False


def _kind(field: str, value: str, resource: str | None) -> str | None:
    if field in _FIELDS:
        return _FIELDS[field]
    if field == "container" and value != "auto":
        return "container"
    if field in ("first_id", "last_id", "after", "before") and resource == "response":
        return "item"
    if field in ("id", "first_id", "last_id", "after", "before"):
        if resource == "model":
            return None
        if value.startswith(("resp_", "litellm_poll_")):
            return "response"
        if value.startswith(("encitem_", "item_")):
            return "item"
        for kind, prefix in PUBLIC_ID_PREFIXES.items():
            if value.startswith(prefix):
                return kind
        return resource if resource in PUBLIC_ID_PREFIXES else "object" if _managed(value) else None
    return None


class PublicInferenceIds:
    def __init__(self, store_factory: Callable[[], InferenceIdStore], brand: str) -> None:
        self._store_factory: Final = store_factory
        self._store: InferenceIdStore | None = None
        self.brand: Final = brand
        self.owner: str | None = None
        self.model: str | None = None
        self.resource: str | None = None
        self.deployment: str | None = None
        self._published: Final[dict[tuple[str, str], str]] = {}
        self._resolved: Final[dict[tuple[str, str], str]] = {}

    def bind(self, user: "UserAPIKeyAuth") -> None:
        identity: Final = (
            json.dumps(["user", user.user_id, user.team_id])
            if user.user_id
            else json.dumps(["key", user.api_key or user.token])
        )
        if not user.user_id and not (user.api_key or user.token):
            raise HTTPException(status_code=401, detail="Authentication required")
        self.owner = hashlib.sha256(identity.encode()).hexdigest()

    def store(self) -> InferenceIdStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    async def identifier(self, kind: str, value: str, *, incoming: bool) -> str:
        public: Final = _PUBLIC_ID.fullmatch(value) is not None
        if incoming:
            if _managed(value) or (kind == "response" and not public):
                raise HTTPException(status_code=400, detail="Invalid resource identifier")
            if not public:
                return value
            if not value.startswith(PUBLIC_ID_PREFIXES[kind]):
                raise HTTPException(status_code=400, detail="Invalid resource identifier")
        elif kind == "reasoning" and not _managed(value):
            return value
        if self.owner is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        cache: Final = self._resolved if incoming else self._published
        identity: Final = self._item_identity(value) if kind == "item" and not incoming else None
        key: Final = (kind, json.dumps([identity, value]) if identity is not None else value)
        if key in cache:
            return cache[key]
        translated: Final = (
            await self.store().resolve(self.owner, kind, value)
            if incoming
            else await self.store().publish(
                self.owner, kind, value, identity=identity, replace=kind == "item" and value.startswith("encitem_")
            )
        )
        if translated is None:
            raise HTTPException(status_code=404, detail="Resource not found or expired")
        cache[key] = translated
        if not incoming:
            self._resolved.pop((kind, translated), None)
        if incoming:
            self._published[(kind, translated)] = value
            if kind in ("response", "container"):
                from litellm.responses.utils import ResponsesAPIRequestUtils

                if kind == "response":
                    self.deployment = ResponsesAPIRequestUtils.get_model_id_from_response_id(translated)
                original: Final = (
                    ResponsesAPIRequestUtils._decode_responses_api_response_id(translated).get("response_id")
                    if kind == "response"
                    else ResponsesAPIRequestUtils.decode_container_id_to_original(translated)
                )
                if original:
                    self._published[(kind, original)] = value
        return translated

    def _item_identity(self, value: str) -> str:
        from litellm.responses.utils import ResponsesAPIRequestUtils

        decoded: Final = ResponsesAPIRequestUtils._decode_encrypted_item_id(value)
        if value.startswith("encitem_") and decoded is None:
            raise HTTPException(status_code=502, detail="Invalid upstream resource identifier")
        return json.dumps([decoded["model_id"], decoded["item_id"]] if decoded else [self.deployment, value])

    def authorized_response_id(self, addressed_id: str) -> str:
        resolved: Final = next(
            (
                internal
                for (kind, public), internal in self._resolved.items()
                if kind == "response" and addressed_id in (public, internal)
            ),
            None,
        )
        if resolved is None:
            raise HTTPException(status_code=403, detail="Resource access denied")
        return resolved

    async def payload(self, value: object, *, incoming: bool, resource: str | None = None) -> object:
        if isinstance(value, list):
            return [await self.payload(item, incoming=incoming, resource=resource) for item in value]
        if not isinstance(value, dict):
            return value
        source: Final = cast(dict[str, object], value)
        provider_fields: Final = source.get("provider_specific_fields")
        unique_fields: Final = (
            {
                key: item
                for key, item in provider_fields.items()
                if key != "native_finish_reason"
                and item not in (None, {}, [])
                and not (key in source and source[key] == item)
            }
            if isinstance(provider_fields, dict)
            else provider_fields
        )
        obj: Final = (
            {
                key: unique_fields if key == "provider_specific_fields" else item
                for key, item in source.items()
                if key != "provider_specific_fields" or unique_fields not in (None, {})
            }
            if not incoming
            else source
        )
        object_type: Final = obj.get("object")
        node_resource: Final = (
            "file"
            if object_type == "container.file"
            else object_type
            if isinstance(object_type, str) and (object_type in PUBLIC_ID_PREFIXES or object_type == "model")
            else resource
        )
        response_id: Final = obj.get("id")
        if not incoming and node_resource == "response" and isinstance(response_id, str):
            from litellm.responses.utils import ResponsesAPIRequestUtils

            self.deployment = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        return {
            key: await self._field(key, item, obj, incoming=incoming, resource=node_resource)
            for key, item in obj.items()
            if incoming or (key not in _PRIVATE_FIELDS and not key.startswith("litellm_"))
        }

    async def _field(
        self, key: str, value: object, obj: Mapping[str, object], *, incoming: bool, resource: str | None
    ) -> object:
        if (
            key == "usage"
            and not incoming
            and isinstance(value, dict)
            and (resource == "response" or "prompt_tokens" in value)
        ):
            return _public_usage(value)
        if isinstance(value, str):
            if (key == "signature" and obj.get("type") in ("thinking", "signature_delta")) or (
                key == "data" and obj.get("type") == "redacted_thinking"
            ):
                return await self.identifier("reasoning", value, incoming=incoming)
            kind: Final = _kind(key, value, self.resource if key in ("first_id", "last_id") else resource)
            if kind is not None and value:
                return await self.identifier(kind, value, incoming=incoming)
            if not incoming and key == "owned_by" and value.lower() == "litellm":
                return self.brand
            if not incoming and key == "model" and self.model and resource != "model":
                return self.model
            return value
        if key in ("file_ids", "result_files", "vector_store_ids") and isinstance(value, list):
            list_kind: Final = "object" if key == "vector_store_ids" else "file"
            return [
                await self.identifier(list_kind, item, incoming=incoming) if isinstance(item, str) else item
                for item in value
            ]
        if key not in _CHILDREN:
            return value
        if key == "input" and not isinstance(value, list):
            return value
        if key in ("input", "output", "content") and (
            obj.get("type") in ("function_call_output", "tool_result", "tool_use") or obj.get("role") == "tool"
        ):
            return value
        child_resource: Final = (
            "response"
            if key == "response"
            else "item"
            if key in ("item", "input", "output") or (key == "data" and self.resource == "response")
            else self.resource
            if key == "data"
            else None
        )
        return await self.payload(value, incoming=incoming, resource=child_resource)

    async def authorize_request(self, request: Request, user: "UserAPIKeyAuth", data: dict[str, object]) -> None:
        if self.owner is not None:
            return
        self.bind(user)
        requested_model: Final = data.get("model")
        self.model = requested_model if isinstance(requested_model, str) else None
        route: Final = request.scope.get("path", "")
        resources: Final = {
            "responses": "response",
            "containers": "container",
            "files": "file",
            "batches": "batch",
            "videos": "video",
            "fine_tuning": "object",
        }
        self.resource = next((resources[part] for part in reversed(route.split("/")) if part in resources), None)
        restored: Final = cast(dict[str, object], await self.payload(data, incoming=True))
        data.update(restored)
        cached_json: Final = getattr(request, "_json", None)
        if isinstance(cached_json, dict):
            cached_json.update(restored)
        request.scope["path_params"] = {
            key: await self.identifier(_FIELDS[key], value, incoming=True)
            if key in _FIELDS and isinstance(value, str)
            else value
            for key, value in request.path_params.items()
        }
        query: Final = tuple(
            [
                (
                    key,
                    await self.identifier(kind, value, incoming=True)
                    if (kind := _kind(key, value, self.resource))
                    else value,
                )
                for key, value in request.query_params.multi_items()
            ]
        )
        request.scope["query_string"] = urlencode(query).encode()
        request._query_params = QueryParams(query)


current_public_ids: Final[ContextVar[PublicInferenceIds | None]] = ContextVar("public_inference_ids", default=None)
