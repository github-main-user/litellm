import base64
import hashlib
import json
import re
import secrets
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
    "call_id": "tool",
    "tool_call_id": "tool",
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
        "tool_calls",
        "function",
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


def _without_redundant_tool_signature(source: dict[str, object], signature: str) -> dict[str, object]:
    return {
        key: {name: item for name, item in value.items() if name != "thought_signature" or item != signature}
        if key == "provider_specific_fields" and isinstance(value, dict)
        else _without_redundant_tool_signature(value, signature)
        if key == "function" and isinstance(value, dict)
        else value
        for key, value in source.items()
    }


_PUBLIC_ID: Final = re.compile(r"(?:resp|enc|item|call|cntr|file|batch|video|obj)_[a-f0-9]{32}\Z")


def get_public_inference_id_store() -> InferenceIdStore:
    from litellm.proxy.db.public_inference_ids import PublicInferenceIdStore
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    return PublicInferenceIdStore(prisma_client.db)


def _managed(value: str) -> bool:
    if value.startswith(("litellm", "encitem_")) or "__thought__" in value:
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
        if resource == "tool":
            return "tool"
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
        self.tool_namespace: str = secrets.token_hex(16)
        self.native_messages: bool = False
        self.choice_index: int = 0
        self.tool_index: int = 0
        self._tool_ids: Final[dict[tuple[str, int, int], str]] = {}
        self._tool_signatures: Final[dict[tuple[str, int, int], str]] = {}
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
        identity: Final = (
            self._item_identity(value)
            if kind == "item" and not incoming
            else self._tool_identity(value)
            if kind == "tool" and not incoming
            else None
        )
        key: Final = (kind, json.dumps([identity, value]) if identity is not None else value)
        if key in cache:
            return cache[key]
        translated: Final = (
            await self._resolve_identifier(kind, value)
            if incoming
            else await self.store().publish(
                self.owner,
                kind,
                json.dumps({"response_id": value, "public_model": self.model})
                if kind == "response" and self.model
                else value,
                identity=value if kind == "response" and self.model else identity,
                replace=(kind == "item" and value.startswith("encitem_"))
                or (kind == "tool" and "__thought__" in value)
                or (kind == "response" and bool(self.model)),
            )
        )
        if translated is None:
            raise HTTPException(status_code=404, detail="Resource not found or expired")
        cache[key] = translated
        if not incoming:
            self._resolved.pop((kind, translated), None)
        if incoming:
            self._published[(kind, translated)] = value
            if kind == "tool":
                self._published[(kind, json.dumps([self._tool_identity(translated), translated]))] = value
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

    async def _resolve_identifier(self, kind: str, value: str) -> str | None:
        stored: Final = await self.store().resolve(cast(str, self.owner), kind, value)
        if kind != "response" or stored is None or not stored.startswith("{"):
            return stored
        record: Final = json.loads(stored)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("response_id"), str)
            or not isinstance(record.get("public_model"), str)
            or not record["response_id"]
            or not record["public_model"]
        ):
            raise HTTPException(status_code=502, detail="Invalid stored response")
        if self.model is None:
            self.model = record["public_model"]
        return record["response_id"]

    def _tool_identity(self, value: str) -> str:
        return json.dumps([self.tool_namespace, self.choice_index, self.tool_index, value.split("__thought__", 1)[0]])

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

    async def _prepare_tool(self, source: dict[str, object]) -> dict[str, object]:
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _encode_tool_call_id_with_signature,
            _get_thought_signature_from_tool,
        )

        index: Final = source.get("index")
        slot: Final = (self.tool_namespace, self.choice_index, index) if isinstance(index, int) else None
        supplied_id: Final = source.get("id")
        original_id: Final = supplied_id if isinstance(supplied_id, str) else self._tool_ids.get(slot) if slot else None
        supplied_signature: Final = _get_thought_signature_from_tool(source)
        signature: Final = (
            supplied_signature
            if isinstance(supplied_signature, str) and supplied_signature
            else self._tool_signatures.get(slot)
            if slot
            else None
        )
        if slot and signature:
            self._tool_signatures[slot] = signature
        if original_id is None:
            if signature and slot is None:
                raise HTTPException(status_code=502, detail="Invalid upstream tool call")
            return source
        encoded_id: Final = (
            _encode_tool_call_id_with_signature(original_id.split("__thought__", 1)[0], signature)
            if signature
            else original_id
        )
        if slot:
            self._tool_ids[slot] = encoded_id
        if supplied_id is None:
            if signature:
                await self.identifier("tool", encoded_id, incoming=False)
            return source
        return {**source, "id": encoded_id}

    async def payload(self, value: object, *, incoming: bool, resource: str | None = None) -> object:
        if isinstance(value, list):
            return [await self.payload(item, incoming=incoming, resource=resource) for item in value]
        if not isinstance(value, dict):
            return value
        raw_source: Final = cast(dict[str, object], value)
        prepared_response_tool: Final = (
            await self._prepare_tool({**raw_source, "id": raw_source.get("call_id")})
            if not incoming and not self.native_messages and raw_source.get("type") == "function_call"
            else None
        )
        original_source: Final = (
            {**raw_source, "call_id": prepared_response_tool["id"]}
            if prepared_response_tool is not None and isinstance(prepared_response_tool.get("id"), str)
            else await self._prepare_tool(raw_source)
            if not incoming and not self.native_messages and resource == "tool"
            else raw_source
        )
        tool_id: Final = original_source.get("id")
        signature: Final = tool_id.partition("__thought__")[2] if isinstance(tool_id, str) else ""
        source: Final = (
            _without_redundant_tool_signature(original_source, signature)
            if not incoming and resource == "tool" and signature
            else original_source
        )
        provider_fields: Final = source.get("provider_specific_fields")
        continuation_fields: Final = (
            {
                key: item
                for key, item in provider_fields.items()
                if key in ("thought_signatures", "thought_signature", "signature")
            }
            if isinstance(provider_fields, dict)
            and resource not in ("tool", "tool_function")
            and source.get("type") != "function_call"
            else {}
        )
        unique_fields: Final = (
            {
                key: item
                for key, item in provider_fields.items()
                if key not in ("native_finish_reason", "web_search_calls")
                and item not in (None, {}, [])
                and not (key in source and source[key] == item)
            }
            if isinstance(provider_fields, dict)
            else provider_fields
        )
        promoted: Final = (
            {
                key: provider_fields[key]
                for key in ("thinking_blocks", "reasoning_items")
                if isinstance(provider_fields, dict)
                and key in provider_fields
                and provider_fields[key] not in (None, {}, [])
                and key not in source
            }
            if not incoming and not self.native_messages
            else {}
        )
        obj: Final = (
            {
                key: (unique_fields if self.native_messages else continuation_fields)
                if key == "provider_specific_fields"
                else item
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
        if not incoming and isinstance(response_id, str) and (node_resource == "response" or "choices" in obj):
            self.tool_namespace = response_id
        if not incoming and node_resource == "response" and isinstance(response_id, str):
            from litellm.responses.utils import ResponsesAPIRequestUtils

            self.deployment = ResponsesAPIRequestUtils.get_model_id_from_response_id(response_id)
        translated: Final = {
            key: await self._field(key, item, obj, incoming=incoming, resource=node_resource)
            for key, item in {**obj, **promoted}.items()
            if incoming
            or (
                key not in _PRIVATE_FIELDS
                and not key.startswith("litellm_")
                and (key != "provider_specific_fields" or self.native_messages or bool(continuation_fields))
            )
        }
        return {
            key: item
            for key, item in translated.items()
            if incoming or key != "provider_specific_fields" or item not in (None, {})
        }

    async def _choice(self, value: object, index: int, *, incoming: bool) -> object:
        previous: Final = self.choice_index
        declared_index: Final = value.get("index") if isinstance(value, dict) else None
        self.choice_index = declared_index if isinstance(declared_index, int) else index
        try:
            return await self.payload(value, incoming=incoming)
        finally:
            self.choice_index = previous

    async def _tool(self, value: object, index: int, *, incoming: bool) -> object:
        previous: Final = self.tool_index
        declared_index: Final = value.get("index") if isinstance(value, dict) else None
        self.tool_index = declared_index if isinstance(declared_index, int) else index
        try:
            return await self.payload(value, incoming=incoming, resource="tool")
        finally:
            self.tool_index = previous

    async def _field(
        self, key: str, value: object, obj: Mapping[str, object], *, incoming: bool, resource: str | None
    ) -> object:
        if key == "choices" and isinstance(value, list):
            return [await self._choice(item, index, incoming=incoming) for index, item in enumerate(value)]
        if key == "tool_calls" and isinstance(value, list):
            return [await self._tool(item, index, incoming=incoming) for index, item in enumerate(value)]
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
            if not incoming and key == "model" and resource != "model":
                if self.model:
                    return self.model
                if resource == "response":
                    raise HTTPException(status_code=502, detail="Response model unavailable")
            return value
        if key in ("file_ids", "result_files", "vector_store_ids") and isinstance(value, list):
            list_kind: Final = "object" if key == "vector_store_ids" else "file"
            return [
                await self.identifier(list_kind, item, incoming=incoming) if isinstance(item, str) else item
                for item in value
            ]
        if (
            key == "source"
            and obj.get("type") in ("image", "document")
            and isinstance(value, dict)
            and value.get("type") in ("file", "content")
        ):
            return await self.payload(value, incoming=incoming)
        if key == "content" and obj.get("type") == "tool_result" and isinstance(value, list):
            return [
                await self.payload(item, incoming=incoming)
                if isinstance(item, dict) and item.get("type") in ("image", "document")
                else item
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
            else "tool_function"
            if key == "function" and resource == "tool"
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
        from litellm.proxy.common_utils.http_parsing_utils import (
            _is_form_content_type,
            _safe_get_request_parsed_body,
            _safe_set_request_parsed_body,
        )

        cached_body: Final = _safe_get_request_parsed_body(request)
        cached_json: Final = getattr(request, "_json", None)
        client_body: Final = dict(
            cached_body if cached_body is not None else cached_json if isinstance(cached_json, dict) else data
        )
        restored: Final = cast(dict[str, object], await self.payload(data, incoming=True))
        restored_body: Final = cast(dict[str, object], await self.payload(client_body, incoming=True))
        data.update(restored)
        _safe_set_request_parsed_body(request, restored_body)
        if not _is_form_content_type(request.headers.get("content-type", "")):
            if isinstance(cached_json, dict):
                cached_json.update(restored_body)
            request._json = restored_body
            request._body = json.dumps(restored_body, ensure_ascii=False).encode("utf-8")
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
