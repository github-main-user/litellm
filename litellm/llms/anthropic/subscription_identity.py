import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.anthropic.oauth_client import AnthropicOAuthTokens

_OBJECT: Final = TypeAdapter(dict[str, object])
_SESSION_HEADER: Final = "x-claude-code-session-id"


@dataclass(frozen=True, slots=True)
class SubscriptionIdentity:
    account_id: str
    device_id: str


def _object(value: object) -> dict[str, object]:
    try:
        return _OBJECT.validate_python(value, strict=True)
    except ValidationError:
        return {}


def _user_metadata(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return _object(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        return _OBJECT.validate_json(value)
    except ValidationError:
        return {"client_user_id": value}


def get_subscription_identity(credential_name: object) -> SubscriptionIdentity | None:
    if not isinstance(credential_name, str):
        return None
    credential: Final = CredentialAccessor.find_credential(credential_name)
    if credential is None:
        return None
    info: Final = _object(cast(object, credential.credential_info))
    if info.get("provider") != "anthropic" or info.get("auth_type") != "oauth":
        return None
    values: Final = _object(cast(object, credential.credential_values))
    encoded: Final = values.get("litellm_internal_anthropic_auth_token")
    if not isinstance(encoded, str):
        return None
    tokens: Final = AnthropicOAuthTokens.from_json(encoded)
    return SubscriptionIdentity(account_id=tokens.account_id or "", device_id=tokens.device_id)


def _session_id(value: object, scope: str, gateway: bool) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed: Final = UUID(value)
    except ValueError:
        pass
    else:
        if not gateway:
            return str(parsed)
    return str(uuid5(NAMESPACE_URL, json.dumps(["litellm-anthropic", scope, value])))


def _gateway_metadata(params: Mapping[str, object]) -> tuple[dict[str, object], list[str | None] | None]:
    for slot in ("litellm_metadata", "metadata"):
        metadata: Final = _object(params.get(slot))
        auth: Final = metadata.get("user_api_key_auth")
        if auth is None or isinstance(auth, (dict, list, str, int, float, bool)):
            continue
        from litellm.proxy._types import UserAPIKeyAuth

        # only proxy-injected typed auth is trusted; end_user_id is caller-controlled.
        if isinstance(auth, UserAPIKeyAuth):
            return metadata, [auth.user_id, auth.team_id, auth.org_id, auth.project_id, auth.api_key]
    return _object(params.get("metadata")), None


def prepare_subscription_identity(
    body: Mapping[str, object],
    headers: Mapping[str, object],
    params: Mapping[str, object],
    identity: SubscriptionIdentity | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if identity is None:
        return dict(body), dict(headers)
    metadata: Final = _object(body.get("metadata"))
    incoming: Final = _user_metadata(metadata.get("user_id"))
    request_metadata, owner = _gateway_metadata(params)
    scope: Final = json.dumps([identity.account_id, identity.device_id, params.get("litellm_credential_name"), owner])
    header_session: Final = next((value for name, value in headers.items() if name.lower() == _SESSION_HEADER), None)
    # the original gateway session takes precedence over headers rewritten by an earlier attempt.
    gateway_session: Final = (
        request_metadata.get("session_id") or params.get("litellm_session_id") or params.get("litellm_trace_id")
        if owner is not None
        else None
    )
    session_id: Final = next(
        (
            normalized
            for candidate in (
                gateway_session,
                header_session,
                incoming.get("session_id"),
                request_metadata.get("session_id"),
                params.get("litellm_trace_id"),
            )
            if (normalized := _session_id(candidate, scope, owner is not None)) is not None
        ),
        str(uuid4()),
    )
    core: Final = {
        "device_id": identity.device_id,
        "account_uuid": identity.account_id,
        "session_id": session_id,
    }
    combined: Final = json.dumps({**incoming, **core}, separators=(",", ":"), ensure_ascii=True)
    user_id: Final = (
        combined
        if len(combined) <= 512
        else json.dumps(
            {
                "client_user_id_sha256": hashlib.sha256(str(metadata.get("user_id", "")).encode()).hexdigest(),
                **core,
            },
            separators=(",", ":"),
        )
    )
    return (
        {**body, "metadata": {**metadata, "user_id": user_id}},
        {
            **{name: value for name, value in headers.items() if name.lower() != _SESSION_HEADER},
            _SESSION_HEADER: session_id,
        },
    )
