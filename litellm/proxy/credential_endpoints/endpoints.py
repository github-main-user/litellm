"""
CRUD endpoints for storing reusable credentials.
"""

import hashlib
from collections.abc import Mapping
from datetime import timedelta
from typing import (
    Annotated,
    Final,
    cast,  # noqa: TID251  # jsonify_object in proxy/utils.py is annotated with a bare dict
)

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.models.credentials import UpdateCredentialItem
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.config_sync_pubsub import publish_config_change_for_object_type
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.repositories.base_repository import is_unique_violation
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import CreateCredentialItem, CredentialItem

router: Final = APIRouter()
_CREDENTIAL_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_CREDENTIAL_PROXY_KEY: Final = "litellm_internal_proxy_url"
_PROXY_CONFIGURED_KEY: Final = "proxy_configured"


def _prepare_proxy_config(
    credential_values: Mapping[str, object],
    credential_info: Mapping[str, object],
    *,
    is_create: bool,
) -> tuple[dict[str, object], dict[str, object], bool]:
    values: Final = dict(credential_values)
    info: Final = dict(credential_info)
    if _CREDENTIAL_PROXY_KEY not in values:
        if is_create:
            info[_PROXY_CONFIGURED_KEY] = False
        return values, info, False

    proxy_url: Final = values.get(_CREDENTIAL_PROXY_KEY)
    if not isinstance(proxy_url, str):
        raise HTTPException(status_code=400, detail="Credential proxy URL must be a string")
    if proxy_url == "":
        values.pop(_CREDENTIAL_PROXY_KEY, None)
        info[_PROXY_CONFIGURED_KEY] = False
        return values, info, True

    try:
        from litellm.litellm_core_utils.credential_proxy import validate_proxy_url

        values[_CREDENTIAL_PROXY_KEY] = validate_proxy_url(proxy_url)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid credential proxy URL") from None
    info[_PROXY_CONFIGURED_KEY] = True
    return values, info, False


def _public_credential_parts(credential: CredentialItem) -> tuple[dict[str, object], dict[str, object]]:
    values: Final = dict(credential.credential_values or {})
    configured: Final = bool(values.pop(_CREDENTIAL_PROXY_KEY, None))
    info: Final = dict(credential.credential_info or {})
    info[_PROXY_CONFIGURED_KEY] = configured
    return values, info


def _credential_lock_keys(*credential_names: str) -> list[int]:
    keys: Final = frozenset(
        int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "big", signed=True)
        for name in credential_names
        for identity in (name, f"anthropic-oauth:credential:{name}")
    )
    return sorted(keys)


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem, new_encryption_key: str | None = None) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values: Final = {}
        for key, value in (credential.credential_values or {}).items():
            encrypted_credential_values[key] = encrypt_value_helper(value, new_encryption_key)

        # Return a new object to avoid mutating the caller's credential, which
        # is kept in memory and should remain unencrypted.
        return CredentialItem(
            credential_name=credential.credential_name,
            credential_values=encrypted_credential_values,
            credential_info=credential.credential_info or {},
        )


def _credential_exists_detail(credential_name: str) -> str:
    return (
        f"Credential '{credential_name}' already exists. "
        f"Update it with PATCH /credentials/{credential_name}, or delete it first."
    )


def get_llm_router() -> litellm.Router | None:
    from litellm.proxy.proxy_server import llm_router

    return llm_router


def _resolve_deployment_credentials(llm_router: litellm.Router | None, model_id: str) -> Mapping[str, object]:
    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail="LLM router not found. Please ensure you have a valid router instance.",
        )
    if llm_router.get_deployment(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    credential_values: Final = llm_router.get_deployment_credentials(model_id)
    if credential_values is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _CREDENTIAL_DICT_ADAPTER.validate_python(credential_values)


@router.post(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def create_credential(
    request: Request,
    fastapi_response: Response,
    credential: CreateCredentialItem,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    llm_router: Annotated[litellm.Router | None, Depends(get_llm_router)] = None,
):
    """
    [BETA] endpoint. This might change unexpectedly.
    Stores credential in DB.
    Reloads credentials in memory.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        credential_values: Final = (
            _resolve_deployment_credentials(llm_router, credential.model_id)
            if credential.model_id
            else credential.credential_values
        )
        if credential_values is None:
            raise HTTPException(
                status_code=400,
                detail="Credential values are required. Unable to infer credential values from model ID.",
            )
        raw_values: Final = _CREDENTIAL_DICT_ADAPTER.validate_python(credential_values)
        processed_values, processed_info, _ = _prepare_proxy_config(
            raw_values,
            credential.credential_info,
            is_create=True,
        )
        processed_credential: Final = CredentialItem(
            credential_name=credential.credential_name,
            credential_values=processed_values,
            credential_info=processed_info,
        )
        encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(processed_credential)
        credentials_dict: Final = encrypted_credential.model_dump()
        credentials_dict_jsonified: Final = cast(  # cast-ok: deep-copies a model_dump, so keys are str
            "dict[str, object]", jsonify_object(credentials_dict)
        )
        try:
            await CredentialsRepository(prisma_client).create(
                data={
                    **credentials_dict_jsonified,
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                }
            )
        except Exception as e:
            if not is_unique_violation(e):
                raise
            raise HTTPException(status_code=409, detail=_credential_exists_detail(credential.credential_name))

        ## ADD TO LITELLM ##
        CredentialAccessor.upsert_credentials([processed_credential])

        return {"success": True, "message": "Credential created successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def get_credentials(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    try:
        masked_credentials = []
        for credential in litellm.credential_list:
            public_values, public_info = _public_credential_parts(credential)
            masked_credentials.append(
                {
                    "credential_name": credential.credential_name,
                    "credential_values": _get_masked_values(public_values),
                    "credential_info": public_info,
                }
            )
        return {"success": True, "credentials": masked_credentials}
    except Exception as e:
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_name/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential_by_name(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    try:
        for credential in litellm.credential_list:
            if credential.credential_name == credential_name:
                public_values, public_info = _public_credential_parts(credential)
                masked_credential = CredentialItem(
                    credential_name=credential.credential_name,
                    credential_values=_get_masked_values(
                        public_values,
                        unmasked_length=4,
                        number_of_asterisks=4,
                    ),
                    credential_info=public_info,
                )
                return masked_credential
        raise HTTPException(
            status_code=404,
            detail="Credential not found. Got credential name: " + credential_name,
        )
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_model/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential_by_model(
    request: Request,
    fastapi_response: Response,
    model_id: str = Path(..., description="The model ID to look up credentials for"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        if llm_router is None:
            raise HTTPException(status_code=500, detail="LLM router not found")
        model: Final = llm_router.get_deployment(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        credential_values: Final = llm_router.get_deployment_credentials(model_id)
        if credential_values is None:
            raise HTTPException(status_code=404, detail="Model not found")
        public_values: Final = dict(credential_values)
        proxy_configured: Final = bool(public_values.pop(_CREDENTIAL_PROXY_KEY, None))
        masked_credential_values: Final = _get_masked_values(
            public_values,
            unmasked_length=4,
            number_of_asterisks=4,
        )
        credential: Final = CredentialItem(
            credential_name=f"{model.model_name}-credential-{model_id}",
            credential_values=masked_credential_values,
            credential_info={_PROXY_CONFIGURED_KEY: proxy_configured},
        )
        return credential
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.delete(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def delete_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        deleted: Final = await CredentialsRepository(prisma_client).delete_by_name(credential_name)
        if deleted is None:
            raise HTTPException(
                status_code=404,
                detail="Credential not found. Got credential name: " + credential_name,
            )

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [cred for cred in litellm.credential_list if cred.credential_name != credential_name]
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        raise handle_exception_on_proxy(e)


def update_db_credential(
    db_credential: CredentialItem,
    updated_patch: CredentialItem,
    new_encryption_key: str | None = None,
    *,
    clear_proxy: bool = False,
) -> CredentialItem:
    """
    Update a credential in the DB.
    """
    merged_credential: Final = CredentialItem(
        credential_name=db_credential.credential_name,
        credential_info=db_credential.credential_info,
        credential_values=db_credential.credential_values,
    )

    encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(
        updated_patch,
        new_encryption_key,
    )
    # update model name
    if encrypted_credential.credential_name:
        merged_credential.credential_name = encrypted_credential.credential_name

    # update litellm params
    if clear_proxy:
        merged_credential.credential_values.pop(_CREDENTIAL_PROXY_KEY, None)
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        encrypted_params: Final = {k: v for k, v in encrypted_credential.credential_values.items()}
        merged_credential.credential_values.update(encrypted_params)

    # update model info
    if encrypted_credential.credential_info:
        """Update credential info"""
        merged_credential.credential_info.update(encrypted_credential.credential_info)
    merged_credential.credential_info[_PROXY_CONFIGURED_KEY] = bool(
        merged_credential.credential_values.get(_CREDENTIAL_PROXY_KEY)
    )

    return merged_credential


async def _update_credential_under_oauth_locks(
    prisma_client: object,
    credentials_repository: CredentialsRepository,
    credential_name: str,
    patch: CredentialItem,
    clear_proxy: bool,
    updated_by: str | None,
) -> CredentialItem:
    database = getattr(prisma_client, "db", None)
    tx_factory = getattr(database, "tx", None) if database is not None else None
    if not callable(tx_factory):
        db_credential = await credentials_repository.find_by_name(credential_name)
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        merged = update_db_credential(db_credential, patch, clear_proxy=clear_proxy)
        jsonified = cast("dict[str, object]", jsonify_object(merged.model_dump()))
        await credentials_repository.update_by_name(
            credential_name,
            data={**jsonified, "updated_by": updated_by},
        )
        return merged

    async with database.tx(timeout=timedelta(minutes=2)) as transaction:
        for lock_key in _credential_lock_keys(credential_name, patch.credential_name):
            await transaction.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
        row = await transaction.litellm_credentialstable.find_unique(where={"credential_name": credential_name})
        db_credential = CredentialsRepository._to_model(row)
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        merged = update_db_credential(db_credential, patch, clear_proxy=clear_proxy)
        jsonified = cast("dict[str, object]", jsonify_object(merged.model_dump()))
        await transaction.litellm_credentialstable.update(
            where={"credential_name": credential_name},
            data={**jsonified, "updated_by": updated_by},
        )
    await publish_config_change_for_object_type("litellm_credentialstable")
    return merged


@router.patch(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def update_credential(
    request: Request,
    fastapi_response: Response,
    credential: UpdateCredentialItem,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    llm_router: Annotated[litellm.Router | None, Depends(get_llm_router)] = None,
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        credentials_repository: Final = CredentialsRepository(prisma_client)
        raw_patch_values: Final = _CREDENTIAL_DICT_ADAPTER.validate_python(
            _resolve_deployment_credentials(llm_router, credential.model_id)
            if credential.model_id
            else credential.credential_values or {}
        )
        patch_values, patch_info, clear_proxy = _prepare_proxy_config(
            raw_patch_values,
            _CREDENTIAL_DICT_ADAPTER.validate_python(credential.credential_info),
            is_create=False,
        )
        patch: Final = CredentialItem(
            credential_name=credential.credential_name,
            credential_info=patch_info,
            credential_values=patch_values,
        )
        merged_credential: Final = await _update_credential_under_oauth_locks(
            prisma_client,
            credentials_repository,
            credential_name,
            patch,
            clear_proxy,
            user_api_key_dict.user_id,
        )

        # Sync in-memory credential_list (skip if not in memory - e.g., proxy restarted)
        new_name: Final = merged_credential.credential_name
        existing_in_memory: CredentialItem | None = None
        for cred in litellm.credential_list:
            if cred.credential_name == credential_name:
                existing_in_memory = cred
                break

        if existing_in_memory is not None:
            in_memory_values: Final = {
                key: decrypt_value_helper(value, key)
                for key, value in merged_credential.credential_values.items()
            }
            in_memory_info: Final = dict(merged_credential.credential_info or {})
            in_memory_info[_PROXY_CONFIGURED_KEY] = bool(in_memory_values.get(_CREDENTIAL_PROXY_KEY))
            updated_in_memory: Final = CredentialItem(
                credential_name=new_name,
                credential_values=in_memory_values,
                credential_info=in_memory_info,
            )
            # Remove old entry if renamed, then use upsert_credentials to handle duplicates
            if new_name != credential_name:
                litellm.credential_list = [c for c in litellm.credential_list if c.credential_name != credential_name]
            CredentialAccessor.upsert_credentials([updated_in_memory])

        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        raise handle_exception_on_proxy(e)
