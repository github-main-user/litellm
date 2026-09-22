"""Tests for the credential management endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.endpoints import get_llm_router
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def _call_as_admin(method: str, path: str, json_body: dict | None = None):
    missing = object()
    previous_override = app.dependency_overrides.get(user_api_key_auth, missing)
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        return client.request(method, path, json=json_body, headers={"Authorization": "Bearer test-key"})
    finally:
        if previous_override is missing:
            app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            app.dependency_overrides[user_api_key_auth] = previous_override


def _patch_credential(name: str, body: dict):
    return _call_as_admin("PATCH", f"/credentials/{name}", body)


def _delete_credential(name: str):
    return _call_as_admin("DELETE", f"/credentials/{name}")


def _list_credentials():
    return _call_as_admin("GET", "/credentials")


@pytest.fixture
def credential_store():
    """Stands the credential store up for one test: whether the database is reachable, what
    the proxy is already serving from memory, which router deployments resolve against, and
    what each repository call hands back."""

    def install(
        *,
        connected: bool = True,
        in_memory: tuple[object, ...] = (),
        llm_router: object | None = None,
        **repository_calls: AsyncMock,
    ) -> None:
        # This fixture exercises the repository fallback used by lightweight DB
        # adapters. Transaction/race behavior has a dedicated fake below.
        patch("litellm.proxy.proxy_server.prisma_client", object() if connected else None).start()
        patch("litellm.proxy.proxy_server.master_key", "sk-test-master").start()
        patch.object(litellm, "credential_list", list(in_memory)).start()
        app.dependency_overrides[get_llm_router] = lambda: llm_router
        repository = patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository").start()
        for call_name, result in repository_calls.items():
            setattr(repository.return_value, call_name, result)

    yield install
    patch.stopall()
    app.dependency_overrides.pop(get_llm_router, None)


def test_update_credential_answers_404_when_the_credential_does_not_exist(credential_store):
    """Regression: the handler used to ``return handle_exception_on_proxy(e)``, which makes
    the exception the response body and lets FastAPI answer 200, so a write the handler
    rejected read as a success to every caller that checks the status. The dashboard's API
    client branches on the status, so it reported a failed edit as applied."""
    credential_store(find_by_name=AsyncMock(return_value=None))

    response = _patch_credential(
        "definitely-not-there",
        {"credential_name": "definitely-not-there", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
    )

    assert response.status_code == 404, f"rejected write answered {response.status_code}: {response.text}"
    assert "error" in response.json()


def test_update_credential_answers_500_when_the_database_is_not_connected(credential_store):
    """The other rejection this handler raises must carry its own status too."""
    credential_store(connected=False)

    response = _patch_credential(
        "any-name",
        {"credential_name": "any-name", "credential_values": {"api_key": "sk-x"}, "credential_info": {}},
    )

    assert response.status_code == 500, f"rejected write answered {response.status_code}: {response.text}"


def test_update_credential_still_answers_200_on_a_successful_write(credential_store):
    """The fix must not turn a legitimate update into an error; the dashboard and the
    Playwright credentials spec both assert the success path."""
    stored = CredentialItem(
        credential_name="existing",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=AsyncMock(return_value=None))

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_values": {"api_key": "sk-new"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


def test_delete_credential_answers_404_when_the_credential_does_not_exist(credential_store):
    """Regression: prisma's ``delete`` hands back None when the ``where`` clause matched no row
    instead of raising, and the handler never looked. Deleting a name that was never stored
    answered 200 "Credential deleted successfully", so an operator scripting cleanup could not
    tell a real deletion from a typo."""
    credential_store(delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("definitely-not-there")

    assert response.status_code == 404, (
        f"delete of a missing credential answered {response.status_code}: {response.text}"
    )
    assert "definitely-not-there" in response.text


def test_delete_credential_still_answers_200_and_drops_the_credential_from_memory(credential_store):
    """The fix must not turn a real deletion into an error, and the deleted credential must
    stop being served from the in-memory list the proxy routes on."""
    stored = CredentialItem(
        credential_name="doomed",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    survivor = CredentialItem(
        credential_name="keeper",
        credential_values={"api_key": "sk-keep"},
        credential_info={},
    )
    credential_store(in_memory=(stored, survivor), delete_by_name=AsyncMock(return_value=MagicMock()))

    response = _delete_credential("doomed")

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert [credential.credential_name for credential in litellm.credential_list] == ["keeper"]


def test_delete_credential_leaves_a_credential_that_only_exists_in_memory_in_place(credential_store):
    """A credential declared in the config yaml is never written to the table, so the delete
    matches no row. Reporting success would be the same lie: it comes straight back on the next
    proxy boot. ``PATCH /credentials/{name}`` already answers 404 for that credential."""
    config_only = CredentialItem(
        credential_name="from-config-yaml",
        credential_values={"api_key": "sk-config"},
        credential_info={},
    )
    credential_store(in_memory=(config_only,), delete_by_name=AsyncMock(return_value=None))

    response = _delete_credential("from-config-yaml")

    assert response.status_code == 404, response.text
    assert [credential.credential_name for credential in litellm.credential_list] == ["from-config-yaml"]


def test_delete_credential_answers_500_when_the_database_is_not_connected(credential_store):
    """The handler used to ``return handle_exception_on_proxy(e)``, which makes the exception the
    response body and lets FastAPI answer 200. A DB-less proxy answered its own 500 as a success."""
    credential_store(connected=False)

    response = _delete_credential("any-name")

    assert response.status_code == 500, f"rejected delete answered {response.status_code}: {response.text}"


class _CredentialThatCannotBeMasked:
    """Stands in for anything that fails while ``GET /credentials`` builds its response."""

    credential_name = "unreadable"
    credential_info: dict = {}

    @property
    def credential_values(self):
        raise RuntimeError("credential store unreadable")


def test_get_credentials_answers_an_error_status_when_the_listing_fails(credential_store):
    """Same ``return`` instead of ``raise`` on the list route: a failed listing was serialized as
    a 200 whose body happened to be an error, so a caller reading the status saw an empty success."""
    credential_store(in_memory=(_CredentialThatCannotBeMasked(),))

    response = _list_credentials()

    assert response.status_code == 500, f"failed listing answered {response.status_code}: {response.text}"
    assert response.json().get("success") is not True


def _create_credential(body: dict):
    return _call_as_admin("POST", "/credentials", body)


class _UniqueViolation(Exception):
    code = "P2002"


def test_create_credential_answers_409_when_the_name_is_already_taken(credential_store):
    """Regression: the unique index used to surface as a Prisma 500 that callers string-matched."""
    credential_store(
        create=AsyncMock(side_effect=_UniqueViolation("Unique constraint failed on the fields: (`credential_name`)")),
    )

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 409, f"name collision answered {response.status_code}: {response.text}"
    message = response.json()["error"]["message"]
    assert message == (
        "Credential 'aws_bedrock' already exists. Update it with PATCH /credentials/aws_bedrock, or delete it first."
    ), f"the operator reads this message verbatim: {message}"
    assert "Unique constraint" not in response.text, f"the Prisma internals must not leak: {response.text}"


def test_create_credential_still_answers_500_when_the_write_fails_for_another_reason(credential_store):
    credential_store(create=AsyncMock(side_effect=Exception("connection reset by peer")))

    response = _create_credential(
        {"credential_name": "aws_bedrock", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 500, f"database fault answered {response.status_code}: {response.text}"


def test_create_credential_still_answers_200_for_a_name_that_is_free(credential_store):
    find_by_name = AsyncMock()
    credential_store(find_by_name=find_by_name, create=AsyncMock(return_value=None))

    response = _create_credential(
        {"credential_name": "brand_new", "credential_values": {"aws_access_key_id": "new"}, "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    find_by_name.assert_not_awaited(), "the unique index is the guard; create must not add a lookup"


def test_update_credential_resolves_credential_values_from_model_id_like_create(credential_store):
    """Regression: PATCH dropped ``model_id`` from the body, so an update that named a
    deployment instead of raw values wrote whatever the caller sent, or nothing."""
    stored = CredentialItem(
        credential_name="from-deployment",
        credential_values={"api_key": "sk-old"},
        credential_info={},
    )
    update_by_name = AsyncMock(return_value=None)
    router = MagicMock()
    router.get_deployment.return_value = {"model_name": "gpt-5.2"}
    router.get_deployment_credentials.return_value = {"api_key": "sk-from-deployment"}
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "deployment-1", "credential_info": {}},
    )

    assert response.status_code == 200, response.text
    router.get_deployment_credentials.assert_called_once_with("deployment-1")
    written = json.loads(update_by_name.await_args.kwargs["data"]["credential_values"])
    assert set(written) == {"api_key"}
    assert written["api_key"] != "sk-old", "the deployment's values must replace the stored ones"
    assert written["api_key"] != "sk-from-deployment", "values are encrypted before they reach the table"


def test_update_credential_answers_404_when_model_id_names_no_deployment(credential_store):
    stored = CredentialItem(
        credential_name="from-deployment", credential_values={"api_key": "sk-old"}, credential_info={}
    )
    update_by_name = AsyncMock(return_value=None)
    router = MagicMock()
    router.get_deployment.return_value = None
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=router)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "no-such-deployment", "credential_info": {}},
    )

    assert response.status_code == 404, response.text
    update_by_name.assert_not_awaited()


def test_update_credential_answers_500_when_model_id_is_given_but_no_router_is_loaded(credential_store):
    stored = CredentialItem(
        credential_name="from-deployment", credential_values={"api_key": "sk-old"}, credential_info={}
    )
    update_by_name = AsyncMock(return_value=None)
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name, llm_router=None)

    response = _patch_credential(
        "from-deployment",
        {"credential_name": "from-deployment", "model_id": "deployment-1", "credential_info": {}},
    )

    assert response.status_code == 500, response.text
    update_by_name.assert_not_awaited()


def test_proxy_create_validates_encrypts_and_only_exposes_safe_metadata(credential_store):
    create = AsyncMock(return_value=None)
    credential_store(create=create)

    response = _create_credential(
        {
            "credential_name": "proxied",
            "credential_values": {
                "api_key": "sk-value",
                "litellm_internal_proxy_url": "SOCKS5H://user:super-secret@proxy.example:1080",
            },
            "credential_info": {"provider": "openai"},
        }
    )

    assert response.status_code == 200, response.text
    written = create.await_args.kwargs["data"]
    values = json.loads(written["credential_values"])
    assert values["litellm_internal_proxy_url"] != "socks5h://user:super-secret@proxy.example:1080"
    assert "super-secret" not in values["litellm_internal_proxy_url"]
    assert json.loads(written["credential_info"])["proxy_configured"] is True
    assert litellm.credential_list[0].credential_values["litellm_internal_proxy_url"].lower().startswith("socks5h://")


def test_proxy_create_rejects_invalid_values_without_echoing_credentials(credential_store):
    create = AsyncMock(return_value=None)
    credential_store(create=create)
    secret = "do-not-echo-this-password"

    response = _create_credential(
        {
            "credential_name": "bad-proxy",
            "credential_values": {"litellm_internal_proxy_url": f"ftp://user:{secret}@proxy.example"},
            "credential_info": {},
        }
    )

    assert response.status_code == 400
    assert secret not in response.text
    create.assert_not_awaited()


@pytest.mark.parametrize("proxy_value", [123, {"url": "http://proxy.example"}, ["http://proxy.example"]])
def test_proxy_create_rejects_non_string_values(credential_store, proxy_value):
    create = AsyncMock(return_value=None)
    credential_store(create=create)

    response = _create_credential(
        {
            "credential_name": "bad-proxy-type",
            "credential_values": {"litellm_internal_proxy_url": proxy_value},
            "credential_info": {},
        }
    )

    assert response.status_code == 400
    create.assert_not_awaited()


def test_generic_log_masking_treats_proxy_fields_as_sensitive():
    from litellm.litellm_core_utils.litellm_logging import _get_masked_values

    secret_url = "http://user:proxy-password@proxy.example:8080"
    masked = _get_masked_values({"proxy_url": secret_url, "http_proxy": secret_url})

    assert masked["proxy_url"] != secret_url
    assert masked["http_proxy"] != secret_url
    assert "proxy-password" not in repr(masked)


def test_proxy_get_never_returns_internal_url_or_password(credential_store):
    credential_store(
        in_memory=(
            CredentialItem(
                credential_name="proxied",
                credential_values={
                    "api_key": "sk-secret",
                    "litellm_internal_proxy_url": "http://user:proxy-password@proxy.example:8080",
                },
                credential_info={"provider": "openai"},
            ),
        )
    )

    response = _list_credentials()

    assert response.status_code == 200
    body = response.json()["credentials"][0]
    assert "litellm_internal_proxy_url" not in body["credential_values"]
    assert "proxy-password" not in response.text
    assert body["credential_info"]["proxy_configured"] is True


def test_proxy_patch_absent_preserves_and_empty_clears_without_wiping_metadata(credential_store):
    stored = CredentialItem(
        credential_name="proxied",
        credential_values={"api_key": "encrypted-key", "litellm_internal_proxy_url": "encrypted-proxy"},
        credential_info={"provider": "anthropic", "auth_type": "oauth", "proxy_configured": True},
    )
    update = AsyncMock(return_value=None)
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update)

    preserved = _patch_credential(
        "proxied",
        {"credential_name": "proxied", "credential_values": {"api_key": "replacement"}, "credential_info": {}},
    )
    assert preserved.status_code == 200, preserved.text
    preserved_data = update.await_args.kwargs["data"]
    assert "litellm_internal_proxy_url" in json.loads(preserved_data["credential_values"])
    assert json.loads(preserved_data["credential_info"]) == {
        "provider": "anthropic",
        "auth_type": "oauth",
        "proxy_configured": True,
    }

    update.reset_mock()
    cleared = _patch_credential(
        "proxied",
        {
            "credential_name": "proxied",
            "credential_values": {"litellm_internal_proxy_url": ""},
            "credential_info": {"proxy_configured": True},
        },
    )
    assert cleared.status_code == 200, cleared.text
    cleared_data = update.await_args.kwargs["data"]
    assert "litellm_internal_proxy_url" not in json.loads(cleared_data["credential_values"])
    assert json.loads(cleared_data["credential_info"]) == {
        "provider": "anthropic",
        "auth_type": "oauth",
        "proxy_configured": False,
    }


@pytest.mark.parametrize("provider", ["chatgpt", "anthropic"])
def test_oauth_proxy_validation_never_echoes_secret_inputs(provider):
    secret = "proxy-password-never-in-response"
    response = _call_as_admin(
        "POST",
        f"/credentials/{provider}/oauth/start",
        {"credential_name": "proxied", "proxy_url": {"password": secret}},
    )
    assert response.status_code == 422, response.text
    assert secret not in response.text
    assert all("input" not in error and "ctx" not in error for error in response.json()["detail"])


def test_proxy_patch_refreshes_memory_from_latest_encrypted_row(credential_store):
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    find = AsyncMock()
    cached = CredentialItem(
        credential_name="proxied",
        credential_values={"api_key": "old-token"},
        credential_info={"provider": "openai", "auth_type": "api_key"},
    )
    credential_store(in_memory=(cached,), find_by_name=find, update_by_name=AsyncMock())
    find.return_value = CredentialItem(
        credential_name="proxied",
        credential_values={"api_key": encrypt_value_helper("rotated-token")},
        credential_info={"provider": "openai", "auth_type": "api_key"},
    )
    response = _patch_credential(
        "proxied",
        {
            "credential_name": "proxied",
            "credential_values": {"litellm_internal_proxy_url": "http://proxy.example:8080"},
            "credential_info": {},
        },
    )
    assert response.status_code == 200, response.text
    latest = CredentialAccessor.find_credential("proxied")
    assert latest is not None
    assert latest.credential_values["api_key"] == "rotated-token"
    assert latest.credential_values["litellm_internal_proxy_url"] == "http://proxy.example:8080"
    assert latest.credential_info["proxy_configured"] is True


@pytest.mark.asyncio
async def test_generic_patch_rereads_after_both_oauth_advisory_locks(monkeypatch):
    """A token refreshed while PATCH was waiting must be the value PATCH merges."""
    from litellm.proxy.credential_endpoints.endpoints import (
        _credential_lock_keys,
        _update_credential_under_oauth_locks,
    )

    class Table:
        def __init__(self):
            self.written = None

        async def find_unique(self, *, where):
            assert transaction.locks == _credential_lock_keys("oauth-credential", "oauth-credential")
            return {
                "credential_name": "oauth-credential",
                "credential_values": {"oauth_token_bundle": "freshly-rotated-encrypted-token"},
                "credential_info": {"provider": "anthropic", "auth_type": "oauth"},
            }

        async def update(self, *, where, data):
            self.written = data

    class Transaction:
        def __init__(self):
            self.litellm_credentialstable = Table()
            self.locks = []
            self.committed = False

        async def execute_raw(self, sql, lock_key):
            self.locks.append(lock_key)

    class TransactionContext:
        def __init__(self, transaction):
            self.transaction = transaction

        async def __aenter__(self):
            return self.transaction

        async def __aexit__(self, exc_type, exc, traceback):
            self.transaction.committed = exc_type is None
            return False

    class Database:
        def __init__(self, transaction):
            self.transaction = transaction

        def tx(self, *, timeout):
            assert timeout.total_seconds() == 120
            return TransactionContext(self.transaction)

    transaction = Transaction()

    async def publish_after_commit(table):
        assert transaction.committed
        assert table == "litellm_credentialstable"

    publish = AsyncMock(side_effect=publish_after_commit)
    monkeypatch.setattr(
        "litellm.proxy.credential_endpoints.endpoints.publish_config_change_for_object_type", publish
    )
    prisma = MagicMock()
    prisma.db = Database(transaction)
    repository = MagicMock()
    repository.find_by_name = AsyncMock(side_effect=AssertionError("stale read outside transaction"))
    monkeypatch.setattr(
        "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
        lambda value, new_encryption_key=None: f"encrypted:{value}",
    )
    patch_item = CredentialItem(
        credential_name="oauth-credential",
        credential_values={"litellm_internal_proxy_url": "http://proxy.example:8080"},
        credential_info={"proxy_configured": True},
    )

    await _update_credential_under_oauth_locks(
        prisma, repository, "oauth-credential", patch_item, False, "admin"
    )

    assert transaction.locks == _credential_lock_keys("oauth-credential", "oauth-credential")
    written_values = json.loads(transaction.litellm_credentialstable.written["credential_values"])
    assert written_values["oauth_token_bundle"] == "freshly-rotated-encrypted-token"
    assert written_values["litellm_internal_proxy_url"] == "encrypted:http://proxy.example:8080"
    repository.find_by_name.assert_not_awaited()
    publish.assert_awaited_once()


def test_update_credential_still_accepts_a_body_without_credential_values(credential_store):
    """Renaming or re-tagging a credential sends only ``credential_info``; that must not 422."""
    stored = CredentialItem(credential_name="existing", credential_values={"api_key": "sk-old"}, credential_info={})
    update_by_name = AsyncMock(return_value=None)
    credential_store(find_by_name=AsyncMock(return_value=stored), update_by_name=update_by_name)

    response = _patch_credential(
        "existing",
        {"credential_name": "existing", "credential_info": {"custom_llm_provider": "openai"}},
    )

    assert response.status_code == 200, response.text
    written = update_by_name.await_args.kwargs["data"]
    assert json.loads(written["credential_info"]) == {
        "custom_llm_provider": "openai",
        "proxy_configured": False,
    }
    assert set(json.loads(written["credential_values"])) == {"api_key"}, "stored values survive an info-only patch"
