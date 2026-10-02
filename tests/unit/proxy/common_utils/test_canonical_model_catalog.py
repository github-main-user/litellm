from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.proxy import proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.canonical_model_catalog import CatalogRecord, catalog_entries
from litellm.router import Router


def _record(name: str, blocked: tuple[bool, ...] = (False,)) -> CatalogRecord:
    return CatalogRecord(
        name=name,
        base_model="lab/model",
        created_at=datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc),
        input_price_per_million_tokens=Decimal("1.123456789012"),
        output_price_per_million_tokens=Decimal("2.000000000001"),
        cache_read_price_per_million_tokens=Decimal("0.000000000001"),
        cache_write_price_per_million_tokens=Decimal("3E+2"),
        connections=[{"deployment": {"blocked": value}} for value in blocked],
    )


def _expected(row: CatalogRecord) -> dict[str, object]:
    return {
        "id": row.name,
        "object": "model",
        "created": int(row.created_at.timestamp()),
        "owned_by": "catalog-brand",
        "base_model": row.base_model,
        "pricing": {
            "unit": "USD_per_million_tokens",
            "input": "1.123456789012",
            "output": "2.000000000001",
            "cache_read": "0.000000000001",
            "cache_write": "300",
        },
    }


def test_catalog_filters_unbound_paused_wildcard_and_unauthorized_records() -> None:
    visible: Final = _record("public", (True, False))
    records: Final = (
        visible,
        _record("paused", (True, True)),
        _record("unbound", ()),
        _record("private"),
        _record("provider/*"),
        _record("unhealthy"),
    )
    entries: Final = catalog_entries(records, "catalog-brand", {"public", "unhealthy"}, {"unhealthy"})
    assert [entry.model_dump() for entry in entries] == [_expected(visible)]


@pytest.fixture
def catalog_proxy(monkeypatch):
    records = (_record("public"), _record("other"), _record("paused", (True,)), _record("unbound", ()))

    class CatalogTable:
        async def find_many(self, **kwargs):
            return records

    router = Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/example", "api_key": "unused"}}
            for name in ("public", "other", "internal-deployment", "openai/*")
        ]
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", SimpleNamespace(db=SimpleNamespace(litellm_canonicalmodel=CatalogTable())))
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setenv("LITELLM_PUBLIC_API_BRAND", "catalog-brand")
    monkeypatch.setattr(ps, "master_key", "sk-catalog-master")
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    return records, router


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?return_wildcard_routes=true&include_metadata=true&include_model_access_groups=true",
        "?only_model_access_groups=true&scope=expand&team_id=another-team&fallback_type=general",
    ],
)
def test_anonymous_catalog_never_exposes_generic_routes_or_metadata(catalog_proxy, query: str) -> None:
    records, router = catalog_proxy
    before = router.get_model_names()
    response = TestClient(ps.app).get("/v1/models" + query)
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": [_expected(records[0]), _expected(records[1])]}
    assert router.get_model_names() == before


@pytest.mark.asyncio
async def test_authenticated_catalog_uses_existing_key_model_filter(catalog_proxy) -> None:
    from starlette.requests import Request

    records, _ = catalog_proxy
    response = await ps.model_list(
        request=Request({"type": "http", "method": "GET", "path": "/v1/models", "headers": []}),
        user_api_key_dict=UserAPIKeyAuth(models=["public"]),
        return_wildcard_routes=True,
        include_model_access_groups=True,
        include_metadata=True,
        scope="expand",
    )
    assert response == {"object": "list", "data": [_expected(records[0])]}


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer invalid"},
        {"x-api-key": "invalid"},
        {"Authorization": ""},
    ],
)
def test_invalid_supplied_credentials_are_not_anonymous(catalog_proxy, headers) -> None:
    response = TestClient(ps.app).get("/v1/models", headers=headers)
    assert response.status_code in (401, 403), response.text
    assert "data" not in response.json()


def test_ordinary_browser_cookies_do_not_require_catalog_authentication(catalog_proxy) -> None:
    records, _ = catalog_proxy
    response = TestClient(ps.app).get("/v1/models", headers={"Cookie": "locale=en; analytics=anonymous"})
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": [_expected(records[0]), _expected(records[1])]}


def test_anthropic_anonymous_catalog_keeps_native_envelope(catalog_proxy) -> None:
    response = TestClient(ps.app).get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["has_more"] is False
    assert body["first_id"] == "public"
    assert body["last_id"] == "other"
    assert [entry["id"] for entry in body["data"]] == ["public", "other"]
    assert all(entry["type"] == "model" for entry in body["data"])


def test_authenticated_http_catalog_and_models_route_remains_private(catalog_proxy) -> None:
    records, _ = catalog_proxy
    client = TestClient(ps.app)
    response = client.get("/v1/models", headers={"Authorization": "Bearer sk-catalog-master"})
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": [_expected(records[0]), _expected(records[1])]}
    private_response = client.get("/models")
    assert private_response.status_code in (401, 403), private_response.text


def test_custom_credential_header_is_not_anonymous(catalog_proxy, monkeypatch) -> None:
    monkeypatch.setattr(ps, "general_settings", {"litellm_key_header_name": "x-catalog-key"})
    response = TestClient(ps.app).get("/v1/models", headers={"x-catalog-key": "invalid"})
    assert response.status_code in (401, 403), response.text


@pytest.mark.asyncio
async def test_noncanonical_authenticated_listing_retains_legacy_format(catalog_proxy, monkeypatch) -> None:
    from starlette.requests import Request

    monkeypatch.setattr(ps, "prisma_client", None)
    response = await ps.model_list(
        request=Request({"type": "http", "method": "GET", "path": "/v1/models", "headers": []}),
        user_api_key_dict=UserAPIKeyAuth(models=["public"]),
    )
    assert response["object"] == "list"
    assert [row["id"] for row in response["data"]] == ["public"]
    assert "pricing" not in response["data"][0]


def test_anonymous_without_canonical_database_never_lists_config_models(catalog_proxy, monkeypatch) -> None:
    monkeypatch.setattr(ps, "prisma_client", None)
    response = TestClient(ps.app).get("/v1/models?return_wildcard_routes=true&scope=expand")
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": []}


def test_catalog_base_model_can_be_null() -> None:
    record = _record("public").model_copy(update={"base_model": None})
    assert catalog_entries((record,), "catalog-brand", None, ())[0].model_dump() == _expected(record)


@pytest.mark.asyncio
async def test_authenticated_anthropic_clients_keep_legacy_model_view(catalog_proxy) -> None:
    from starlette.requests import Request

    response = await ps.model_list(
        request=Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/v1/models",
                "headers": [(b"anthropic-version", b"2023-06-01")],
            }
        ),
        user_api_key_dict=UserAPIKeyAuth(models=["public"]),
    )
    assert response["has_more"] is False
    assert [row["id"] for row in response["data"]] == ["public"]
    assert response["data"][0]["type"] == "model"
    assert "max_input_tokens" in response["data"][0]
    assert "pricing" not in response["data"][0]


def test_catalog_owner_has_a_stable_non_provider_default(catalog_proxy, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_PUBLIC_API_BRAND")
    response = TestClient(ps.app).get("/v1/models")
    assert response.status_code == 200, response.text
    assert {entry["owned_by"] for entry in response.json()["data"]} == {"litellm"}


def test_untrusted_oauth_identity_headers_are_not_treated_as_anonymous(catalog_proxy, monkeypatch) -> None:
    monkeypatch.setattr(
        ps,
        "general_settings",
        {
            "enable_oauth2_proxy_auth": True,
            "oauth2_config_mappings": {"user_id": "x-catalog-user"},
        },
    )
    response = TestClient(ps.app).get("/v1/models", headers={"x-catalog-user": "forged-user"})
    assert response.status_code in (401, 403), response.text
