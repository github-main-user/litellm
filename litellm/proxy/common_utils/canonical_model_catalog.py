from collections.abc import Container
from datetime import datetime
from decimal import Decimal
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


class CatalogDeployment(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    blocked: bool


class CatalogConnection(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    deployment: CatalogDeployment | None


class CatalogRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    base_model: str
    created_at: datetime
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    cache_read_price_per_million_tokens: Decimal
    cache_write_price_per_million_tokens: Decimal
    connections: list[CatalogConnection]


class CatalogPricing(BaseModel):
    unit: Literal["USD_per_million_tokens"] = "USD_per_million_tokens"
    input: str
    output: str
    cache_read: str
    cache_write: str


class CatalogEntry(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    base_model: str
    pricing: CatalogPricing


def catalog_entries(
    records: tuple[CatalogRecord, ...],
    owner: str,
    available_names: Container[str] | None,
    hidden_names: Container[str],
) -> tuple[CatalogEntry, ...]:
    visible: Final = (
        row
        for row in records
        if "*" not in row.name
        and row.name not in hidden_names
        and (available_names is None or row.name in available_names)
        and any(
            connection.deployment is not None and not connection.deployment.blocked for connection in row.connections
        )
    )
    return tuple(
        CatalogEntry(
            id=row.name,
            created=int(row.created_at.timestamp()),
            owned_by=owner,
            base_model=row.base_model,
            pricing=CatalogPricing(
                input=format(row.input_price_per_million_tokens, "f"),
                output=format(row.output_price_per_million_tokens, "f"),
                cache_read=format(row.cache_read_price_per_million_tokens, "f"),
                cache_write=format(row.cache_write_price_per_million_tokens, "f"),
            ),
        )
        for row in visible
    )
