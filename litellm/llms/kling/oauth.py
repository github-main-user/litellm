import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field, SecretStr

RESOURCE: Final = "https://kling.ai/mcp"
ISSUER: Final = "https://kling.ai/auth"
SCOPES: Final = "generation.create generation.read account.credit.read"


class Registration(BaseModel):
    client_id: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: SecretStr = Field(min_length=1)
    refresh_token: SecretStr | None = Field(default=None, min_length=1)
    expires_in: int = Field(gt=0)


class TokenBundle(BaseModel):
    client_id: str
    access_token: str = Field(repr=False, min_length=1)
    refresh_token: str = Field(repr=False, min_length=1)
    expires_at: float


@dataclass(frozen=True, slots=True)
class Authorization:
    url: str
    state: str = field(repr=False)
    verifier: str = field(repr=False)


def authorize(client_id: str, redirect_uri: str) -> Authorization:
    verifier: Final = secrets.token_urlsafe(64)
    digest: Final = hashlib.sha256(verifier.encode("ascii")).digest()
    state: Final = secrets.token_urlsafe(32)
    params: Final = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii"),
        }
    )
    return Authorization(f"{ISSUER}/authorize?{params}", state, verifier)


async def register(client: httpx.AsyncClient, redirect_uri: str) -> str:
    response: Final = await client.post(
        f"{ISSUER}/register",
        json={
            "client_name": "Void",
            "redirect_uris": [redirect_uri],
            "response_types": ["code"],
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": "none",
        },
    )
    response.raise_for_status()
    return Registration.model_validate_json(response.content).client_id


async def _grant(
    client: httpx.AsyncClient,
    client_id: str,
    grant: Literal["authorization_code", "refresh_token"],
    parameters: dict[str, str],
    previous: str | None,
) -> TokenBundle:
    response: Final = await client.post(
        f"{ISSUER}/token", data={**parameters, "client_id": client_id, "grant_type": grant, "resource": RESOURCE}
    )
    response.raise_for_status()
    result: Final = TokenResponse.model_validate_json(response.content)
    refresh_token: Final = result.refresh_token.get_secret_value() if result.refresh_token else previous
    if not refresh_token:
        raise ValueError("Kling did not issue a refresh token")
    return TokenBundle(
        client_id=client_id,
        access_token=result.access_token.get_secret_value(),
        refresh_token=refresh_token,
        expires_at=time.time() + result.expires_in,
    )


async def exchange(
    client: httpx.AsyncClient, client_id: str, code: str, verifier: str, redirect_uri: str
) -> TokenBundle:
    return await _grant(
        client,
        client_id,
        "authorization_code",
        {"code": code, "code_verifier": verifier, "redirect_uri": redirect_uri},
        None,
    )


async def refresh(client: httpx.AsyncClient, current: TokenBundle) -> TokenBundle:
    return await _grant(
        client, current.client_id, "refresh_token", {"refresh_token": current.refresh_token}, current.refresh_token
    )
