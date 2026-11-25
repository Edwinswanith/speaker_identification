from __future__ import annotations

import json
from hmac import compare_digest

from fastapi import Depends, Header, HTTPException, status
from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


def configured_clients() -> dict[str, str]:
    """Return client-token -> tenant mapping from tenant-name -> token JSON."""
    try:
        value = json.loads(settings.api_keys_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("API_KEYS_JSON must be valid JSON mapping tenant IDs to tokens") from exc
    if not isinstance(value, dict):
        raise RuntimeError("API_KEYS_JSON must be a JSON object mapping tenant IDs to tokens")
    result: dict[str, str] = {}
    for tenant, token in value.items():
        if not isinstance(tenant, str) or not tenant.strip() or not isinstance(token, str) or len(token) < 32:
            raise RuntimeError("API_KEYS_JSON must map each non-empty tenant ID to a token of at least 32 characters")
        if token in result:
            raise RuntimeError("Each API token must belong to only one tenant")
        result[token] = tenant.strip()
    return result


def get_tenant(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer API token")
    token = authorization[7:].strip()
    for configured, tenant_id in configured_clients().items():
        if compare_digest(token, configured):
            return tenant_id
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


def fernet() -> Fernet:
    key = settings.embedding_encryption_key.strip()
    if not key:
        raise RuntimeError("EMBEDDING_ENCRYPTION_KEY must be configured before the API starts")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("EMBEDDING_ENCRYPTION_KEY must be a valid Fernet key") from exc


def encrypt_embedding(vector_bytes: bytes) -> str:
    return fernet().encrypt(vector_bytes).decode("ascii")


def decrypt_embedding(ciphertext: str) -> bytes:
    try:
        return fernet().decrypt(ciphertext.encode("ascii"))
    except (InvalidToken, UnicodeEncodeError) as exc:
        raise RuntimeError("Stored voice reference cannot be decrypted; check the configured encryption key") from exc
