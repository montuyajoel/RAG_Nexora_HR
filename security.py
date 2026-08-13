import secrets

from fastapi import (
    HTTPException,
    Security,
)

from fastapi.security import APIKeyHeader

from config import (
    API_KEY,
    ADMIN_API_KEY,
)


if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is required."
    )

if not ADMIN_API_KEY:
    raise RuntimeError(
        "ADMIN_API_KEY environment variable is required."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


admin_api_key_header = APIKeyHeader(
    name="X-Admin-API-Key",
    auto_error=False,
)


def verify_api_key(
    api_key: str = Security(
        api_key_header
    ),
):
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required.",
        )

    if not secrets.compare_digest(
        api_key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


def verify_admin_api_key(
    api_key: str = Security(
        admin_api_key_header
    ),
):
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Admin API key required.",
        )

    if not secrets.compare_digest(
        api_key,
        ADMIN_API_KEY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid admin API key.",
        )

    return api_key