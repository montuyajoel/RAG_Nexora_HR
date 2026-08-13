import secrets

from fastapi import (
    HTTPException,
    Security,
)

from fastapi.security import APIKeyHeader

from config import API_KEY


if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is required."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(
    api_key: str = Security(api_key_header),
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