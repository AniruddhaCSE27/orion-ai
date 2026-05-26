import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


logger = logging.getLogger(__name__)


def create_limiter() -> Limiter:
    return Limiter(key_func=get_remote_address, headers_enabled=False)


limiter = create_limiter()


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.warning(
        "Rate limit exceeded for path=%s client=%s detail=%s",
        request.url.path,
        request.client.host if request.client else "unknown",
        exc.detail,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please retry later.",
            "path": request.url.path,
            "request_id": request_id,
        },
    )
