import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from app.auth import create_access_token, verify_credentials
from app.config import Settings, get_settings
from app.rate_limiter import limiter
from app.schema import TokenResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
@limiter.limit(get_settings().auth_rate_limit)
async def issue_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    if not verify_credentials(form_data.username, form_data.password, settings):
        logger.warning("Authentication failed for username=%s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("Authentication succeeded for username=%s", form_data.username)
    access_token = create_access_token(
        subject=form_data.username,
        settings=settings,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )
    return TokenResponse(
        access_token=access_token,
        expires_in_minutes=settings.access_token_expire_minutes,
    )
