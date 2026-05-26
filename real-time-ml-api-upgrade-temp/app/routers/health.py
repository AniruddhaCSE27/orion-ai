from fastapi import APIRouter, Request

from app.config import get_settings
from app.model_loader import is_model_ready
from app.schema import HealthResponse


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def liveness_probe() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_probe(request: Request) -> HealthResponse:
    settings = get_settings()
    model_loaded = is_model_ready(getattr(request.app.state, "model_artifact", None))
    status_value = "ok" if model_loaded else "degraded"
    return HealthResponse(
        status=status_value,
        service_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        model_loaded=model_loaded,
    )

