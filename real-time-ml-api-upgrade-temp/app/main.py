import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware import add_application_middleware
from app.model_loader import load_model_artifact
from app.rate_limiter import limiter, rate_limit_exceeded_handler
from app.routers import auth as auth_router
from app.routers import health as health_router
from app.routers import inference as inference_router
from app.schema import ServiceMetadataResponse


configure_logging()
logger = logging.getLogger(__name__)


def create_lifespan(settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model_artifact = load_model_artifact(settings.model_path)
            logger.info(
                "Application startup complete for service=%s version=%s environment=%s",
                settings.app_name,
                settings.app_version,
                settings.environment,
            )
            yield
        finally:
            logger.info("Application shutdown complete for service=%s", settings.app_name)

    return lifespan


def create_app() -> FastAPI:
    settings = get_settings()
    docs_url = None if settings.is_production else "/docs"
    redoc_url = None if settings.is_production else "/redoc"
    openapi_url = None if settings.is_production else "/openapi.json"

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=create_lifespan(settings),
    )

    app.state.limiter = limiter
    add_application_middleware(app, settings)

    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning("Validation error on path=%s errors=%s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid request payload.",
                "errors": exc.errors(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled application error on path=%s error=%s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Please try again later.",
                "request_id": request_id,
            },
        )

    @app.get("/", response_model=ServiceMetadataResponse, tags=["service"])
    async def root() -> ServiceMetadataResponse:
        return ServiceMetadataResponse(
            service_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            docs_enabled=not settings.is_production,
        )

    app.include_router(health_router.router)
    app.include_router(auth_router.router)
    app.include_router(inference_router.router)

    return app


app = create_app()
