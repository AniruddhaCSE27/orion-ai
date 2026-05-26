import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schema import PredictionRequest
from app.utils import build_prediction_response, load_model_artifact, prepare_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("real_time_ml_api")

model_artifact: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_artifact
    try:
        model_artifact = load_model_artifact()
        logger.info("Application startup complete. Model is ready for predictions.")
        yield
    except Exception:
        logger.exception("Application startup failed while loading the model artifact.")
        raise
    finally:
        logger.info("Application shutdown complete.")


app = FastAPI(
    title="Real-Time ML Prediction API",
    description="Production-ready FastAPI service for real-time binary classification.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error received: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(_, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    return {
        "message": "Welcome to the Real-Time ML Prediction API System.",
        "model_type": "Binary classification",
    }


@app.get("/health", tags=["General"])
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", tags=["Inference"])
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    if model_artifact is None:
        logger.error("Prediction attempted before the model artifact was loaded.")
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        payload = request.model_dump()
        logger.info("Prediction request received.")
        features = prepare_features(payload, model_artifact)
        response = build_prediction_response(model_artifact, features)
        logger.info("Prediction completed successfully.")
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
