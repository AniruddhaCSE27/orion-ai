import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import get_current_user
from app.config import get_settings
from app.dependencies import get_model_artifact, get_request_id
from app.rate_limiter import limiter
from app.schema import PredictionRequest, PredictionResponse
from app.utils import build_prediction_response


logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit(get_settings().predict_rate_limit)
async def predict(
    request: Request,
    payload: PredictionRequest,
    current_user: str = Depends(get_current_user),
    artifact: Dict[str, Any] = Depends(get_model_artifact),
    request_id: str = Depends(get_request_id),
) -> PredictionResponse:
    try:
        payload_dict = payload.model_dump()
        logger.info("Prediction requested by username=%s", current_user)
        response = build_prediction_response(artifact, payload_dict, request_id)
        return PredictionResponse(**response)
    except ValueError as exc:
        logger.warning(
            "Prediction validation failed for username=%s error=%s",
            current_user,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prediction request could not be processed.",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed for username=%s error=%s", current_user, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from exc
