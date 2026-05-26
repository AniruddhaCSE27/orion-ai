from typing import Any, Dict

from fastapi import HTTPException, Request, status


def get_model_artifact(request: Request) -> Dict[str, Any]:
    artifact = getattr(request.app.state, "model_artifact", None)
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return artifact


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")

