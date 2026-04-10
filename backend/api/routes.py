import logging

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from backend.services.memory import clear_memory
from backend.services.research_service import run_research_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)


def _error_payload(error: str, details: str | None = None) -> dict:
    payload = {
        "success": False,
        "error": error,
    }
    if details:
        payload["details"] = details
    return payload


def _json_response(endpoint_name: str, payload: dict, status_code: int = 200) -> JSONResponse:
    encoded_payload = jsonable_encoder(payload)
    logger.info(
        "endpoint=%s success=%s response_keys=%s",
        endpoint_name,
        bool(encoded_payload.get("success", False)),
        sorted(encoded_payload.keys()),
    )
    return JSONResponse(status_code=status_code, content=encoded_payload)


@router.get("/")
def home():
    try:
        return _json_response(
            "home",
            {
                "success": True,
                "status": "Backend is running",
                "product": "ORION AI",
                "capabilities": ["web_rag", "memory", "multi_agent"],
            },
        )
    except Exception as exc:
        logger.exception("endpoint=home failed")
        return _json_response(
            "home",
            _error_payload("Backend status check failed.", str(exc)),
            status_code=500,
        )


@router.get("/health")
def health():
    try:
        return _json_response(
            "health",
            {
                "success": True,
                "status": "ok",
            },
        )
    except Exception as exc:
        logger.exception("endpoint=health failed")
        return _json_response(
            "health",
            _error_payload("Health check failed.", str(exc)),
            status_code=500,
        )


@router.post("/research")
def run_research(query: str):
    try:
        payload = run_research_pipeline(query)
        status_code = 200 if payload.get("success", False) else 500
        return _json_response("run_research", payload, status_code=status_code)
    except Exception as exc:
        logger.exception("endpoint=run_research failed")
        return _json_response(
            "run_research",
            _error_payload("Research execution failed.", str(exc)),
            status_code=500,
        )


@router.post("/memory/clear")
def clear_research_memory():
    try:
        clear_memory()
        return _json_response(
            "clear_research_memory",
            {
                "success": True,
                "message": "Memory cleared successfully.",
            },
        )
    except Exception as exc:
        logger.exception("endpoint=clear_research_memory failed")
        return _json_response(
            "clear_research_memory",
            _error_payload("Failed to clear memory.", str(exc)),
            status_code=500,
        )
