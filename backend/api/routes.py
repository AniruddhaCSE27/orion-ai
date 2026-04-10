import importlib.util
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from backend.services.document_ingestion import save_uploaded_file
from backend.services.memory import clear_memory
from backend.services.research_service import index_document_file, run_research_pipeline

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
    return _json_response("home", {"success": True, "status": "Backend is running"})


@router.post("/research")
def run_research(query: str):
    try:
        payload = run_research_pipeline(query)
        return _json_response("run_research", payload)
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


@router.post("/documents/index")
async def index_documents(request: Request):
    try:
        if importlib.util.find_spec("multipart") is None:
            return _json_response(
                "index_documents",
                _error_payload("Document uploads require the 'python-multipart' package."),
                status_code=500,
            )

        form = await request.form()
        files = form.getlist("files")
        indexed_files = []
        failed_files = []

        for upload in files:
            if not getattr(upload, "filename", ""):
                continue

            suffix = Path(upload.filename).suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                return _json_response(
                    "index_documents",
                    _error_payload(f"Unsupported file type: {upload.filename}"),
                    status_code=400,
                )

            content = await upload.read()
            saved_path = save_uploaded_file(upload.filename, content)
            file_result = index_document_file(saved_path)
            if file_result.get("success", False):
                indexed_files.append(file_result)
            else:
                failed_files.append(file_result)

        if failed_files and not indexed_files:
            first_failure = failed_files[0]
            return _json_response(
                "index_documents",
                _error_payload(
                    first_failure.get("error", "Document indexing failed."),
                    first_failure.get("details"),
                ),
                status_code=500,
            )

        if not indexed_files:
            return _json_response(
                "index_documents",
                _error_payload("No supported files were provided."),
                status_code=400,
            )

        payload = {
            "success": True,
            "message": "Documents indexed successfully.",
            "files": indexed_files,
            "pages_read": sum(int(item.get("pages_read", 0) or 0) for item in indexed_files),
            "characters_extracted": sum(int(item.get("characters_extracted", 0) or 0) for item in indexed_files),
            "chunks_created": sum(int(item.get("chunks_created", 0) or 0) for item in indexed_files),
            "indexed_document_count": sum(int(item.get("indexed_document_count", 0) or 0) for item in indexed_files),
            "document_chunks_indexed": sum(int(item.get("chunks_created", 0) or 0) for item in indexed_files),
        }
        if failed_files:
            payload["file_errors"] = failed_files
        return _json_response("index_documents", payload)
    except Exception as exc:
        logger.exception("endpoint=index_documents failed")
        return _json_response(
            "index_documents",
            _error_payload("Document indexing failed.", str(exc)),
            status_code=500,
        )
