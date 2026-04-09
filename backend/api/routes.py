import importlib.util
from pathlib import Path

from fastapi import APIRouter, Request

from backend.services.memory import clear_memory
from backend.services.document_ingestion import save_uploaded_file
from backend.services.research_service import index_document_file, run_research_pipeline

router = APIRouter()


@router.get("/")
def home():
    return {"status": "Backend is running 🚀"}


@router.post("/research")
def run_research(query: str):
    try:
        return run_research_pipeline(query)
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/memory/clear")
def clear_research_memory():
    clear_memory()
    return {
        "success": True,
        "message": "Memory cleared successfully."
    }


@router.post("/documents/index")
async def index_documents(request: Request):
    if importlib.util.find_spec("multipart") is None:
        return {
            "success": False,
            "error": "Document uploads require the 'python-multipart' package."
        }

    form = await request.form()
    files = form.getlist("files")
    indexed_files = []

    for upload in files:
        if not getattr(upload, "filename", ""):
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            return {
                "success": False,
                "error": f"Unsupported file type: {upload.filename}"
            }

        content = await upload.read()
        saved_path = save_uploaded_file(upload.filename, content)
        indexed_files.append(index_document_file(saved_path))

    if not indexed_files:
        return {
            "success": False,
            "error": "No supported files were provided."
        }

    return {
        "success": True,
        "message": "Documents indexed successfully.",
        "files": indexed_files,
        "document_chunks_indexed": sum(item.get("chunks_indexed", 0) for item in indexed_files),
    }
