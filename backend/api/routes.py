from fastapi import APIRouter

from backend.services.memory import clear_memory
from backend.services.research_service import run_research_pipeline

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
