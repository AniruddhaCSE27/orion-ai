import json
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[1] / "data"
MEMORY_PATH = DB_DIR / "memory.json"
MAX_INTERACTIONS = 10


def _ensure_storage():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("[]", encoding="utf-8")


def _load_history():
    _ensure_storage()
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = []
    return data if isinstance(data, list) else []


def _save_history(history):
    _ensure_storage()
    MEMORY_PATH.write_text(
        json.dumps(history[-MAX_INTERACTIONS:], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def save_to_memory(query: str, report: str):
    history = _load_history()
    history.append({"query": query, "report": report})
    _save_history(history)


def get_recent_history(limit: int = 5):
    safe_limit = max(1, min(limit, MAX_INTERACTIONS))
    history = _load_history()
    return history[-safe_limit:]


def format_history_context(limit: int = 5) -> str:
    history = get_recent_history(limit=limit)
    if not history:
        return ""

    sections = []
    for idx, item in enumerate(history, start=1):
        sections.append(
            (
                f"Interaction {idx}\n"
                f"Query: {item['query']}\n"
                f"Response: {item['report']}"
            )
        )
    return "\n\n".join(sections)


def clear_memory():
    _save_history([])
