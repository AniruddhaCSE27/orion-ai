from backend.agents.planner import plan
from backend.agents.researcher import research
from backend.agents.writer import write
from backend.services.memory import format_history_context, save_to_memory
from backend.services.vector_store import VectorStore


def _extract_sources(research_data):
    if not isinstance(research_data, dict):
        return []

    sources = []
    for item in research_data.get("results", []):
        sources.append(
            {
                "title": item.get("title", "Untitled source"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )
    return sources


def _build_documents(research_data):
    if not isinstance(research_data, dict):
        return []

    documents = []
    for item in research_data.get("results", []):
        title = item.get("title", "Untitled source")
        content = item.get("content", "")
        url = item.get("url", "")
        text = f"{title}\n{content}".strip()

        if not text:
            continue

        documents.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "text": text,
            }
        )
    return documents


def run_research_pipeline(query: str):
    conversation_context = format_history_context(limit=5)
    plan_text = plan(query, conversation_context=conversation_context)
    research_data = research(plan_text)
    vector_store = VectorStore()
    documents = _build_documents(research_data)
    vector_store.add_documents(documents)

    retrieved_context = vector_store.similarity_search(
        query=f"{query}\n{plan_text}",
        top_k=3,
    )

    writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
    writer_payload["retrieved_context"] = retrieved_context

    final_text = write(
        plan_text,
        writer_payload,
        conversation_context=conversation_context,
    )
    sources = _extract_sources(research_data)
    save_to_memory(query, final_text)

    return {
        "success": True,
        "plan": plan_text,
        "findings": writer_payload,
        "final_report": final_text,
        "sources": sources,
        # Compatibility keys for the current frontend.
        "research": writer_payload,
        "final": final_text,
    }
