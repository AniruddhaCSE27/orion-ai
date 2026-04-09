from backend.agents.planner import plan
from backend.agents.researcher import research
from backend.agents.writer import write
from backend.services.memory import format_history_context, save_to_memory
from backend.services.document_ingestion import index_document
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
                "source_type": "web",
                "source_filename": "",
                "chunk_id": 0,
                "title": title,
                "url": url,
                "content": content,
                "text": text,
            }
        )
    return documents


def _merge_retrieval_results(web_results, document_results, limit=6):
    merged = list(web_results) + list(document_results)
    merged.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return merged[:limit]


def _split_sources_by_type(items):
    web_sources = []
    document_sources = []

    for item in items:
        source_type = item.get("source_type", "web")
        if source_type == "document":
            document_sources.append(
                {
                    "title": item.get("title") or item.get("source_filename", "Document"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "source_filename": item.get("source_filename", ""),
                    "chunk_id": item.get("chunk_id", 0),
                    "source_type": "document",
                }
            )
        else:
            web_sources.append(
                {
                    "title": item.get("title", "Untitled source"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "source_type": "web",
                }
            )
    return web_sources, document_sources


def _build_key_findings(retrieved_context, web_sources, document_sources):
    findings = []

    for item in retrieved_context[:4]:
        title = item.get("title", "Untitled source")
        content = (item.get("content") or "").strip()
        url = item.get("url", "").strip()
        source_type = item.get("source_type", "web")
        source_label = "document" if source_type == "document" else "web"
        snippet = content[:180] + "..." if len(content) > 180 else content

        if snippet and url:
            findings.append(f"- **{title}** ({source_label}): {snippet} ([source]({url}))")
        elif snippet and source_type == "document":
            filename = item.get("source_filename", title)
            findings.append(f"- **{title}** (document: {filename}): {snippet}")
        elif snippet:
            findings.append(f"- **{title}** ({source_label}): {snippet}")
        elif url:
            findings.append(f"- **{title}** ({source_label}): [source]({url})")

    if findings:
        return "\n".join(findings)

    fallback = []
    for source in web_sources[:2]:
        title = source.get("title", "Untitled source")
        url = source.get("url", "").strip()
        if url:
            fallback.append(f"- **{title}** (web): [source]({url})")
        else:
            fallback.append(f"- **{title}** (web)")
    for source in document_sources[:2]:
        title = source.get("title", "Document")
        filename = source.get("source_filename", title)
        fallback.append(f"- **{title}** (document: {filename})")
    return "\n".join(fallback) if fallback else "- No source-backed findings available."


def _build_sources_markdown(sources, source_type="web"):
    if not sources:
        return "- No sources available."

    lines = []
    for source in sources:
        title = source.get("title", "Untitled source")
        url = source.get("url", "").strip()
        if source_type == "document":
            filename = source.get("source_filename", title)
            chunk_id = source.get("chunk_id", 0)
            lines.append(f"- **{filename}** (chunk {chunk_id})")
        elif url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def _build_structured_response(plan_text, key_findings, final_report, web_sources_markdown, document_sources_markdown):
    return (
        "## Research Plan\n"
        f"{plan_text}\n\n"
        "## Key Findings\n"
        f"{key_findings}\n\n"
        "## Final Report\n"
        f"{final_report}\n\n"
        "## Web Sources\n"
        f"{web_sources_markdown}\n\n"
        "## Document Sources\n"
        f"{document_sources_markdown}"
    )


def run_research_pipeline(query: str):
    conversation_context = format_history_context(limit=5)
    plan_text = plan(query, conversation_context=conversation_context)
    research_data = research(plan_text)
    web_store = VectorStore(namespace="web")
    document_store = VectorStore(namespace="documents")
    documents = _build_documents(research_data)
    web_store.add_documents(documents)

    web_context = web_store.similarity_search(
        query=f"{query}\n{plan_text}",
        top_k=4,
    )
    document_context = document_store.similarity_search(
        query=f"{query}\n{plan_text}",
        top_k=4,
    )
    retrieved_context = _merge_retrieval_results(web_context, document_context, limit=6)

    writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
    writer_payload["retrieved_context"] = retrieved_context

    final_report = write(
        plan_text,
        writer_payload,
        conversation_context=conversation_context,
    )
    web_sources = _extract_sources(research_data)
    retrieved_web_sources, retrieved_document_sources = _split_sources_by_type(retrieved_context)
    web_sources = web_sources or retrieved_web_sources
    document_sources = retrieved_document_sources
    key_findings = _build_key_findings(retrieved_context, web_sources, document_sources)
    web_sources_markdown = _build_sources_markdown(web_sources, source_type="web")
    document_sources_markdown = _build_sources_markdown(document_sources, source_type="document")
    structured_response = _build_structured_response(
        plan_text,
        key_findings,
        final_report,
        web_sources_markdown,
        document_sources_markdown,
    )
    save_to_memory(query, structured_response)

    return {
        "success": True,
        "plan": plan_text,
        "findings": key_findings,
        "key_findings": key_findings,
        "final_report": final_report,
        "structured_response": structured_response,
        "sources": web_sources + document_sources,
        "web_sources": web_sources,
        "document_sources": document_sources,
        # Compatibility keys for the current frontend.
        "research": writer_payload,
        "final": final_report,
    }


def index_document_file(file_path):
    return index_document(file_path)
