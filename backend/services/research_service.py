from backend.agents.planner import plan
from backend.agents.researcher import research
from backend.agents.writer import write
from backend.services.memory import format_history_context, save_to_memory
from backend.services.document_ingestion import index_document
from backend.services.vector_store import VectorStore
import re


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


def _query_terms(query: str):
    terms = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
        "give", "me", "show", "find", "analyze", "best", "what", "which", "is",
        "are", "my", "about", "from", "that", "this", "please"
    }
    return [term for term in terms if term not in stopwords and len(term) > 2]


def _query_relevance_score(query: str, item):
    haystack = " ".join(
        [
            item.get("title", ""),
            item.get("content", ""),
            item.get("source_filename", ""),
            item.get("text", ""),
        ]
    ).lower()
    terms = _query_terms(query)
    if not terms:
        return 0.0

    score = 0.0
    for term in terms:
        if term in haystack:
            score += 1.0
            if item.get("title", "").lower().find(term) != -1:
                score += 0.75
    return score / max(1, len(terms))


def _rerank_for_query(query: str, items, limit=6):
    ranked = []
    for item in items:
        combined_score = float(item.get("score", 0.0)) + (1.5 * _query_relevance_score(query, item))
        enriched = dict(item)
        enriched["query_score"] = combined_score
        ranked.append(enriched)
    ranked.sort(key=lambda item: item.get("query_score", 0.0), reverse=True)
    return ranked[:limit]


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


def _make_query_focused_findings(query: str, retrieved_context):
    query_terms = _query_terms(query)
    findings = []

    for item in retrieved_context[:5]:
        title = item.get("title", "Untitled source")
        content = " ".join((item.get("content") or "").split())
        source_type = item.get("source_type", "web")
        label = "document" if source_type == "document" else "web"

        snippet = content
        for term in query_terms:
            idx = content.lower().find(term)
            if idx != -1:
                start = max(0, idx - 60)
                end = min(len(content), idx + 140)
                snippet = content[start:end].strip()
                break

        snippet = snippet[:180] + "..." if len(snippet) > 180 else snippet
        if snippet:
            findings.append(f"- **{title}** ({label}): {snippet}")
        if len(findings) >= 5:
            break

    return "\n".join(findings) if findings else "- No tightly matched findings were retrieved."


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


def _contains_report_style_language(text: str):
    lowered = (text or "").lower()
    banned_phrases = [
        "objective",
        "data collection",
        "analysis framework",
        "analysis",
        "framework",
        "implementation plan",
    ]
    return any(phrase in lowered for phrase in banned_phrases)


def _is_generic_response(query: str, final_report: str):
    query_terms = _query_terms(query)
    report_lower = (final_report or "").lower()
    if not query_terms:
        return False

    matches = sum(1 for term in query_terms if term in report_lower)
    return matches < max(1, min(2, len(query_terms)))


def _short_direct_answer(query: str, key_findings: str, final_report: str):
    findings_lines = [line.strip() for line in (key_findings or "").splitlines() if line.strip()]
    concise_findings = "\n".join(findings_lines[:3]) if findings_lines else "- Limited directly relevant findings were retrieved."
    return (
        "## Top Recommendations\n"
        f"{concise_findings}\n\n"
        "## Why These Recommendations\n"
        "- These options were prioritized because they align most closely with the user query and retrieved evidence.\n\n"
        "## Personalized Insight\n"
        f"- Answering: {query}"
    )


def _resume_query(query: str):
    lowered = (query or "").lower()
    triggers = ["resume", "my profile", "my skills"]
    return any(trigger in lowered for trigger in triggers)


def _has_resume_context(document_sources):
    for source in document_sources:
        filename = (source.get("source_filename") or "").lower()
        title = (source.get("title") or "").lower()
        if any(token in f"{filename} {title}" for token in ["resume", "cv", "profile"]):
            return True
    return bool(document_sources)


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
    merged_context = _merge_retrieval_results(web_context, document_context, limit=8)
    retrieved_context = _rerank_for_query(query, merged_context, limit=6)

    writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
    writer_payload["retrieved_context"] = retrieved_context
    writer_payload["user_query"] = query

    web_sources = _extract_sources(research_data)
    retrieved_web_sources, retrieved_document_sources = _split_sources_by_type(retrieved_context)
    web_sources = web_sources or retrieved_web_sources
    document_sources = retrieved_document_sources
    writer_payload["resume_query"] = _resume_query(query)
    writer_payload["has_resume_context"] = _has_resume_context(document_sources)

    final_report = write(
        plan_text,
        writer_payload,
        conversation_context=conversation_context,
    )
    key_findings = _make_query_focused_findings(query, retrieved_context)
    web_sources_markdown = _build_sources_markdown(web_sources, source_type="web")
    document_sources_markdown = _build_sources_markdown(document_sources, source_type="document")
    if _contains_report_style_language(final_report) or _is_generic_response(query, final_report):
        final_report = _short_direct_answer(query, key_findings, final_report)
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
