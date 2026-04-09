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


def _mode_label(mode_key: str):
    return {
        "resume": "Resume Analyzer",
        "study": "Study Mode",
        "document": "Document Analysis",
        "hybrid_document": "Document Analysis",
        "interview": "Interview Prep",
        "web": "Web Research",
        "general": "General Mode",
    }.get(mode_key, "General Mode")


def _needs_explicit_web_mode(query: str):
    lowered = (query or "").lower()
    web_markers = [
        "latest", "current", "today", "recent", "trends", "compare", "comparison",
        "best tools", "news", "2026", "updates", "latest news", "current trends",
    ]
    return any(marker in lowered for marker in web_markers)


def _classify_query_intent(query: str, has_indexed_documents: bool):
    lowered = (query or "").lower()
    resume_markers = [
        "resume", "my profile", "my skills", "best roles", "best jobs", "career",
        "job", "roles", "position", "fit for", "suitable roles", "career options"
    ]
    study_markers = [
        "upsc", "chapter", "exam", "2 mark", "5 mark", "10 mark", "question",
        "questions", "qs", "important topics", "short answer", "long answer",
        "revision", "summarize this chapter", "explain this topic"
    ]
    interview_markers = [
        "interview", "mock interview", "viva", "interview questions",
        "prepare me for interview", "ask me interview questions"
    ]
    document_markers = [
        "this", "from this", "uploaded", "document", "pdf", "notes", "file",
        "summarize this", "explain this", "from the document", "summarize the chapter",
        "explain the chapter", "what questions", "what can come from this"
    ]

    if any(marker in lowered for marker in interview_markers):
        return "interview"
    if any(marker in lowered for marker in resume_markers):
        return "resume"
    if any(marker in lowered for marker in study_markers):
        return "study"
    if _needs_explicit_web_mode(query):
        return "web"
    if has_indexed_documents and any(marker in lowered for marker in document_markers):
        return "document"
    if has_indexed_documents:
        return "hybrid_document"
    return "general"


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


def _mode_source_boost(mode_key: str, item, query: str):
    source_type = item.get("source_type", "web")
    filename = (item.get("source_filename") or "").lower()
    title = (item.get("title") or "").lower()
    combined_name = f"{filename} {title}"
    lowered_query = (query or "").lower()
    score = 0.0

    if source_type == "document":
        score += 0.5
    if mode_key in {"study", "document", "hybrid_document"} and source_type == "document":
        score += 2.0
    if mode_key == "resume" and source_type == "document":
        score += 1.5
        if any(token in combined_name for token in ["resume", "cv", "profile"]):
            score += 2.0
    if mode_key == "interview" and source_type == "document":
        score += 1.0
        if any(token in combined_name for token in ["resume", "cv", "profile"]):
            score += 1.0
    if mode_key == "web" and source_type == "web":
        score += 2.0
    if any(token in lowered_query for token in ["latest", "current", "recent", "today"]) and source_type == "web":
        score += 1.5

    return score


def _rerank_for_query(query: str, items, limit=6, mode_key: str = "general"):
    ranked = []
    for item in items:
        combined_score = (
            float(item.get("score", 0.0))
            + (1.5 * _query_relevance_score(query, item))
            + _mode_source_boost(mode_key, item, query)
        )
        enriched = dict(item)
        enriched["query_score"] = combined_score
        ranked.append(enriched)
    ranked.sort(key=lambda item: item.get("query_score", 0.0), reverse=True)
    return ranked[:limit]


def _prioritize_for_intent(query_type: str, items):
    prioritized = []
    for item in items:
        boosted = dict(item)
        score = float(boosted.get("query_score", boosted.get("score", 0.0)))
        if boosted.get("source_type") == "document":
            score += 1.0
        if query_type in {"study", "document", "hybrid_document"} and boosted.get("source_type") == "document":
            score += 2.0
        if query_type == "resume" and boosted.get("source_type") == "document":
            score += 1.5
        if query_type == "interview" and boosted.get("source_type") == "document":
            score += 1.0
        if query_type == "web" and boosted.get("source_type") == "web":
            score += 0.5
        boosted["query_score"] = score
        prioritized.append(boosted)
    prioritized.sort(key=lambda item: item.get("query_score", 0.0), reverse=True)
    return prioritized


def _has_strong_document_match(query: str, items):
    if not items:
        return False

    query_terms = _query_terms(query)
    if not query_terms:
        return bool(items)

    best_score = 0.0
    for item in items:
        score = _query_relevance_score(query, item)
        best_score = max(best_score, score)
        if score >= 0.34:
            return True

    return best_score >= 0.2 and len(items) >= 2


def _document_query_for_mode(query: str, mode_key: str, retrieval_query: str):
    if mode_key == "resume":
        return f"{query}\nresume profile skills projects experience"
    if mode_key == "interview":
        return f"{query}\ninterview questions viva project experience"
    if mode_key in {"study", "document", "hybrid_document"}:
        return query
    return retrieval_query


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


def _is_document_oriented_query(query: str):
    lowered = (query or "").lower()
    markers = [
        "summarize", "explain", "what questions", "from this", "chapter",
        "this document", "this pdf", "notes", "uploaded file", "what can come from this",
    ]
    return any(marker in lowered for marker in markers)


def _answer_payload_to_markdown(answer_payload):
    recommendations = answer_payload.get("recommendations", [])
    reasons = answer_payload.get("reasons", [])
    insights = answer_payload.get("insights", "")
    improvement_tips = answer_payload.get("improvement_tips", [])
    primary_title = answer_payload.get("primary_title", "Direct Answer")
    reasons_title = answer_payload.get("reasons_title", "Why This Answer")
    insights_title = answer_payload.get("insights_title", "Insight")
    improvement_title = answer_payload.get("improvement_title", "Improvement Tips")
    extra_sections = answer_payload.get("extra_sections", [])

    sections = [
        f"## {primary_title}\n" + "\n".join(f"- {item}" for item in recommendations),
        f"## {reasons_title}\n" + "\n".join(f"- {item}" for item in reasons),
        f"## {insights_title}\n" + (insights or "- This answer stays aligned to the query."),
    ]

    if improvement_tips:
        sections.append(f"## {improvement_title}\n" + "\n".join(f"- {item}" for item in improvement_tips))

    for section in extra_sections:
        title = section.get("title", "").strip()
        items = section.get("items", [])
        if title and items:
            sections.append(f"## {title}\n" + "\n".join(f"- {item}" for item in items))

    return "\n\n".join(sections)


def _contains_report_style_language(text: str):
    lowered = (text or "").lower()
    banned_phrases = [
        "objective",
        "data collection",
        "analysis framework",
        "framework",
        "implementation plan",
    ]
    return any(phrase in lowered for phrase in banned_phrases)


def _is_generic_response(query: str, final_report: str, query_type: str = "general"):
    if query_type in {"study", "document", "resume", "interview"}:
        return False

    query_terms = _query_terms(query)
    report_lower = (final_report or "").lower()
    if not query_terms:
        return False

    matches = sum(1 for term in query_terms if term in report_lower)
    return matches < max(1, min(2, len(query_terms)))


def _fallback_answer_payload(query: str):
    return _fallback_answer_payload_by_type(query, "general")


def _fallback_answer_payload_by_type(query: str, query_type: str):
    if query_type == "resume":
        return {
            "primary_title": "Best Roles for You",
            "recommendations": [
                "Product-focused software roles",
                "Data-oriented analyst roles",
                "Automation-focused implementation roles",
            ],
            "reasons_title": "Why These Roles",
            "reasons": [
                "These suggestions stay useful when profile context is limited but the query is still career-oriented.",
                "They map broadly to technical, analytical, and implementation-heavy strengths.",
                "They can be refined further once resume-specific evidence is stronger.",
            ],
            "insights_title": "Strengths in Your Profile",
            "insights": "Based on general assumptions, here are the best roles.",
            "improvement_title": "Improvement Suggestions",
            "improvement_tips": [
                "Add measurable impact to projects and experience bullets.",
                "Tailor the profile to one role family at a time.",
            ],
            "extra_sections": [
                {
                    "title": "Skill Gaps",
                    "items": [
                        "Clarify one primary target role or stack.",
                        "Show stronger evidence of outcomes and ownership.",
                    ],
                },
                {
                    "title": "Suggested Next Steps",
                    "items": [
                        "Improve the resume around one clear role direction.",
                        "Prepare concise stories around your strongest projects.",
                    ],
                },
            ],
        }
    if query_type == "study":
        return {
            "primary_title": "Chapter Summary",
            "recommendations": [
                "Important topics could not be extracted confidently from the uploaded study material yet.",
                "Probable exam questions need a cleaner document index or a more specific chapter prompt.",
                "Revision output should be retried after re-indexing the file.",
            ],
            "reasons_title": "Important Topics",
            "reasons": [
                "This appears to be a document-study query and should be answered from uploaded material first.",
                "The current document retrieval was not strong enough to generate specific chapter-based questions safely.",
                "A more specific prompt or re-indexed file should improve the answer quality.",
            ],
            "insights_title": "Revision Points",
            "insights": "Document not properly indexed. Try re-indexing.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [
                {"title": "2-mark Questions", "items": ["Please re-index the file or ask a narrower chapter question."]},
                {"title": "5-mark Questions", "items": ["Please re-index the file or ask a narrower chapter question."]},
                {"title": "10-mark Questions", "items": ["Please re-index the file or ask a narrower chapter question."]},
            ],
        }
    if query_type == "document":
        return {
            "primary_title": "Summary",
            "recommendations": [
                "The uploaded document needs stronger indexing before a reliable summary can be generated.",
                "The answer should be retried after re-indexing or with a more specific question.",
                "Document-grounded questions will work best once chunk retrieval improves.",
            ],
            "reasons_title": "Important Topics",
            "reasons": [
                "This query depends on uploaded material rather than generic web context.",
                "The current retrieval was too weak to safely claim document-specific details.",
                "A cleaner index or narrower prompt should improve the next answer.",
            ],
            "insights_title": "Key Definitions / Concepts",
            "insights": "Document not properly indexed. Try re-indexing.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [
                {"title": "Probable Questions", "items": ["Please re-index the file or ask a more specific question."]},
                {"title": "Short Answer Questions", "items": ["Please re-index the file or ask a more specific question."]},
                {"title": "Long Answer Questions", "items": ["Please re-index the file or ask a more specific question."]},
            ],
        }
    if query_type == "interview":
        return {
            "primary_title": "Top Interview Questions",
            "recommendations": [
                "Tell me about your most relevant project and your exact contribution.",
                "What trade-offs did you make in your implementation?",
                "How would you improve the solution if you rebuilt it today?",
            ],
            "reasons_title": "Focus Areas",
            "reasons": [
                "These questions test clarity, ownership, and technical depth.",
                "They work as a practical fallback even when profile context is limited.",
                "They can be expanded into a mock interview quickly.",
            ],
            "insights_title": "Difficulty Level",
            "insights": "Start with foundational questions, then move into project depth and follow-up pressure questions.",
            "improvement_title": "Follow-up Questions",
            "improvement_tips": [
                "How would you measure success?",
                "What would you do differently next time?",
            ],
            "extra_sections": [
                {
                    "title": "Short Model Answers",
                    "items": [
                        "Use one project example with your role, action, and outcome.",
                        "Keep answers concrete and tied to tools, decisions, and impact.",
                    ],
                }
            ],
        }
    if query_type == "web":
        return {
            "primary_title": "Top Recommendations",
            "recommendations": [
                "Most relevant direct answer based on current web evidence",
                "Most useful comparison point from retrieved sources",
                "Most practical next step based on the evidence",
            ],
            "reasons_title": "Why These",
            "reasons": [
                "This is a web-oriented query, so current evidence matters more than stored document context.",
                "The fallback stays concise and useful without drifting into unrelated recommendations.",
                "The answer still prioritizes direct usefulness over report structure.",
            ],
            "insights_title": "Quick Insights",
            "insights": f"This fallback stays aligned to the web-oriented query: {query}",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [],
        }
    return {
        "primary_title": "Direct Answer",
        "recommendations": [
            "Refine the question to target the exact decision or topic you want answered",
            "Focus on the most relevant retrieved evidence instead of broad background",
            "Use a narrower prompt for a more precise answer",
        ],
        "reasons_title": "Why This Answer",
        "reasons": [
            "This is a general research query rather than a career or document-specific request.",
            "A narrower query usually improves precision and reduces generic output.",
            "The fallback stays query-aware and avoids unrelated recommendations.",
        ],
        "insights_title": "Personalized Insight",
        "insights": f"This fallback path keeps the answer aligned to the general query: {query}",
        "improvement_title": "Improvement Tips",
        "improvement_tips": [],
        "extra_sections": [],
    }


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
    return False


def run_research_pipeline(query: str):
    conversation_context = format_history_context(limit=5)
    document_store = VectorStore(namespace="documents")
    indexed_doc_chunks_count = len(document_store.documents)
    has_indexed_documents = indexed_doc_chunks_count > 0
    document_oriented_query = _is_document_oriented_query(query)
    query_type = _classify_query_intent(query, has_indexed_documents)
    mode = _mode_label(query_type)
    has_docs = has_indexed_documents
    if has_docs and query_type == "general":
        query_type = "hybrid_document"
        mode = _mode_label(query_type)
    print(f"QUERY TYPE DETECTED: {query_type}")
    print(f"HAS DOCS: {has_docs}")
    print(f"DOC CHUNKS COUNT: {indexed_doc_chunks_count}")
    print(f"MODE DETECTED: {mode}")
    plan_text = plan(query, conversation_context=conversation_context, query_type=query_type)
    research_data = research(plan_text)
    web_store = VectorStore(namespace="web")
    documents = _build_documents(research_data)
    web_store.add_documents(documents)

    retrieval_query = f"{query}\n{plan_text}"
    if has_indexed_documents:
        document_query = _document_query_for_mode(query, query_type, retrieval_query)
        document_context = document_store.similarity_search(query=document_query, top_k=8)
        document_context = _rerank_for_query(query, document_context, limit=6, mode_key=query_type)
        strong_document_match = _has_strong_document_match(query, document_context)
    else:
        document_context = []
        strong_document_match = False

    if query_type in {"study", "document"} and has_indexed_documents:
        if strong_document_match:
            web_context = web_store.similarity_search(query=query, top_k=2)
            merged_context = _merge_retrieval_results(web_context, document_context, limit=8)
        else:
            web_context = []
            merged_context = []
    elif has_indexed_documents:
        web_top_k = 6 if query_type == "web" else 3
        web_query = query if query_type == "web" else retrieval_query
        web_context = web_store.similarity_search(query=web_query, top_k=web_top_k)
        merged_context = _merge_retrieval_results(web_context, document_context, limit=8)
    else:
        web_query = query if query_type == "web" else retrieval_query
        web_context = web_store.similarity_search(query=web_query, top_k=6 if query_type == "web" else 4)
        merged_context = list(web_context)

    print(f"DOCUMENT CHUNKS RETRIEVED: {len(document_context)}")
    print(f"WEB CHUNKS RETRIEVED: {len(web_context)}")
    print(f"DOCUMENT RETRIEVAL FOUND CHUNKS: {bool(document_context)}")
    print(f"DOCUMENT RETRIEVAL STRONG MATCH: {strong_document_match}")
    document_retrieval_failed = bool(has_indexed_documents and query_type in {"study", "document"} and not strong_document_match)
    document_not_indexed = bool(document_oriented_query and not has_indexed_documents)
    print(f"RETRIEVED DOCUMENT CHUNKS COUNT: {len(document_context)}")
    print(f"DOCUMENT RETRIEVAL FAILED: {document_retrieval_failed}")
    print(f"DOCUMENT NOT INDEXED: {document_not_indexed}")

    retrieved_context = _rerank_for_query(query, merged_context, limit=6, mode_key=query_type)
    retrieved_context = _prioritize_for_intent(query_type, retrieved_context)[:6]

    writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
    writer_payload["retrieved_context"] = retrieved_context
    writer_payload["user_query"] = query
    writer_payload["query_type"] = query_type
    writer_payload["has_document_context"] = bool(document_context)
    writer_payload["strong_document_match"] = strong_document_match
    writer_payload["mode"] = mode

    web_sources = _extract_sources(research_data)
    retrieved_web_sources, retrieved_document_sources = _split_sources_by_type(retrieved_context)
    if document_retrieval_failed or document_not_indexed:
        web_sources = []
        document_sources = []
        retrieved_context = []
        writer_payload["retrieved_context"] = []
    else:
        web_sources = retrieved_web_sources or (web_sources if query_type == "web" else web_sources[:2])
        document_sources = retrieved_document_sources
    writer_payload["resume_query"] = query_type == "resume"
    writer_payload["has_resume_context"] = _has_resume_context(document_sources)

    answer_payload = write(
        plan_text,
        writer_payload,
        conversation_context=conversation_context,
    )
    if document_retrieval_failed or document_not_indexed:
        answer_payload = _fallback_answer_payload_by_type(query, query_type)
        print("FALLBACK TRIGGERED: document retrieval unavailable for document-based query")
    if not isinstance(answer_payload, dict):
        answer_payload = _fallback_answer_payload_by_type(query, query_type)
        print("FALLBACK TRIGGERED: writer returned non-dict payload")
    key_findings = _make_query_focused_findings(query, retrieved_context)
    web_sources_markdown = _build_sources_markdown(web_sources, source_type="web")
    document_sources_markdown = _build_sources_markdown(document_sources, source_type="document")
    final_report = _answer_payload_to_markdown(answer_payload)
    if _contains_report_style_language(final_report) or _is_generic_response(query, final_report, query_type):
        answer_payload = _fallback_answer_payload_by_type(query, query_type)
        final_report = _answer_payload_to_markdown(answer_payload)
        print("FALLBACK TRIGGERED: generic or report-style output detected")
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
        "answer_payload": answer_payload,
        "final_report": final_report,
        "structured_response": structured_response,
        "sources": web_sources + document_sources,
        "web_sources": web_sources,
        "document_sources": document_sources,
        "query_type": query_type,
        "mode": mode,
        "has_indexed_documents": has_indexed_documents,
        "doc_chunks_count": indexed_doc_chunks_count,
        "document_retrieval_failed": document_retrieval_failed,
        "document_not_indexed": document_not_indexed,
        "status_message": (
            "Document uploaded but not indexed. Please click 'Index Documents'."
            if document_not_indexed else
            "Document not properly indexed. Try re-indexing."
            if document_retrieval_failed else
            ""
        ),
        # Compatibility keys for the current frontend.
        "research": writer_payload,
        "final": final_report,
    }


def index_document_file(file_path):
    return index_document(file_path)
