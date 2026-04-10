import logging
import re
from pathlib import Path

from backend.agents.planner import plan
from backend.agents.researcher import research
from backend.agents.writer import write
from backend.services.memory import format_history_context, save_to_memory
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


MODE_LABELS = {
    "resume": "Resume Analyzer",
    "study": "Study Mode",
    "interview": "Interview Prep",
    "web": "Web Research",
}


MODE_QUERY_HINTS = {
    "resume": "resume profile skills experience strengths gaps career fit",
    "study": "concept summary explanation revision important questions overview",
    "interview": "interview questions model answers follow ups technical behavioral",
    "web": "latest compare recommendation evidence sources",
}


RESUME_MARKERS = {
    "resume", "cv", "profile", "best roles", "best jobs", "career", "roles", "job fit", "suitable roles"
}
STUDY_MARKERS = {
    "chapter", "exam", "2 mark", "5 mark", "10 mark", "important topics", "revision", "summarize",
    "explain", "study", "learn", "teach me", "topic", "notes", "short answer", "long answer"
}
INTERVIEW_MARKERS = {
    "interview", "mock interview", "viva", "interview questions", "prepare me", "ask me questions"
}
WEB_MARKERS = {
    "latest", "current", "today", "recent", "compare", "comparison", "tools", "news", "updates", "best"
}
GENERIC_META_PHRASES = {
    "use the strongest source",
    "focus on the comparison",
    "take the next practical step",
    "lead with the direct answer",
    "use the highest-relevance sources",
    "depends on the evidence",
    "state the factors",
    "the answer should",
    "the most useful answer",
}


def _safe_detail(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_detail(item) for item in value]
    if isinstance(value, set):
        return [_safe_detail(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _safe_detail(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _safe_detail(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _safe_detail(vars(value))
        except Exception:
            pass
    return str(value)


def _success_payload(endpoint_name: str, payload: dict) -> dict:
    safe_payload = _safe_detail(payload)
    logger.info(
        "endpoint=%s success=%s response_keys=%s",
        endpoint_name,
        bool(safe_payload.get("success", False)),
        sorted(safe_payload.keys()),
    )
    return safe_payload


def _error_payload(endpoint_name: str, error: str, details=None) -> dict:
    payload = {
        "success": False,
        "error": error,
    }
    if details:
        payload["details"] = _safe_detail(details)
    logger.info(
        "endpoint=%s success=%s response_keys=%s",
        endpoint_name,
        False,
        sorted(payload.keys()),
    )
    return payload


def _query_terms(query: str):
    terms = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
        "give", "me", "show", "find", "analyze", "best", "what", "which", "is",
        "are", "my", "about", "from", "that", "this", "please", "help", "want",
    }
    return [term for term in terms if term not in stopwords and len(term) > 2]


def _mode_label(mode_key: str):
    return MODE_LABELS.get(mode_key, "Web Research")


def _classify_query_intent(query: str):
    lowered = (query or "").lower()
    if any(marker in lowered for marker in INTERVIEW_MARKERS):
        return "interview"
    if any(marker in lowered for marker in RESUME_MARKERS):
        return "resume"
    if any(marker in lowered for marker in STUDY_MARKERS):
        return "study"
    if any(marker in lowered for marker in WEB_MARKERS):
        return "web"
    return "web"


def _query_relevance_score(query: str, item):
    haystack = " ".join(
        [
            item.get("title", ""),
            item.get("content", ""),
            item.get("text", ""),
            item.get("url", ""),
        ]
    ).lower()
    terms = _query_terms(query)
    if not terms:
        return 0.0

    score = 0.0
    for term in terms:
        if term in haystack:
            score += 1.0
            if term in (item.get("title", "").lower()):
                score += 0.75
    return score / max(1, len(terms))


def _mode_source_boost(mode_key: str, item, query: str):
    score = 0.0
    lowered_query = (query or "").lower()
    if mode_key == "web":
        score += 1.0
    if any(token in lowered_query for token in ["latest", "current", "recent", "today"]):
        score += 1.25
    if mode_key == "study" and any(token in lowered_query for token in ["chapter", "topic", "exam", "question"]):
        score += 0.75
    if mode_key == "resume" and any(token in lowered_query for token in ["resume", "role", "career"]):
        score += 0.75
    if mode_key == "interview" and any(token in lowered_query for token in ["interview", "question", "mock"]):
        score += 0.75
    return score


def _build_web_documents(research_data):
    if not isinstance(research_data, dict):
        return []

    web_documents = []
    for item in research_data.get("results", []):
        title = item.get("title", "Untitled source")
        content = item.get("content", "")
        url = item.get("url", "")
        text = f"{title}\n{content}".strip()
        if not text:
            continue
        web_documents.append(
            {
                "source_type": "web",
                "title": title,
                "url": url,
                "content": content,
                "text": text,
            }
        )
    return web_documents


def _rerank_for_query(query: str, items, limit: int = 6, mode_key: str = "web"):
    ranked = []
    for item in items:
        combined_score = (
            float(item.get("score", 0.0))
            + (1.75 * _query_relevance_score(query, item))
            + _mode_source_boost(mode_key, item, query)
        )
        enriched = dict(item)
        enriched["query_score"] = combined_score
        ranked.append(enriched)
    ranked.sort(key=lambda item: item.get("query_score", 0.0), reverse=True)
    return ranked[:limit]


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
                "source_type": "web",
            }
        )
    return sources


def _dedupe_sources(sources, limit: int = 6):
    deduped = []
    seen = set()
    for source in sources:
        key = ((source.get("url") or "").strip().lower(), (source.get("title") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "title": source.get("title", "Untitled source"),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
                "source_type": "web",
            }
        )
        if len(deduped) >= limit:
            break
    return deduped


def _build_key_findings(query: str, retrieved_context):
    query_terms = _query_terms(query)
    findings = []
    for item in retrieved_context[:5]:
        title = item.get("title", "Untitled source")
        content = " ".join((item.get("content") or "").split())
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
            findings.append(f"- **{title}**: {snippet}")
    return "\n".join(findings) if findings else "- No tightly matched findings were retrieved."


def _build_sources_markdown(sources):
    if not sources:
        return "- No sources available."

    lines = []
    for source in sources:
        title = source.get("title", "Untitled source")
        url = (source.get("url") or "").strip()
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def _build_structured_response(plan_text, key_findings, final_report, sources_markdown):
    return (
        "## Research Plan\n"
        f"{plan_text}\n\n"
        "## Key Findings\n"
        f"{key_findings}\n\n"
        "## Final Answer\n"
        f"{final_report}\n\n"
        "## Sources\n"
        f"{sources_markdown}"
    )


def _answer_payload_to_markdown(answer_payload):
    recommendations = answer_payload.get("recommendations", [])
    reasons = answer_payload.get("reasons", [])
    insights = answer_payload.get("insights", "")
    improvement_tips = answer_payload.get("improvement_tips", [])
    primary_title = answer_payload.get("primary_title", "Direct Answer")
    reasons_title = answer_payload.get("reasons_title", "Why This Answer")
    insights_title = answer_payload.get("insights_title", "Key Insights")
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


def _is_generic_response(query: str, final_report: str):
    query_terms = _query_terms(query)
    report_lower = (final_report or "").lower()
    if not query_terms:
        return any(phrase in report_lower for phrase in GENERIC_META_PHRASES)
    if any(phrase in report_lower for phrase in GENERIC_META_PHRASES):
        return True
    matches = sum(1 for term in query_terms if term in report_lower)
    return matches < max(1, min(2, len(query_terms)))


def _fallback_answer_payload_by_type(query: str, query_type: str):
    query_label = query.strip() or "the current question"
    if query_type == "resume":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {query_label}, the strongest answer depends on the exact skills, projects, and outcomes visible in the profile plus current market demand.",
                "The most defensible recommendation is usually the role family where technical depth and measurable impact are clearest.",
                "A narrower target role will produce a more precise resume analysis than a broad career prompt.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Resume Analyzer combines role-fit reasoning with web-grounded market context.",
                "It stays useful even when the profile context is only partially available.",
                "The next best step is usually tighter positioning and clearer evidence of impact.",
            ],
            "insights_title": "Key Insights",
            "insights": "Use follow-up questions to drill into best roles, missing skills, or stronger resume bullets.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [
                "Add metrics and ownership to key project bullets.",
                "Align the resume summary to one target role family.",
            ],
            "extra_sections": [],
        }
    if query_type == "study":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {query_label}, the clearest starting point is the core definition, scope, and the most exam-relevant ideas.",
                "The strongest answer should emphasize recurring concepts, cause-effect links, and likely question angles.",
                "If you need more depth, the next pass should break the topic into definitions, short answers, or long answers.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Study Mode is designed for teaching clarity and revision usefulness.",
                "The answer stays grounded in retrieved sources rather than unsupported recall.",
                "It works well for iterative chapter-by-chapter follow-up questions.",
            ],
            "insights_title": "Key Insights",
            "insights": f"This study answer is optimized for the topic: {query}",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [],
        }
    if query_type == "interview":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {query_label}, the strongest interview answer is the one supported by a concrete example, decision, and outcome.",
                "Good interview preparation should show what you did, why you chose that approach, and what happened next.",
                "Expect follow-up pressure on trade-offs, ownership, and measurable impact.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Interview Prep is strongest when answers are specific and evidence-backed.",
                "The answer is optimized for practical speaking use, not passive reading.",
                "You can use follow-up prompts to generate mock interview rounds.",
            ],
            "insights_title": "Key Insights",
            "insights": f"This interview-prep answer is aligned to the current query: {query}",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [
                "Tighten one project story with clearer outcomes.",
                "Prepare stronger trade-off explanations.",
            ],
            "extra_sections": [],
        }
    return {
        "primary_title": "Direct Answer",
        "recommendations": [
            f"For {query_label}, the retrieved evidence supports a cautious, source-grounded answer rather than a generic conclusion.",
            "Where the evidence is mixed, the best answer is conditional and should state the main factors driving the outcome.",
            "Any strong claim should stay tied to the current reporting and analysis available in the retrieved sources.",
        ],
        "reasons_title": "Why This Answer",
        "reasons": [
            "The answer is kept query-focused instead of broad or report-like.",
            "Retrieved evidence is ranked before writing the final response.",
            "It is designed to work well for follow-up questions in chat.",
        ],
        "insights_title": "Key Insights",
        "insights": f"This web-grounded fallback stays aligned to the query: {query}",
        "improvement_title": "Improvement Tips",
        "improvement_tips": [],
        "extra_sections": [],
    }


def _has_resume_context(query: str, conversation_context: str):
    lowered = f"{query} {conversation_context}".lower()
    return any(token in lowered for token in ["resume", "cv", "profile", "experience", "skills", "project"])


def run_research_pipeline(query: str):
    try:
        conversation_context = format_history_context(limit=5)
        query_type = _classify_query_intent(query)
        mode = _mode_label(query_type)

        logger.info("query_type=%s mode=%s", query_type, mode)
        plan_text = plan(query, conversation_context=conversation_context, query_type=query_type)
        research_data = research(plan_text)

        web_store = VectorStore(namespace="web")
        web_documents = _build_web_documents(research_data)
        web_store.add_documents(web_documents)

        retrieval_hint = MODE_QUERY_HINTS.get(query_type, MODE_QUERY_HINTS["web"])
        retrieval_query = f"{query}\n{plan_text}\n{retrieval_hint}"
        web_context = web_store.similarity_search(query=retrieval_query, top_k=8)
        retrieved_context = _rerank_for_query(query, web_context, limit=6, mode_key=query_type)

        web_sources = _dedupe_sources(retrieved_context or _extract_sources(research_data), limit=6)
        writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
        writer_payload["question"] = query
        writer_payload["retrieved_context"] = retrieved_context
        writer_payload["evidence"] = retrieved_context
        writer_payload["user_query"] = query
        writer_payload["query_type"] = query_type
        writer_payload["mode"] = mode
        writer_payload["has_resume_context"] = _has_resume_context(query, conversation_context)

        answer_payload = write(
            plan_text,
            writer_payload,
            conversation_context=conversation_context,
        )
        if not isinstance(answer_payload, dict):
            answer_payload = _fallback_answer_payload_by_type(query, query_type)
            logger.info("fallback_triggered=writer_non_dict")

        key_findings = _build_key_findings(query, retrieved_context)
        final_report = _answer_payload_to_markdown(answer_payload)
        if _contains_report_style_language(final_report) or _is_generic_response(query, final_report):
            answer_payload = _fallback_answer_payload_by_type(query, query_type)
            final_report = _answer_payload_to_markdown(answer_payload)
            logger.info("fallback_triggered=generic_or_report_style")

        sources_markdown = _build_sources_markdown(web_sources)
        structured_response = _build_structured_response(
            plan_text,
            key_findings,
            final_report,
            sources_markdown,
        )
        save_to_memory(query, structured_response)

        payload = {
            "success": True,
            "plan": plan_text,
            "findings": key_findings,
            "key_findings": key_findings,
            "answer_payload": answer_payload,
            "final_report": final_report,
            "structured_response": structured_response,
            "sources": web_sources,
            "web_sources": web_sources,
            "query_type": query_type,
            "mode": mode,
            "memory_used": bool(conversation_context),
            "research": writer_payload,
            "final": final_report,
        }
        return _success_payload("run_research_pipeline", payload)
    except Exception as exc:
        logger.exception("endpoint=run_research_pipeline failed")
        return _error_payload("run_research_pipeline", "Research execution failed.", str(exc))
