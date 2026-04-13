import logging
import re
import time
import traceback
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


def _ensure_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(_safe_detail(value))


def _finalize_response(endpoint_name: str, response):
    if response is None:
        response = {"success": False, "error": "Empty response"}
    if not isinstance(response, dict):
        response = {
            "success": False,
            "error": "Invalid response",
            "debug_exception": _ensure_string(response),
        }

    safe_response = _safe_detail(response)
    if not isinstance(safe_response, dict):
        safe_response = {
            "success": False,
            "error": "Invalid response",
            "debug_exception": _ensure_string(safe_response),
        }

    safe_response["success"] = bool(safe_response.get("success", False))
    if safe_response["success"]:
        final_answer = _ensure_string(safe_response.get("final_report") or safe_response.get("final") or safe_response.get("raw_answer"))
        if final_answer:
            safe_response.setdefault("answer", final_answer)
        else:
            safe_response["success"] = False
            safe_response["error"] = "Empty response"
            safe_response["answer"] = ""
    else:
        safe_response["error"] = _ensure_string(safe_response.get("error") or "Internal server error")

    safe_response["raw_answer"] = _ensure_string(safe_response.get("raw_answer"))
    safe_response.setdefault("answer", _ensure_string(safe_response.get("answer")))
    print(f"FINAL RESPONSE: {safe_response}")
    logger.info("endpoint=%s finalized_response_keys=%s", endpoint_name, sorted(safe_response.keys()))
    return safe_response


class OutboundServiceError(RuntimeError):
    def __init__(self, message: str, failed_service: str, exception_type: str, retry_count: int):
        super().__init__(message)
        self.failed_service = failed_service
        self.exception_type = exception_type
        self.retry_count = retry_count


def _sanitize_exception_message(exc: Exception) -> str:
    message = " ".join(str(exc).split()).strip()
    if "connection reset by peer" in message.lower():
        return "Connection reset by peer."
    if "connection aborted" in message.lower():
        return "Connection aborted by remote host."
    return message or exc.__class__.__name__


def _classify_outbound_failure(exc: Exception, service_hint: str):
    exception_type = exc.__class__.__name__
    lowered = f"{exception_type} {_sanitize_exception_message(exc)}".lower()
    normalized_service = service_hint.upper()

    if "connection reset" in lowered or "connection aborted" in lowered:
        code = f"{normalized_service}_CONNECTION_RESET"
        if normalized_service == "TAVILY":
            user_error = "Live search failed temporarily. Please try again."
            public_error = "Tavily connection reset"
        elif normalized_service == "OPENAI":
            user_error = "Answer generation failed temporarily. Please try again."
            public_error = "OpenAI connection reset"
        else:
            user_error = "Network request failed temporarily. Please try again."
            public_error = "Request connection reset"
    elif "timeout" in lowered:
        code = "REQUEST_TIMEOUT"
        if normalized_service == "TAVILY":
            user_error = "Live search failed temporarily. Please try again."
            public_error = "Tavily request timeout"
        elif normalized_service == "OPENAI":
            user_error = "Answer generation failed temporarily. Please try again."
            public_error = "OpenAI request timeout"
        else:
            user_error = "Request timed out. Please try again."
            public_error = "Request timeout"
    else:
        code = "REQUEST_EXCEPTION"
        if normalized_service == "TAVILY":
            user_error = "Live search failed temporarily. Please try again."
            public_error = "Tavily request failed"
        elif normalized_service == "OPENAI":
            user_error = "Answer generation failed temporarily. Please try again."
            public_error = "OpenAI request failed"
        else:
            user_error = "Request failed temporarily. Please try again."
            public_error = "Request exception"

    return {
        "code": code,
        "failed_service": normalized_service,
        "exception_type": exception_type,
        "user_error": user_error,
        "public_error": public_error,
        "details": _sanitize_exception_message(exc),
    }


def _call_with_retries(operation, *, stage: str, service_hint: str, retries: int = 2, backoff_seconds: float = 0.5):
    last_failure = None
    for attempt in range(retries + 1):
        try:
            return operation(), attempt
        except Exception as exc:
            failure = _classify_outbound_failure(exc, service_hint)
            last_failure = failure
            logger.exception(
                "outbound_failure code=%s stage=%s failed_service=%s exception_type=%s retry_count=%s",
                failure["code"],
                stage,
                failure["failed_service"],
                failure["exception_type"],
                attempt,
            )
            print(
                f"outbound_failure code={failure['code']} stage={stage} failed_service={failure['failed_service']} "
                f"exception_type={failure['exception_type']} retry_count={attempt}"
            )
            if attempt >= retries:
                raise OutboundServiceError(
                    failure["public_error"],
                    failed_service=failure["failed_service"],
                    exception_type=failure["exception_type"],
                    retry_count=attempt,
                ) from exc
            time.sleep(backoff_seconds * (attempt + 1))
    raise OutboundServiceError(
        last_failure["public_error"] if last_failure else "Request failed",
        failed_service=last_failure["failed_service"] if last_failure else service_hint.upper(),
        exception_type=last_failure["exception_type"] if last_failure else "RuntimeError",
        retry_count=retries,
    )


def _outbound_error_payload(endpoint_name: str, exc: OutboundServiceError) -> dict:
    return _error_payload(
        endpoint_name,
        str(exc),
        details={
            "debug_failed_service": exc.failed_service,
            "debug_exception_type": exc.exception_type,
            "debug_retry_count": exc.retry_count,
        },
    ) | {
        "debug_failed_service": exc.failed_service,
        "debug_exception_type": exc.exception_type,
        "debug_retry_count": exc.retry_count,
    }


def _log_stage_failure(stage: str, exc: Exception):
    logger.exception("research_stage=%s failed", stage)
    print(f"research_stage={stage} failed error={exc}")


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
        content = _extract_result_content(item)
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
                "content": _extract_result_content(item),
                "source_type": "web",
            }
        )
    return sources


def _extract_result_content(item):
    if not isinstance(item, dict):
        return ""
    parts = []
    for key in ("content", "raw_content", "snippet", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = value.strip()
            if cleaned not in parts:
                parts.append(cleaned)
    return "\n".join(parts).strip()


def _source_snippet(item, limit: int = 220):
    content = " ".join((_extract_result_content(item) or "").split())
    if len(content) > limit:
        return content[:limit].rstrip() + "..."
    return content


def _build_evidence_text(items, limit: int = 6):
    blocks = []
    limit = 5 if limit > 5 else limit
    for item in (items or [])[:limit]:
        title = item.get("title", "Untitled source")
        url = item.get("url", "")
        snippet = _source_snippet(item, limit=700)
        if not snippet:
            continue
        block = f"Title: {title}"
        block += f"\nSnippet: {snippet}"
        if url:
            block += f"\nURL: {url}"
        blocks.append(block)
    return "\n\n".join(blocks).strip()


def _simplify_query(query: str):
    terms = _query_terms(query)
    if not terms:
        return (query or "").strip()
    return " ".join(terms[:8]).strip()


def _expanded_query(query: str, query_type: str):
    suffix = {
        "resume": "role fit skills market demand",
        "study": "summary explanation examples important points",
        "interview": "interview questions answers examples",
        "web": "2026 list reviews comparison",
    }.get(query_type, "2026 list reviews comparison")
    base = _simplify_query(query) or (query or "").strip()
    return f"{base} {suffix}".strip()


def _is_recommendation_query(query: str):
    lowered = (query or "").lower()
    markers = [
        "best tools", "top tools", "ai tools", "apps", "productivity apps",
        "best apps", "top apps", "best websites", "top websites",
        "for students", "for coding", "coding students", "recommend",
    ]
    return any(marker in lowered for marker in markers)


def _minimum_evidence_length(query: str, query_type: str):
    if query_type == "web" and _is_recommendation_query(query):
        return 80
    if query_type in {"interview", "study"}:
        return 220
    return 320


def _has_usable_evidence(query: str, items, evidence_text: str, query_type: str):
    if not items:
        return False
    if _is_recommendation_query(query):
        non_empty_items = 0
        for item in items[:5]:
            title = (item.get("title") or "").strip()
            content = (_extract_result_content(item) or "").strip()
            if title or content:
                non_empty_items += 1
        if non_empty_items >= 2 and (evidence_text or "").strip():
            return True
    if len(evidence_text or "") >= _minimum_evidence_length(query, query_type):
        return True
    query_terms = _query_terms(query)
    for item in items[:4]:
        haystack = " ".join(
            [
                item.get("title", ""),
                item.get("content", ""),
                item.get("text", ""),
                item.get("url", ""),
            ]
        ).lower()
        if not query_terms:
            return True
        matches = sum(1 for term in query_terms if term in haystack)
        if matches >= 1:
            return True
    return False


def _merge_ranked_context(query: str, query_type: str, *context_groups, limit: int = 6):
    merged = []
    seen = set()
    for group in context_groups:
        for item in group or []:
            key = (
                (item.get("url") or "").strip().lower(),
                (item.get("title") or "").strip().lower(),
                (_extract_result_content(item) or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return _rerank_for_query(query, merged, limit=limit, mode_key=query_type)


def _debug_sources(prefix: str, items):
    count = len(items or [])
    logger.info("%s source_count=%s", prefix, count)
    print(f"{prefix} source_count={count}")
    for index, item in enumerate((items or [])[:2], start=1):
        snippet = _source_snippet(item, limit=180)
        title = item.get("title", "Untitled source")
        logger.info("%s source_%s title=%s snippet=%s", prefix, index, title, snippet)
        print(f"{prefix} source_{index} title={title} snippet={snippet}")


def _debug_raw_tavily_results(prefix: str, research_data):
    results = research_data.get("results", []) if isinstance(research_data, dict) else []
    logger.info("%s raw_result_count=%s", prefix, len(results))
    print(f"{prefix} raw_result_count={len(results)}")
    for index, item in enumerate(results[:3], start=1):
        title = item.get("title", "Untitled source")
        snippet = _source_snippet(item, limit=220)
        logger.info("%s raw_result_%s title=%s snippet=%s", prefix, index, title, snippet)
        print(f"{prefix} raw_result_{index} title={title} snippet={snippet}")


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

    sections = [f"## {primary_title}\n" + "\n".join(f"- {item}" for item in recommendations)]

    if reasons:
        sections.append(f"## {reasons_title}\n" + "\n".join(f"- {item}" for item in reasons))
    if insights:
        sections.append(f"## {insights_title}\n" + insights)

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
    if _is_recommendation_query(query):
        bullet_count = sum(1 for line in (final_report or "").splitlines() if line.strip().startswith("- "))
        if bullet_count >= 3 and not any(phrase in report_lower for phrase in GENERIC_META_PHRASES):
            return False
    if not query_terms:
        return any(phrase in report_lower for phrase in GENERIC_META_PHRASES)
    if any(phrase in report_lower for phrase in GENERIC_META_PHRASES):
        return True
    matches = sum(1 for term in query_terms if term in report_lower)
    return matches < max(1, min(2, len(query_terms)))


def _fallback_answer_payload_by_type(query: str, query_type: str):
    return {
        "primary_title": "Direct Answer",
        "recommendations": ["I couldn't find enough reliable live data for this query right now."],
        "reasons_title": "Why",
        "reasons": [],
        "insights_title": "Key Insights",
        "insights": "",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
    }


def _debug_query_type(query: str):
    return "recommendation" if _is_recommendation_query(query) else "analysis"


def _clean_candidate_name(value: str):
    text = " ".join((value or "").split()).strip(" -,:;|")
    if not text:
        return ""
    lowered = text.lower()
    blocked = {
        "best ai tools for students",
        "top ai tools for students 2026",
        "student productivity ai tools",
        "latest ai tools for coding students",
        "best productivity apps for students",
        "top productivity apps",
        "best websites for coding students",
        "ai tools",
        "productivity apps",
        "coding students",
        "students",
        "tools",
        "apps",
        "websites",
        "platforms",
    }
    if lowered in blocked:
        return ""
    if len(text) > 60:
        return ""
    return text


def _extract_named_recommendations(items, limit: int = 5):
    candidates = []
    seen = set()
    patterns = [
        r"\b(?:ChatGPT|Perplexity|Notion AI|Notion|Grammarly|GitHub Copilot|Copilot|Claude|Canva|Otter|QuillBot|Gemini|Cursor|Replit|Khanmigo|Duolingo Max)\b",
        r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2}\b",
    ]

    for item in items or []:
        raw_parts = [item.get("title", ""), item.get("content", ""), item.get("text", "")]
        title = item.get("title", "")
        for segment in re.split(r"[|,:/•\-]", title):
            cleaned = _clean_candidate_name(segment)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                candidates.append(cleaned)
        for raw_part in raw_parts:
            for pattern in patterns:
                for match in re.findall(pattern, raw_part or ""):
                    cleaned = _clean_candidate_name(match)
                    if cleaned and cleaned.lower() not in seen:
                        seen.add(cleaned.lower())
                        candidates.append(cleaned)
                    if len(candidates) >= limit:
                        return candidates[:limit]
    return candidates[:limit]


def _build_recommendation_answer_payload(query: str, items):
    recommendations = _extract_named_recommendations(items, limit=5)
    if not recommendations:
        return None
    supporting_points = []
    for item in (items or [])[:3]:
        title = item.get("title", "Source")
        snippet = _source_snippet(item, limit=180)
        if snippet:
            supporting_points.append(f"{title}: {snippet}")
    return {
        "primary_title": "Direct Answer",
        "recommendations": recommendations,
        "reasons_title": "Why",
        "reasons": supporting_points[:3],
        "insights_title": "Key Insights",
        "insights": "",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
        "raw_answer": "\n".join(f"- {item}" for item in recommendations),
        "_debug_parse_success": True,
    }


def _has_resume_context(query: str, conversation_context: str):
    lowered = f"{query} {conversation_context}".lower()
    return any(token in lowered for token in ["resume", "cv", "profile", "experience", "skills", "project"])


def run_research_pipeline(query: str):
    try:
        fallback_triggered = False
        fallback_reason = ""
        debug_failed_service = ""
        debug_exception_type = ""
        debug_retry_count = 0
        plan_retry_count = 0
        tavily_retry_count = 0
        embedding_retry_count = 0
        writer_retry_count = 0
        logger.info("incoming_query=%s", query)
        print(f"incoming_query={query}")
        conversation_context = format_history_context(limit=5)
        query_type = _classify_query_intent(query)
        mode = _mode_label(query_type)

        logger.info("research_pipeline query=%s query_type=%s mode=%s", query, query_type, mode)
        print(f"research_pipeline query={query} query_type={query_type} mode={mode}")
        try:
            plan_text, plan_retry_count = _call_with_retries(
                lambda: plan(query, conversation_context=conversation_context, query_type=query_type),
                stage="planner_call",
                service_hint="OPENAI",
            )
        except OutboundServiceError as exc:
            return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))

        try:
            research_data, tavily_retry_count = _call_with_retries(
                lambda: research(query, plan_text=plan_text),
                stage="tavily_search",
                service_hint="TAVILY",
            )
        except OutboundServiceError as exc:
            return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))

        initial_results = research_data.get("results", []) if isinstance(research_data, dict) else []
        logger.info(
            "research_results query_used=%s result_count=%s attempted_queries=%s",
            (research_data or {}).get("query_used", ""),
            len(initial_results),
            (research_data or {}).get("attempted_queries", []),
        )
        print(
            f"research_results query_used={(research_data or {}).get('query_used', '')} "
            f"result_count={len(initial_results)}"
        )
        logger.info("retry_queries_used=%s", (research_data or {}).get("attempted_queries", []))
        print(f"retry_queries_used={(research_data or {}).get('attempted_queries', [])}")
        _debug_raw_tavily_results("tavily_initial", research_data)

        if not initial_results:
            retry_query = _expanded_query(query, query_type)
            logger.info("research_retry reason=no_results retry_query=%s", retry_query)
            print(f"research_retry reason=no_results retry_query={retry_query}")
            try:
                research_data, tavily_retry_count = _call_with_retries(
                    lambda: research(retry_query, plan_text=plan_text),
                    stage="tavily_retry_search",
                    service_hint="TAVILY",
                )
            except OutboundServiceError as exc:
                return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
            initial_results = research_data.get("results", []) if isinstance(research_data, dict) else []
            _debug_raw_tavily_results("tavily_no_result_retry", research_data)

        try:
            web_store = VectorStore(namespace="web")
        except Exception as exc:
            _log_stage_failure("vector_store_init", exc)
            return _finalize_response("run_research_pipeline", _error_payload(
                "run_research_pipeline",
                "Evidence store initialization failed.",
                {"stage": "vector_store_init", "message": _sanitize_exception_message(exc)},
            ))

        web_documents = _build_web_documents(research_data)
        _debug_sources("tavily_results", web_documents)
        logger.info("tavily_response_count=%s", len(web_documents))
        print(f"tavily_response_count={len(web_documents)}")
        try:
            if web_documents:
                _, embedding_retry_count = _call_with_retries(
                    lambda: web_store.add_documents(web_documents),
                    stage="vector_store_add_documents",
                    service_hint="OPENAI",
                )
        except OutboundServiceError as exc:
            return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))

        retrieval_hint = MODE_QUERY_HINTS.get(query_type, MODE_QUERY_HINTS["web"])
        retrieval_query = f"{query}\n{plan_text}\n{retrieval_hint}"
        current_context = _rerank_for_query(query, web_documents, limit=8, mode_key=query_type)
        try:
            cached_context, embedding_retry_count = _call_with_retries(
                lambda: web_store.similarity_search(query=retrieval_query, top_k=8),
                stage="vector_store_similarity_search",
                service_hint="OPENAI",
            )
        except OutboundServiceError as exc:
            return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
        retrieved_context = _merge_ranked_context(query, query_type, current_context, cached_context, limit=6)

        evidence_text = _build_evidence_text(retrieved_context)
        evidence_minimum = _minimum_evidence_length(query, query_type)
        evidence_usable = _has_usable_evidence(query, retrieved_context, evidence_text, query_type)
        logger.info(
            "evidence_check min_chars=%s evidence_chars=%s usable=%s recommendation_query=%s",
            evidence_minimum,
            len(evidence_text),
            evidence_usable,
            _is_recommendation_query(query),
        )
        print(
            f"evidence_check min_chars={evidence_minimum} evidence_chars={len(evidence_text)} "
            f"usable={evidence_usable} recommendation_query={_is_recommendation_query(query)}"
        )
        if not evidence_usable:
            retry_query = _expanded_query(query, query_type)
            logger.info(
                "research_retry reason=weak_evidence evidence_chars=%s retry_query=%s",
                len(evidence_text),
                retry_query,
            )
            print(f"research_retry reason=weak_evidence evidence_chars={len(evidence_text)} retry_query={retry_query}")
            try:
                retry_data, tavily_retry_count = _call_with_retries(
                    lambda: research(retry_query, plan_text=plan_text),
                    stage="tavily_weak_evidence_retry",
                    service_hint="TAVILY",
                )
            except OutboundServiceError as exc:
                return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
            _debug_raw_tavily_results("tavily_weak_evidence_retry", retry_data)
            retry_documents = _build_web_documents(retry_data)
            _debug_sources("tavily_retry_results", retry_documents)
            try:
                if retry_documents:
                    _, embedding_retry_count = _call_with_retries(
                        lambda: web_store.add_documents(retry_documents),
                        stage="vector_store_add_retry_documents",
                        service_hint="OPENAI",
                    )
            except OutboundServiceError as exc:
                return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
            retry_current = _rerank_for_query(query, retry_documents, limit=8, mode_key=query_type)
            try:
                retry_cached, embedding_retry_count = _call_with_retries(
                    lambda: web_store.similarity_search(query=f"{retry_query}\n{retrieval_hint}", top_k=8),
                    stage="vector_store_retry_similarity_search",
                    service_hint="OPENAI",
                )
            except OutboundServiceError as exc:
                return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
            retrieved_context = _merge_ranked_context(query, query_type, current_context, retry_current, cached_context, retry_cached, limit=6)
            if isinstance(retry_data, dict) and retry_data.get("results"):
                research_data = retry_data
            evidence_text = _build_evidence_text(retrieved_context)
            evidence_usable = _has_usable_evidence(query, retrieved_context, evidence_text, query_type)

        source_count = len(retrieved_context)
        logger.info("retrieved_context_count=%s evidence_chars=%s", source_count, len(evidence_text))
        print(f"retrieved_context_count={source_count} evidence_chars={len(evidence_text)}")
        _debug_sources("retrieved_context", retrieved_context)
        logger.info("debug query=%s sources_fetched_count=%s evidence_length=%s fallback_triggered=%s", query, source_count, len(evidence_text), fallback_triggered)
        print(f"debug query={query} sources_fetched_count={source_count} evidence_length={len(evidence_text)} fallback_triggered={fallback_triggered}")

        web_sources = _dedupe_sources(retrieved_context or _extract_sources(research_data), limit=6)
        writer_payload = dict(research_data) if isinstance(research_data, dict) else {}
        writer_payload["question"] = query
        writer_payload["retrieved_context"] = retrieved_context
        writer_payload["evidence"] = evidence_text
        writer_payload["evidence_items"] = retrieved_context
        writer_payload["sources"] = [item.get("url", "") for item in web_sources if item.get("url")]
        writer_payload["debug_source_count"] = source_count
        writer_payload["debug_evidence_chars"] = len(evidence_text)
        writer_payload["user_query"] = query
        writer_payload["query_type"] = query_type
        writer_payload["debug_query_type"] = _debug_query_type(query)
        writer_payload["mode"] = mode
        writer_payload["recommendation_query"] = _is_recommendation_query(query)
        writer_payload["has_resume_context"] = _has_resume_context(query, conversation_context)
        logger.info("writer_input source_count=%s evidence_length=%s", len(writer_payload["sources"]), len(writer_payload["evidence"]))
        print(f"writer_input source_count={len(writer_payload['sources'])} evidence_length={len(writer_payload['evidence'])}")

        if writer_payload["evidence"] is None:
            writer_payload["evidence"] = ""
        if writer_payload["retrieved_context"] is None:
            writer_payload["retrieved_context"] = []

        recommendation_payload = None
        if _is_recommendation_query(query) and source_count > 0 and (evidence_text or "").strip():
            recommendation_payload = _build_recommendation_answer_payload(query, retrieved_context)

        answer_payload = None
        if not evidence_usable:
            if source_count == 0 or not (evidence_text or "").strip():
                answer_payload = _fallback_answer_payload_by_type(query, query_type)
                fallback_triggered = True
                fallback_reason = "no_usable_evidence"
                logger.info("fallback_triggered=no_usable_evidence")
                print("fallback_triggered=no_usable_evidence")
            else:
                evidence_usable = True
        if evidence_usable and answer_payload is None:
            try:
                answer_payload, writer_retry_count = _call_with_retries(
                    lambda: write(
                        plan_text,
                        writer_payload,
                        conversation_context=conversation_context,
                    ),
                    stage="writer_call",
                    service_hint="OPENAI",
                )
                logger.info("writer_call=success")
                print("writer_call=success")
                logger.info("writer_raw_present=%s parse_success=%s", bool(answer_payload.get("raw_answer")), bool(answer_payload.get("_debug_parse_success", False)))
                print(f"writer_raw_present={bool(answer_payload.get('raw_answer'))} parse_success={bool(answer_payload.get('_debug_parse_success', False))}")
            except OutboundServiceError as exc:
                return _finalize_response("run_research_pipeline", _outbound_error_payload("run_research_pipeline", exc))
        if not isinstance(answer_payload, dict):
            if recommendation_payload:
                answer_payload = recommendation_payload
                fallback_reason = "used_recommendation_payload"
            else:
                answer_payload = _fallback_answer_payload_by_type(query, query_type)
                fallback_triggered = True
                fallback_reason = "writer_non_dict"
                logger.info("fallback_triggered=writer_non_dict")

        writer_raw_answer = answer_payload.get("raw_answer", "") if isinstance(answer_payload, dict) else ""
        parse_success = bool(answer_payload.get("_debug_parse_success", False)) if isinstance(answer_payload, dict) else False
        logger.info("parsed_answer_payload=%s", answer_payload)
        print(f"parsed_answer_payload={answer_payload}")

        key_findings = _build_key_findings(query, retrieved_context)
        final_report = _answer_payload_to_markdown(answer_payload)
        if _contains_report_style_language(final_report) or _is_generic_response(query, final_report):
            raw_writer_usable = bool(writer_raw_answer and writer_raw_answer.strip())
            if source_count == 0 or not (evidence_text or "").strip():
                answer_payload = _fallback_answer_payload_by_type(query, query_type)
                final_report = _answer_payload_to_markdown(answer_payload)
                fallback_triggered = True
                fallback_reason = "generic_or_report_style"
                logger.info("fallback_triggered=generic_or_report_style")
            elif raw_writer_usable:
                answer_payload = {
                    "primary_title": "Direct Answer",
                    "recommendations": [writer_raw_answer],
                    "reasons_title": "Why",
                    "reasons": [],
                    "insights_title": "Key Insights",
                    "insights": "",
                    "improvement_title": "",
                    "improvement_tips": [],
                    "extra_sections": [],
                    "raw_answer": writer_raw_answer,
                    "_debug_parse_success": False,
                }
                final_report = _answer_payload_to_markdown(answer_payload)
                fallback_reason = "used_raw_writer_output"
            elif recommendation_payload:
                answer_payload = recommendation_payload
                final_report = _answer_payload_to_markdown(answer_payload)
                fallback_reason = "used_recommendation_payload"

        logger.info("debug query=%s sources_fetched_count=%s evidence_length=%s fallback_triggered=%s", query, source_count, len(evidence_text), fallback_triggered)
        print(f"debug query={query} sources_fetched_count={source_count} evidence_length={len(evidence_text)} fallback_triggered={fallback_triggered}")
        print(f"FALLBACK REASON: {fallback_reason}")
        print(f"WRITER RAW: {writer_raw_answer}")

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
            "fallback_triggered": fallback_triggered,
            "debug_source_count": source_count,
            "debug_evidence_length": len(evidence_text),
            "debug_query_type": _debug_query_type(query),
            "debug_writer_raw_present": bool(writer_raw_answer and writer_raw_answer.strip()),
            "debug_parse_success": parse_success,
            "debug_fallback_reason": fallback_reason,
            "debug_failed_service": debug_failed_service,
            "debug_exception_type": debug_exception_type,
            "debug_retry_count": max(plan_retry_count, tavily_retry_count, embedding_retry_count, writer_retry_count, debug_retry_count),
            "debug_retry_used": (research_data or {}).get("attempted_queries", []),
            "raw_answer": writer_raw_answer,
            "research": writer_payload,
            "final": final_report,
        }
        logger.info("final_response_text=%s", final_report)
        print(f"final_response_text={final_report}")
        logger.info("final_response_keys=%s", sorted(payload.keys()))
        print(f"final_response_keys={sorted(payload.keys())}")
        return _finalize_response("run_research_pipeline", _success_payload("run_research_pipeline", payload))
    except Exception as exc:
        error_trace = traceback.format_exc()
        logger.exception("endpoint=run_research_pipeline failed")
        print(f"ERROR TRACE: {error_trace}")
        return _finalize_response(
            "run_research_pipeline",
            {
                "success": False,
                "error": "Internal server error",
                "debug_exception": _sanitize_exception_message(exc),
                "details": {"traceback": error_trace},
            },
        )
