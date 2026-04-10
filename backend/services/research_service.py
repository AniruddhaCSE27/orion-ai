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
    for key in ("content", "raw_content", "snippet", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_snippet(item, limit: int = 220):
    content = " ".join((_extract_result_content(item) or "").split())
    if len(content) > limit:
        return content[:limit].rstrip() + "..."
    return content


def _build_evidence_text(items, limit: int = 6):
    blocks = []
    for item in (items or [])[:limit]:
        title = item.get("title", "Untitled source")
        url = item.get("url", "")
        snippet = _source_snippet(item, limit=700)
        if not snippet:
            continue
        block = f"Source: {title}"
        if url:
            block += f"\nURL: {url}"
        block += f"\nEvidence: {snippet}"
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


def _has_resume_context(query: str, conversation_context: str):
    lowered = f"{query} {conversation_context}".lower()
    return any(token in lowered for token in ["resume", "cv", "profile", "experience", "skills", "project"])


def run_research_pipeline(query: str):
    try:
        fallback_triggered = False
        logger.info("incoming_query=%s", query)
        print(f"incoming_query={query}")
        conversation_context = format_history_context(limit=5)
        query_type = _classify_query_intent(query)
        mode = _mode_label(query_type)

        logger.info("research_pipeline query=%s query_type=%s mode=%s", query, query_type, mode)
        print(f"research_pipeline query={query} query_type={query_type} mode={mode}")
        plan_text = plan(query, conversation_context=conversation_context, query_type=query_type)

        try:
            research_data = research(query, plan_text=plan_text)
        except Exception as exc:
            _log_stage_failure("tavily_search", exc)
            return _error_payload(
                "run_research_pipeline",
                "Live search failed.",
                {"stage": "tavily_search", "message": str(exc)},
            )

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

        if not initial_results:
            retry_query = _expanded_query(query, query_type)
            logger.info("research_retry reason=no_results retry_query=%s", retry_query)
            print(f"research_retry reason=no_results retry_query={retry_query}")
            try:
                research_data = research(retry_query, plan_text=plan_text)
            except Exception as exc:
                _log_stage_failure("tavily_retry_search", exc)
                return _error_payload(
                    "run_research_pipeline",
                    "Live search retry failed.",
                    {"stage": "tavily_retry_search", "message": str(exc)},
                )
            initial_results = research_data.get("results", []) if isinstance(research_data, dict) else []

        try:
            web_store = VectorStore(namespace="web")
        except Exception as exc:
            _log_stage_failure("vector_store_init", exc)
            return _error_payload(
                "run_research_pipeline",
                "Evidence store initialization failed.",
                {"stage": "vector_store_init", "message": str(exc)},
            )

        web_documents = _build_web_documents(research_data)
        _debug_sources("tavily_results", web_documents)
        logger.info("tavily_response_count=%s", len(web_documents))
        print(f"tavily_response_count={len(web_documents)}")
        try:
            if web_documents:
                web_store.add_documents(web_documents)
        except Exception as exc:
            _log_stage_failure("vector_store_add_documents", exc)
            return _error_payload(
                "run_research_pipeline",
                "Evidence indexing failed.",
                {"stage": "vector_store_add_documents", "message": str(exc)},
            )

        retrieval_hint = MODE_QUERY_HINTS.get(query_type, MODE_QUERY_HINTS["web"])
        retrieval_query = f"{query}\n{plan_text}\n{retrieval_hint}"
        current_context = _rerank_for_query(query, web_documents, limit=8, mode_key=query_type)
        try:
            cached_context = web_store.similarity_search(query=retrieval_query, top_k=8)
        except Exception as exc:
            _log_stage_failure("vector_store_similarity_search", exc)
            return _error_payload(
                "run_research_pipeline",
                "Evidence retrieval failed.",
                {"stage": "vector_store_similarity_search", "message": str(exc)},
            )
        retrieved_context = _merge_ranked_context(query, query_type, current_context, cached_context, limit=6)

        evidence_text = _build_evidence_text(retrieved_context)
        if len(evidence_text) < 500:
            retry_query = _expanded_query(query, query_type)
            logger.info(
                "research_retry reason=weak_evidence evidence_chars=%s retry_query=%s",
                len(evidence_text),
                retry_query,
            )
            print(f"research_retry reason=weak_evidence evidence_chars={len(evidence_text)} retry_query={retry_query}")
            try:
                retry_data = research(retry_query, plan_text=plan_text)
            except Exception as exc:
                _log_stage_failure("tavily_weak_evidence_retry", exc)
                return _error_payload(
                    "run_research_pipeline",
                    "Live search retry failed after weak evidence.",
                    {"stage": "tavily_weak_evidence_retry", "message": str(exc)},
                )
            retry_documents = _build_web_documents(retry_data)
            _debug_sources("tavily_retry_results", retry_documents)
            try:
                if retry_documents:
                    web_store.add_documents(retry_documents)
            except Exception as exc:
                _log_stage_failure("vector_store_add_retry_documents", exc)
                return _error_payload(
                    "run_research_pipeline",
                    "Evidence indexing failed during retry.",
                    {"stage": "vector_store_add_retry_documents", "message": str(exc)},
                )
            retry_current = _rerank_for_query(query, retry_documents, limit=8, mode_key=query_type)
            try:
                retry_cached = web_store.similarity_search(query=f"{retry_query}\n{retrieval_hint}", top_k=8)
            except Exception as exc:
                _log_stage_failure("vector_store_retry_similarity_search", exc)
                return _error_payload(
                    "run_research_pipeline",
                    "Evidence retrieval failed during retry.",
                    {"stage": "vector_store_retry_similarity_search", "message": str(exc)},
                )
            retrieved_context = _merge_ranked_context(query, query_type, current_context, retry_current, cached_context, retry_cached, limit=6)
            if isinstance(retry_data, dict) and retry_data.get("results"):
                research_data = retry_data
            evidence_text = _build_evidence_text(retrieved_context)

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
        writer_payload["mode"] = mode
        writer_payload["has_resume_context"] = _has_resume_context(query, conversation_context)
        logger.info("writer_input source_count=%s evidence_length=%s", len(writer_payload["sources"]), len(writer_payload["evidence"]))
        print(f"writer_input source_count={len(writer_payload['sources'])} evidence_length={len(writer_payload['evidence'])}")

        if writer_payload["evidence"] is None:
            writer_payload["evidence"] = ""
        if writer_payload["retrieved_context"] is None:
            writer_payload["retrieved_context"] = []

        try:
            answer_payload = write(
                plan_text,
                writer_payload,
                conversation_context=conversation_context,
            )
            logger.info("writer_call=success")
            print("writer_call=success")
        except Exception as exc:
            _log_stage_failure("writer_call", exc)
            return _error_payload(
                "run_research_pipeline",
                "Answer generation failed.",
                {
                    "stage": "writer_call",
                    "message": str(exc),
                    "source_count": source_count,
                    "evidence_chars": len(evidence_text),
                },
            )
        if not isinstance(answer_payload, dict):
            answer_payload = _fallback_answer_payload_by_type(query, query_type)
            fallback_triggered = True
            logger.info("fallback_triggered=writer_non_dict")

        key_findings = _build_key_findings(query, retrieved_context)
        final_report = _answer_payload_to_markdown(answer_payload)
        if _contains_report_style_language(final_report) or _is_generic_response(query, final_report):
            answer_payload = _fallback_answer_payload_by_type(query, query_type)
            final_report = _answer_payload_to_markdown(answer_payload)
            fallback_triggered = True
            logger.info("fallback_triggered=generic_or_report_style")

        logger.info("debug query=%s sources_fetched_count=%s evidence_length=%s fallback_triggered=%s", query, source_count, len(evidence_text), fallback_triggered)
        print(f"debug query={query} sources_fetched_count={source_count} evidence_length={len(evidence_text)} fallback_triggered={fallback_triggered}")

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
            "research": writer_payload,
            "final": final_report,
        }
        logger.info("final_response_keys=%s", sorted(payload.keys()))
        print(f"final_response_keys={sorted(payload.keys())}")
        return _success_payload("run_research_pipeline", payload)
    except Exception as exc:
        logger.exception("endpoint=run_research_pipeline failed")
        return _error_payload("run_research_pipeline", "Research execution failed.", str(exc))
