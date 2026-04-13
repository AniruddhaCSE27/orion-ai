import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from backend.agents.researcher import research
from backend.agents.writer import write
from backend.core.config import config
from backend.services.memory import format_history_context, save_to_memory
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

RECOMMENDATION_MARKERS = {"best", "top", "tools", "apps", "platforms", "websites", "software"}
COMPARISON_MARKERS = {"vs", "versus", "compare", "comparison", "difference", "better"}
TREND_MARKERS = {"latest", "new", "current", "today", "2025", "2026", "recent", "trend", "trending"}
EXPLANATION_MARKERS = {"what", "why", "how", "explain", "meaning", "guide"}


def _safe_detail(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_detail(item) for item in value]
    return str(value)


def _ensure_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(_safe_detail(value))


def _word_count(text: str) -> int:
    return len((_ensure_string(text)).split())


def _finalize_response(endpoint_name: str, response):
    if response is None:
        response = {"success": False, "error": "Empty response"}
    if not isinstance(response, dict):
        response = {"success": False, "error": "Invalid response", "debug_exception": _ensure_string(response)}

    safe_response = _safe_detail(response)
    if safe_response.get("success"):
        best_answer = _ensure_string(
            safe_response.get("direct_answer")
            or safe_response.get("answer")
            or safe_response.get("report")
            or safe_response.get("final_report")
            or safe_response.get("final")
        ).strip()
        if not best_answer:
            safe_response = {"success": False, "error": "Empty response"}
        else:
            safe_response["answer"] = best_answer
            safe_response["direct_answer"] = best_answer
    else:
        safe_response["error"] = _ensure_string(safe_response.get("error") or "Internal server error")

    print(f"FINAL RESPONSE: {safe_response}")
    logger.info("endpoint=%s response_keys=%s", endpoint_name, sorted(safe_response.keys()))
    return safe_response


def _query_terms(text: str):
    return [
        term for term in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(term) > 2 and term not in {"the", "and", "for", "with", "that", "this", "from"}
    ]


def _classify_query(query: str) -> str:
    lowered = (query or "").lower()
    tokens = set(_query_terms(query))
    if tokens & RECOMMENDATION_MARKERS:
        return "recommendation"
    if tokens & COMPARISON_MARKERS:
        return "comparison"
    if tokens & TREND_MARKERS:
        return "trend"
    if tokens & EXPLANATION_MARKERS:
        return "explanatory"
    return "factual"


def _mode_for_query_type(query_type: str) -> str:
    if query_type == "recommendation":
        return "Web Recommendations"
    if query_type == "comparison":
        return "Web Comparison"
    if query_type == "trend":
        return "Trend Analysis"
    if query_type == "explanatory":
        return "Explainer"
    return "Web Research"


def _build_plan(query: str, query_type: str) -> str:
    intent_map = {
        "recommendation": "Identify the best options, why they stand out, and who each one fits best.",
        "comparison": "Compare the most relevant options on the dimensions the query implies.",
        "trend": "Summarize the latest developments and explain what changed recently.",
        "explanatory": "Explain the topic clearly, then support it with current web evidence.",
        "factual": "Answer the question directly using the strongest current evidence available.",
    }
    plan_lines = [
        "## Research Plan",
        f"- Query type: {query_type}",
        f"- Objective: {intent_map.get(query_type, intent_map['factual'])}",
        "- Retrieval: search current web sources and normalize the strongest supporting snippets.",
        "- Delivery: return an answer-first response with source-backed takeaways.",
    ]
    if query:
        plan_lines.insert(1, f"- Query: {query}")
    return "\n".join(plan_lines)


def _extract_result_content(item: dict) -> str:
    for key in ("content", "raw_content", "snippet", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()).strip()
    return ""


def _normalize_evidence_item(item: dict, rank: int, query_terms) -> Optional[dict]:
    title = _ensure_string(item.get("title", "Untitled source")).strip()
    snippet = _extract_result_content(item)
    url = _ensure_string(item.get("url", "")).strip()
    content = _ensure_string(item.get("raw_content") or snippet).strip()

    if not any([title, snippet, content]):
        return None

    haystack = " ".join(part for part in [title, snippet, content] if part).lower()
    keyword_hits = sum(1 for term in query_terms if term in haystack)
    score = round(keyword_hits + (1 / max(rank, 1)), 4)
    source_name = urlparse(url).netloc.replace("www.", "") if url else "web"

    return {
        "title": title or "Untitled source",
        "url": url,
        "snippet": snippet,
        "content": content,
        "source_name": source_name,
        "score": score,
        "rank": rank,
    }


def _normalize_results(results, query: str):
    query_terms = _query_terms(query)
    normalized = []
    seen = set()
    for rank, item in enumerate(results or [], start=1):
        if not isinstance(item, dict):
            continue
        normalized_item = _normalize_evidence_item(item, rank, query_terms)
        if not normalized_item:
            continue
        key = (
            normalized_item["url"].lower(),
            normalized_item["title"].lower(),
            normalized_item["snippet"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_item)
    return normalized


def _rerank_evidence(query: str, evidence_items):
    query_terms = _query_terms(query)

    def score(item):
        haystack = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("content", "")]).lower()
        term_hits = sum(1 for term in query_terms if term in haystack)
        snippet_bonus = min(len(item.get("snippet", "")) / 240.0, 1.0)
        return (term_hits * 5) + item.get("score", 0) + snippet_bonus

    reranked = sorted(evidence_items, key=score, reverse=True)
    return reranked[:5] or evidence_items[:5]


def _build_vector_documents(evidence_items):
    documents = []
    for item in evidence_items:
        text = "\n".join(
            part for part in [
                item.get("title", ""),
                item.get("snippet", ""),
                item.get("content", ""),
                item.get("url", ""),
            ] if part
        ).strip()
        if not text:
            continue
        documents.append(
            {
                "title": item.get("title", "Untitled source"),
                "url": item.get("url", ""),
                "text": text,
                "content": item.get("snippet") or item.get("content") or "",
                "source_name": item.get("source_name", "web"),
            }
        )
    return documents


def _augment_with_vector_rerank(query: str, evidence_items):
    if not evidence_items or not config.OPENAI_API_KEY:
        return evidence_items

    try:
        store = VectorStore(namespace="orion_web_cache")
        store.add_documents(_build_vector_documents(evidence_items))
        similar = store.similarity_search(query=query, top_k=min(3, len(evidence_items)))
    except Exception:
        logger.exception("vector_rerank_failed")
        return evidence_items

    if not similar:
        return evidence_items

    merged = []
    seen = set()
    source_lookup = {
        (
            item.get("url", "").lower(),
            item.get("title", "").lower(),
        ): item
        for item in evidence_items
    }

    for item in similar:
        key = (item.get("url", "").lower(), item.get("title", "").lower())
        original = source_lookup.get(key)
        if original and key not in seen:
            merged.append(original)
            seen.add(key)

    for item in evidence_items:
        key = (item.get("url", "").lower(), item.get("title", "").lower())
        if key not in seen:
            merged.append(item)
            seen.add(key)

    return merged[:5]


def _build_evidence_block(evidence_items):
    blocks = []
    for item in evidence_items[:5]:
        block_lines = []
        if item.get("title"):
            block_lines.append(f"Title: {item['title']}")
        if item.get("snippet"):
            block_lines.append(f"Snippet: {item['snippet']}")
        elif item.get("content"):
            block_lines.append(f"Snippet: {item['content']}")
        if item.get("url"):
            block_lines.append(f"URL: {item['url']}")
        if block_lines:
            blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks).strip()


def _sentence_excerpt(text: str, limit: int = 180) -> str:
    cleaned = " ".join((_ensure_string(text)).split()).strip()
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0]
    return sentence[:limit].rstrip() + ("..." if len(sentence) > limit else "")


def _extract_named_recommendations(evidence_items):
    recommendations = []
    seen = set()
    for item in evidence_items:
        title = _ensure_string(item.get("title")).strip()
        snippet = _sentence_excerpt(item.get("snippet") or item.get("content"))
        name = re.split(r"[:|\-–—]", title, maxsplit=1)[0].strip() if title else ""
        if not name:
            continue
        normalized_name = name.lower()
        if normalized_name in seen:
            continue
        seen.add(normalized_name)
        line = f"{name}: {snippet}" if snippet else name
        recommendations.append(line)
        if len(recommendations) >= 5:
            break
    return recommendations


def _build_grounded_fallback_answer(query: str, query_type: str, evidence_items):
    if query_type == "recommendation":
        recommendations = _extract_named_recommendations(evidence_items)
        if recommendations:
            return "\n".join(f"- {item}" for item in recommendations)

    top_items = []
    for item in evidence_items[:3]:
        title = item.get("title", "")
        snippet = _sentence_excerpt(item.get("snippet") or item.get("content"))
        if title and snippet:
            top_items.append(f"{title}: {snippet}")
        elif snippet:
            top_items.append(snippet)
        elif title:
            top_items.append(title)

    if query_type == "comparison" and len(top_items) >= 2:
        return " ".join(top_items[:2])
    if top_items:
        return " ".join(top_items)
    return "No relevant live data found."


def _answer_payload_from_parts(direct_answer: str, query_type: str, reasons=None, insights=None):
    reasons = [item for item in (reasons or []) if _ensure_string(item).strip()][:3]
    insights = [item for item in (insights or []) if _ensure_string(item).strip()][:3]

    if query_type == "recommendation":
        recommendations = []
        for line in _ensure_string(direct_answer).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("- ", "* ")):
                recommendations.append(stripped[2:].strip())
            else:
                recommendations.append(stripped)
        if not recommendations and direct_answer:
            recommendations = [direct_answer]
    else:
        recommendations = [_ensure_string(direct_answer).strip()] if _ensure_string(direct_answer).strip() else []

    return {
        "primary_title": "Direct Answer",
        "recommendations": recommendations[:5],
        "reasons_title": "Why",
        "reasons": reasons,
        "insights_title": "Key Insights",
        "insights": "\n".join(f"- {item}" for item in insights) if insights else "",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
        "raw_answer": _ensure_string(direct_answer).strip(),
    }


def _build_report(answer_payload, sources):
    sections = ["## Direct Answer"]
    recommendations = answer_payload.get("recommendations", [])
    sections.extend(f"- {item}" for item in recommendations if item)

    reasons = answer_payload.get("reasons", [])
    if reasons:
        sections.append("\n## Why")
        sections.extend(f"- {item}" for item in reasons)

    insights = _ensure_string(answer_payload.get("insights")).strip()
    if insights:
        sections.append("\n## Key Insights")
        sections.append(insights)

    sections.append("\n## Sources")
    if sources:
        sections.extend(
            f"- [{item['title']}]({item['url']})" if item.get("url") else f"- {item['title']}"
            for item in sources
        )
    else:
        sections.append("- No sources available.")
    return "\n".join(sections).strip()


def run_research_pipeline(query: str):
    try:
        query = _ensure_string(query).strip()
        if not query:
            return _finalize_response("run_research_pipeline", {"success": False, "error": "Query is required."})
        if not config.TAVILY_API_KEY:
            return _finalize_response(
                "run_research_pipeline",
                {"success": False, "error": "Missing TAVILY_API_KEY."},
            )
        if not config.OPENAI_API_KEY:
            return _finalize_response(
                "run_research_pipeline",
                {"success": False, "error": "Missing OPENAI_API_KEY."},
            )

        query_type = _classify_query(query)
        mode = _mode_for_query_type(query_type)
        plan_text = _build_plan(query, query_type)
        conversation_context = format_history_context(limit=4)

        logger.info("query_received=%s query_type=%s mode=%s", query, query_type, mode)
        print(f"QUERY RECEIVED: {query}")

        research_data = research(query, plan_text=plan_text)
        raw_results = research_data.get("results", []) if isinstance(research_data, dict) else []
        logger.info("search_result_count=%s", len(raw_results))
        print(f"SEARCH RESULT COUNT: {len(raw_results)}")

        normalized_results = _normalize_results(raw_results, query)
        reranked_results = _rerank_evidence(query, normalized_results)
        reranked_results = _augment_with_vector_rerank(query, reranked_results)
        evidence_block = _build_evidence_block(reranked_results)

        usable_evidence_count = len(reranked_results)
        logger.info(
            "usable_evidence_count=%s reranked_evidence_count=%s evidence_length=%s",
            usable_evidence_count,
            len(reranked_results),
            len(evidence_block),
        )
        print(f"USABLE EVIDENCE COUNT: {usable_evidence_count}")
        print(f"EVIDENCE LENGTH: {len(evidence_block)}")

        if usable_evidence_count == 0 or not evidence_block:
            no_data_answer = "No relevant live data found."
            answer_payload = _answer_payload_from_parts(no_data_answer, query_type)
            report = _build_report(answer_payload, [])
            payload = {
                "success": True,
                "query": query,
                "mode": mode,
                "answer": no_data_answer,
                "direct_answer": no_data_answer,
                "plan": plan_text,
                "evidence": [],
                "sources": [],
                "source_count": 0,
                "report": report,
                "report_word_count": _word_count(report),
                "plan_word_count": _word_count(plan_text),
                "answer_payload": answer_payload,
                "final_report": report,
                "structured_response": report,
                "final": report,
                "query_type": query_type,
                "web_sources": [],
                "research": {
                    "results": [],
                    "query_used": research_data.get("query_used", "") if isinstance(research_data, dict) else "",
                    "attempted_queries": research_data.get("attempted_queries", []) if isinstance(research_data, dict) else [],
                },
                "debug": {
                    "raw_result_count": len(raw_results),
                    "usable_evidence_count": 0,
                    "reranked_evidence_count": 0,
                    "writer_output_length": 0,
                    "fallback_triggered_reason": "no_usable_search_results",
                },
            }
            return _finalize_response("run_research_pipeline", payload)

        writer_output = write(
            query=query,
            query_type=query_type,
            evidence=evidence_block,
            sources=reranked_results,
            conversation_context=conversation_context,
        )
        logger.info("writer_invoked=true writer_output_length=%s", len(_ensure_string(writer_output.get("answer"))))
        print(f"WRITER OUTPUT LENGTH: {len(_ensure_string(writer_output.get('answer')))}")

        direct_answer = _ensure_string(writer_output.get("answer")).strip()
        if not direct_answer:
            direct_answer = _build_grounded_fallback_answer(query, query_type, reranked_results)
            fallback_reason = "writer_blank_used_evidence_fallback"
        else:
            fallback_reason = ""

        answer_payload = writer_output.get("answer_payload") or _answer_payload_from_parts(direct_answer, query_type)
        if not answer_payload.get("recommendations"):
            answer_payload = _answer_payload_from_parts(direct_answer, query_type)

        if not _ensure_string(answer_payload.get("raw_answer")).strip():
            answer_payload["raw_answer"] = direct_answer

        sources = [
            {
                "title": item.get("title", "Untitled source"),
                "url": item.get("url", ""),
                "content": item.get("snippet", ""),
                "source_name": item.get("source_name", "web"),
                "score": item.get("score", 0),
                "rank": item.get("rank", 0),
            }
            for item in reranked_results
        ]
        report = _build_report(answer_payload, sources)
        save_to_memory(query, report)

        payload = {
            "success": True,
            "query": query,
            "mode": mode,
            "answer": direct_answer,
            "direct_answer": direct_answer,
            "plan": plan_text,
            "evidence": reranked_results,
            "sources": sources,
            "source_count": len(sources),
            "report": report,
            "report_word_count": _word_count(report),
            "plan_word_count": _word_count(plan_text),
            "answer_payload": answer_payload,
            "final_report": report,
            "structured_response": report,
            "final": report,
            "query_type": query_type,
            "web_sources": sources,
            "research": {
                "results": sources,
                "query_used": research_data.get("query_used", "") if isinstance(research_data, dict) else "",
                "attempted_queries": research_data.get("attempted_queries", []) if isinstance(research_data, dict) else [],
            },
            "debug": {
                "raw_result_count": len(raw_results),
                "usable_evidence_count": usable_evidence_count,
                "reranked_evidence_count": len(reranked_results),
                "writer_output_length": len(direct_answer),
                "fallback_triggered_reason": fallback_reason,
            },
        }
        return _finalize_response("run_research_pipeline", payload)
    except Exception as exc:
        logger.exception("run_research_pipeline_failed")
        return _finalize_response(
            "run_research_pipeline",
            {"success": False, "error": str(exc)},
        )
