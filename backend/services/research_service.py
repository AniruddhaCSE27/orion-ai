import logging
import re
import time
from pathlib import Path

from tavily import TavilyClient

from backend.agents.writer import write
from backend.core.config import config
from backend.services.memory import save_to_memory

logger = logging.getLogger(__name__)
tavily = TavilyClient(api_key=config.TAVILY_API_KEY)

RECOMMENDATION_MARKERS = ["best", "top", "tools", "apps", "platforms", "websites"]


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


def _finalize_response(endpoint_name: str, response):
    if response is None:
        response = {"success": False, "error": "Empty response"}
    if not isinstance(response, dict):
        response = {"success": False, "error": "Invalid response", "debug_exception": _ensure_string(response)}

    safe_response = _safe_detail(response)
    if safe_response.get("success"):
        answer = _ensure_string(
            safe_response.get("answer") or safe_response.get("final_report") or safe_response.get("final")
        )
        if not answer:
            safe_response = {"success": False, "error": "Empty response"}
        else:
            safe_response["answer"] = answer
    else:
        safe_response["error"] = _ensure_string(safe_response.get("error") or "Internal server error")

    print(f"FINAL RESPONSE: {safe_response}")
    logger.info("endpoint=%s response_keys=%s", endpoint_name, sorted(safe_response.keys()))
    return safe_response


def _query_type(query: str) -> str:
    lowered = (query or "").lower()
    if any(token in lowered for token in RECOMMENDATION_MARKERS):
        return "recommendation"
    return "research"


def _search_with_retries(query: str, retries: int = 2):
    queries = [query]
    simplified = " ".join(re.findall(r"[a-zA-Z0-9]+", (query or "").lower())[:8]).strip()
    if simplified and simplified not in queries:
        queries.append(simplified)
    if "students" in (query or "").lower():
        queries.append("student productivity ai tools")
    if "best ai tools for students" in (query or "").lower():
        queries.append("top ai tools for students 2026")

    last_error = None
    attempted = []
    for index, candidate in enumerate(queries[: retries + 1]):
        attempted.append(candidate)
        try:
            result = tavily.search(query=candidate, max_results=5, search_depth="advanced")
            raw_results = result.get("results", []) if isinstance(result, dict) else []
            print(f"TAVILY QUERY: {candidate}")
            print(f"TAVILY RESULT COUNT: {len(raw_results)}")
            logger.info("query=%s tavily_query=%s tavily_result_count=%s", query, candidate, len(raw_results))
            if raw_results:
                return result, attempted
        except Exception as exc:
            last_error = exc
            logger.exception("tavily_search_failed query=%s retry_query=%s", query, candidate)
            time.sleep(0.4 * (index + 1))

    if last_error is not None:
        raise last_error
    return {"results": []}, attempted


def _extract_result_content(item):
    for key in ("content", "raw_content", "snippet", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_sources(results):
    sources = []
    for item in (results or [])[:5]:
        title = _ensure_string(item.get("title", "Untitled source")).strip()
        snippet = _ensure_string(_extract_result_content(item)).strip()
        url = _ensure_string(item.get("url", "")).strip()
        if not any([title, snippet, url]):
            continue
        sources.append({"title": title, "content": snippet, "url": url})
    return sources


def _build_evidence(sources):
    blocks = []
    for source in (sources or [])[:5]:
        block_lines = []
        if source.get("title"):
            block_lines.append(f"Title: {source['title']}")
        if source.get("content"):
            block_lines.append(f"Snippet: {source['content']}")
        if source.get("url"):
            block_lines.append(f"URL: {source['url']}")
        if block_lines:
            blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks).strip()


def _source_only_answer(sources, query_type: str) -> str:
    if query_type == "recommendation":
        lines = []
        for source in (sources or [])[:5]:
            title = _ensure_string(source.get("title")).strip()
            snippet = _ensure_string(source.get("content")).strip()
            if not title and not snippet:
                continue
            if title and snippet:
                lines.append(f"- {title}: {snippet}")
            elif title:
                lines.append(f"- {title}")
            else:
                lines.append(f"- {snippet}")
        return "\n".join(lines).strip()

    if not sources:
        return ""
    first = sources[0]
    title = _ensure_string(first.get("title")).strip()
    snippet = _ensure_string(first.get("content")).strip()
    if title and snippet:
        return f"{title}: {snippet}"
    return title or snippet


def _answer_payload_from_answer(answer: str, query_type: str):
    if query_type == "recommendation":
        recommendations = []
        for line in (answer or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("- ", "* ")):
                recommendations.append(stripped[2:].strip())
            else:
                recommendations.append(stripped)
            if len(recommendations) >= 5:
                break
        if not recommendations and answer:
            recommendations = [answer]
        return {
            "primary_title": "Direct Answer",
            "recommendations": recommendations,
            "reasons_title": "Why",
            "reasons": [],
            "insights_title": "Key Insights",
            "insights": "",
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [],
        }

    return {
        "primary_title": "Direct Answer",
        "recommendations": [answer] if answer else [],
        "reasons_title": "Why",
        "reasons": [],
        "insights_title": "Key Insights",
        "insights": "",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
    }


def _answer_payload_to_markdown(answer_payload):
    recommendations = answer_payload.get("recommendations", [])
    lines = ["## Direct Answer"]
    lines.extend(f"- {item}" for item in recommendations if item)
    return "\n".join(lines)


def run_research_pipeline(query: str):
    try:
        result, retry_queries = _search_with_retries(query)
        results = result.get("results", []) if isinstance(result, dict) else []
        sources = _build_sources(results)
        source_count = len(sources)
        evidence = _build_evidence(sources)
        query_type = _query_type(query)

        print(f"QUERY: {query}")
        print(f"DEBUG SOURCE COUNT: {source_count}")
        print(f"DEBUG EVIDENCE LENGTH: {len(evidence)}")
        logger.info(
            "query=%s query_type=%s source_count=%s evidence_length=%s",
            query,
            query_type,
            source_count,
            len(evidence),
        )

        if source_count == 0:
            no_data_answer = "No relevant live data found."
            return _finalize_response(
                "run_research_pipeline",
                {
                    "success": True,
                    "answer": no_data_answer,
                    "sources": [],
                    "web_sources": [],
                    "debug_source_count": 0,
                    "answer_payload": _answer_payload_from_answer(no_data_answer, query_type),
                    "final_report": "## Direct Answer\n- No relevant live data found.",
                    "structured_response": "## Direct Answer\n- No relevant live data found.",
                    "final": "## Direct Answer\n- No relevant live data found.",
                },
            )

        answer_result = write(query=query, query_type=query_type, evidence=evidence)
        final_answer = _ensure_string(answer_result.get("answer")).strip()
        if not final_answer:
            final_answer = _source_only_answer(sources, query_type)

        answer_payload = answer_result.get("answer_payload") or _answer_payload_from_answer(final_answer, query_type)
        if not answer_payload.get("recommendations"):
            answer_payload = _answer_payload_from_answer(final_answer, query_type)
        final_report = _answer_payload_to_markdown(answer_payload)
        save_to_memory(query, final_report)

        return _finalize_response(
            "run_research_pipeline",
            {
                "success": True,
                "answer": final_answer,
                "sources": [{"title": source["title"], "url": source["url"]} for source in sources],
                "web_sources": sources,
                "debug_source_count": source_count,
                "answer_payload": answer_payload,
                "final_report": final_report,
                "structured_response": final_report,
                "final": final_report,
                "debug_retry_used": retry_queries,
            },
        )
    except Exception as exc:
        logger.exception("run_research_pipeline_failed")
        return _finalize_response("run_research_pipeline", {"success": False, "error": str(exc)})
