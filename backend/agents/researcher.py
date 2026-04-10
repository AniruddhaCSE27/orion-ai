import logging
import re

from tavily import TavilyClient

from backend.core.config import config

logger = logging.getLogger(__name__)
tavily = TavilyClient(api_key=config.TAVILY_API_KEY)


def _query_terms(text: str):
    terms = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
        "is", "are", "was", "were", "be", "best", "what", "which", "how", "why",
        "please", "find", "show", "give", "latest", "current",
    }
    return [term for term in terms if term not in stopwords and len(term) > 2]


def _simplify_query(query: str):
    terms = _query_terms(query)
    if not terms:
        return query.strip()
    return " ".join(terms[:8]).strip()


def _expanded_query(query: str):
    simplified = _simplify_query(query)
    if not simplified:
        return query.strip()
    return f"{simplified} 2026 list reviews comparison"


def _is_recommendation_query(query: str):
    lowered = (query or "").lower()
    markers = [
        "best tools", "top tools", "ai tools", "apps", "productivity apps",
        "best apps", "top apps", "best websites", "top websites", "for students",
        "for coding", "coding students", "recommend", "tools for",
    ]
    return any(marker in lowered for marker in markers)


def _recommendation_variants(query: str):
    lowered = (query or "").lower().strip()
    simplified = _simplify_query(query)
    variants = []

    if "student" in lowered:
        variants.extend(
            [
                f"top ai tools for students {simplified}".strip(),
                f"best student productivity AI tools {simplified}".strip(),
            ]
        )
    if "coding" in lowered or "developer" in lowered or "programming" in lowered:
        variants.extend(
            [
                f"latest ai tools for coding students {simplified}".strip(),
                f"top ai coding tools for students {simplified}".strip(),
            ]
        )
    if "app" in lowered:
        variants.append(f"best productivity apps for students {simplified}".strip())
    if not variants:
        variants.extend(
            [
                f"top {simplified} 2026 list".strip(),
                f"best {simplified} recommendations".strip(),
            ]
        )
    return variants


def _candidate_queries(query: str, plan_text: str):
    candidates = []
    base_candidates = [
        (query or "").strip(),
        _simplify_query(query),
        _expanded_query(query),
    ]
    if _is_recommendation_query(query):
        base_candidates.extend(_recommendation_variants(query))
    base_candidates.append(f"{(query or '').strip()} {plan_text[:120].strip()}".strip())

    for item in base_candidates:
        cleaned = " ".join(item.split()).strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates[:6]


def _result_items(result):
    if isinstance(result, dict):
        return result.get("results", []) or []
    return []


def research(query: str, plan_text: str = ""):
    attempts = []
    last_result = {"results": []}

    for candidate in _candidate_queries(query, plan_text):
        logger.info("research_search query=%s", candidate)
        try:
            result = tavily.search(
                query=candidate,
                max_results=6,
                search_depth="advanced",
            )
        except Exception as exc:
            logger.exception("research_search_failed query=%s", candidate)
            attempts.append({"query": candidate, "error": str(exc), "result_count": 0})
            continue

        items = _result_items(result)
        attempts.append({"query": candidate, "result_count": len(items)})
        last_result = result if isinstance(result, dict) else {"results": []}
        logger.info("research_search_results query=%s result_count=%s", candidate, len(items))

        if items:
            last_result = dict(last_result)
            last_result["query_used"] = candidate
            last_result["attempted_queries"] = attempts
            return last_result

    if isinstance(last_result, dict):
        last_result = dict(last_result)
        last_result["query_used"] = ""
        last_result["attempted_queries"] = attempts
        return last_result
    return {"results": [], "query_used": "", "attempted_queries": attempts}
