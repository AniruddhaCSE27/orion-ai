from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    lines = []
    for line in str(value).splitlines():
        cleaned = " ".join(line.split()).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _is_recommendation_query(query: str) -> bool:
    lowered = (query or "").lower()
    markers = ["best", "top", "tools", "apps", "platforms", "websites", "software"]
    return any(marker in lowered for marker in markers)


def _section_lines(raw_answer: str):
    direct_answer = []
    reasons = []
    insights = []
    current = "direct_answer"

    for line in raw_answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lowered = stripped.lower().rstrip(":")
        if lowered in {"direct answer", "final answer"}:
            current = "direct_answer"
            continue
        if lowered == "why":
            current = "reasons"
            continue
        if lowered in {"key insights", "insights"}:
            current = "insights"
            continue
        if lowered in {"conclusion", "sources", "evidence"}:
            current = "insights"
            continue

        if stripped.startswith(("- ", "* ")):
            value = stripped[2:].strip()
        else:
            value = stripped

        if current == "reasons":
            reasons.append(value)
        elif current == "insights":
            insights.append(value)
        else:
            direct_answer.append(value)

    return direct_answer, reasons, insights


def _answer_payload_from_text(raw_answer: str, recommendation_query: bool) -> dict:
    direct_answer_lines, reasons, insights = _section_lines(raw_answer)

    if recommendation_query:
        recommendations = direct_answer_lines[:5]
        if not recommendations and raw_answer:
            recommendations = [raw_answer]
        if not reasons:
            reasons = insights[:2]
        insights_text = "\n".join(f"- {item}" for item in insights[:3]) if insights else ""
        answer_text = "\n".join(f"- {item}" for item in recommendations)
    else:
        answer_text = "\n".join(direct_answer_lines).strip() or raw_answer
        recommendations = [answer_text] if answer_text else []
        insights_text = "\n".join(f"- {item}" for item in insights[:3]) if insights else ""

    return {
        "primary_title": "Direct Answer",
        "recommendations": recommendations,
        "reasons_title": "Why",
        "reasons": reasons[:3],
        "insights_title": "Key Insights",
        "insights": insights_text,
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
        "raw_answer": answer_text or raw_answer,
    }


def _fallback_answer_from_sources(query_type: str, sources) -> str:
    if query_type == "recommendation":
        lines = []
        for item in (sources or [])[:5]:
            title = _normalize_text(item.get("title", ""))
            snippet = _normalize_text(item.get("snippet") or item.get("content") or "")
            if title and snippet:
                lines.append(f"- {title}: {snippet}")
            elif title:
                lines.append(f"- {title}")
        return "\n".join(lines).strip()

    top_bits = []
    for item in (sources or [])[:3]:
        title = _normalize_text(item.get("title", ""))
        snippet = _normalize_text(item.get("snippet") or item.get("content") or "")
        if title and snippet:
            top_bits.append(f"{title}: {snippet}")
        elif snippet:
            top_bits.append(snippet)
    return " ".join(top_bits).strip()


def write(query: str, query_type: str, evidence: str, sources=None, conversation_context: str = "") -> dict:
    recommendation_query = query_type == "recommendation" or _is_recommendation_query(query)
    system_prompt = """You are ORION AI's writer, a precise web-grounded research assistant.

Rules:
- Answer the user's exact query directly and specifically.
- Use the provided evidence only.
- If evidence exists, do not say there is not enough data.
- Sound clear, practical, and human.
- Avoid methodology or self-referential language.

Output format:
DIRECT ANSWER:
<main answer>

WHY:
- concise support point
- concise support point

KEY INSIGHTS:
- concise supporting insight
- concise supporting insight

If the query is a recommendation query:
- lead with 3 to 5 named tools, products, or websites
- keep each bullet useful and brief
"""

    context_block = ""
    if conversation_context:
        context_block = f"\nRecent conversation context:\n{conversation_context}\n"

    user_prompt = f"""Query: {query}
Query type: {query_type}
{context_block}
Evidence:
{evidence}
"""

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_answer = _normalize_text(response.choices[0].message.content or "")
    print("WRITER RAW OUTPUT:", raw_answer)

    if not raw_answer:
        raw_answer = _fallback_answer_from_sources(query_type, sources or [])

    payload = _answer_payload_from_text(raw_answer, recommendation_query)
    direct_answer = payload.get("raw_answer") or raw_answer

    if not direct_answer:
        direct_answer = _fallback_answer_from_sources(query_type, sources or [])
        payload = _answer_payload_from_text(direct_answer, recommendation_query)

    return {
        "answer": direct_answer,
        "raw_answer": raw_answer,
        "answer_payload": payload,
    }
