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
    markers = ["best", "top", "tools", "apps", "platforms", "websites"]
    return any(marker in lowered for marker in markers)


def _answer_payload_from_text(raw_answer: str, recommendation_query: bool) -> dict:
    if recommendation_query:
        recommendations = []
        for line in raw_answer.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("- ", "* ")):
                recommendations.append(stripped[2:].strip())
            else:
                recommendations.append(stripped)
            if len(recommendations) >= 5:
                break
        if not recommendations and raw_answer:
            recommendations = [raw_answer]
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
        "recommendations": [raw_answer] if raw_answer else [],
        "reasons_title": "Why",
        "reasons": [],
        "insights_title": "Key Insights",
        "insights": "",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [],
    }


def write(query: str, query_type: str, evidence: str) -> dict:
    recommendation_query = query_type == "recommendation" or _is_recommendation_query(query)
    system_prompt = """You are a precise AI research assistant.

RULES:
- ALWAYS answer using the provided evidence
- NEVER say \"not enough data\" if evidence exists
- Be direct and specific
- Use bullet points for recommendations

IF recommendation query:
- Return 3 to 5 named tools or products
- Give each one a one-line explanation

IF research query:
- Give a clear explanation using the evidence
"""

    user_prompt = f"""Query: {query}

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

    return {
        "answer": raw_answer,
        "raw_answer": raw_answer,
        "answer_payload": _answer_payload_from_text(raw_answer, recommendation_query),
    }
