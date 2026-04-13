import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

GENERIC_META_PHRASES = [
    "the answer should",
    "the response should",
    "the clearest response",
    "depends on the retrieved evidence",
    "depends on the evidence",
    "if reporting is mixed",
    "source-backed answer first",
    "this fallback",
    "aligned to the query",
    "best supported by",
    "focus on the strongest source",
    "focus on",
    "state the factors",
    "the most useful answer",
    "methodology",
    "framework",
]


def _query_terms(user_query: str):
    terms = re.findall(r"[a-zA-Z0-9]+", (user_query or "").lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
        "will", "would", "should", "could", "is", "are", "was", "were", "be",
        "me", "my", "your", "this", "that", "these", "those", "what", "which",
        "who", "when", "where", "why", "how", "please",
    }
    return [term for term in terms if term not in stopwords and len(term) > 2]


def _is_recommendation_query(user_query: str):
    lowered = (user_query or "").lower()
    markers = [
        "best", "top", "tools", "apps", "websites", "software",
        "platforms", "resources", "for students", "for coding",
    ]
    return any(marker in lowered for marker in markers)


def _clean_text(value: str):
    return " ".join((value or "").split()).strip()


def _clean_snippet(text: str, limit: int = 220):
    cleaned = _clean_text(text)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "..."
    return cleaned


def _contains_generic_meta_language(text: str):
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in GENERIC_META_PHRASES)


def _extract_section_block(text: str, section_name: str):
    pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n[A-Z ][A-Z ]*:|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_bullets(block: str, limit: int = 5):
    bullets = []
    for line in (block or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return [item for item in bullets if item][:limit]


def _extract_meaningful_lines(text: str, limit: int = 5):
    items = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower().rstrip(":")
        if lowered in {"direct answer", "final answer", "why", "key insights", "key points", "evidence", "conclusion"}:
            continue
        if stripped.startswith(("- ", "* ")):
            stripped = stripped[2:].strip()
        items.append(stripped)
        if len(items) >= limit:
            break
    return items


def _extract_fallback_direct_answer(text: str, limit: int = 5):
    bullets = _extract_bullets(text, limit=limit)
    if bullets:
        return bullets
    lines = _extract_meaningful_lines(text, limit=limit)
    if lines:
        return lines
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if paragraphs:
        return [paragraphs[0]]
    cleaned = _clean_text(text)
    return [cleaned] if cleaned else []


def _build_evidence_block(evidence, limit: int = 5):
    if isinstance(evidence, str):
        cleaned = _clean_text(evidence)
        if len(cleaned) > 2200:
            cleaned = cleaned[:2200].rstrip() + "..."
        return cleaned or "- No retrieved evidence available."
    if not evidence:
        return "- No retrieved evidence available."
    lines = []
    for item in evidence[:limit]:
        title = item.get("title", "Retrieved source")
        url = item.get("url", "")
        snippet = _clean_snippet(item.get("content", ""))
        line = f"- {title}"
        if snippet:
            line += f": {snippet}"
        if url:
            line += f" ({url})"
        lines.append(line)
    return "\n".join(lines)


def _context_bullets(evidence, limit: int = 3):
    bullets = []
    for item in evidence[:limit]:
        title = item.get("title", "Retrieved source")
        snippet = _clean_snippet(item.get("content", ""), limit=180)
        if snippet:
            bullets.append(f"{title}: {snippet}")
    return bullets


def _fallback_answer(question: str, mode_key: str, evidence):
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


def _parse_writer_output(text: str, question: str, mode_key: str, evidence):
    direct_answer_block = _extract_section_block(text, "DIRECT ANSWER") or _extract_section_block(text, "FINAL ANSWER")
    direct_answer = _clean_text(direct_answer_block)
    direct_answer_bullets = _extract_bullets(direct_answer_block, limit=5)
    why_points = _extract_bullets(_extract_section_block(text, "WHY"), limit=5)
    key_points = _extract_bullets(_extract_section_block(text, "KEY INSIGHTS") or _extract_section_block(text, "KEY POINTS"), limit=5)
    evidence_points = _extract_bullets(_extract_section_block(text, "EVIDENCE"), limit=4)
    conclusion = _clean_text(_extract_section_block(text, "CONCLUSION"))

    if not direct_answer:
        fallback_recommendations = _extract_fallback_direct_answer(text, limit=5)
        if not fallback_recommendations:
            return {
                **_fallback_answer(question, mode_key, evidence),
                "raw_answer": _clean_text(text),
                "_debug_parse_success": False,
            }
        return {
            "primary_title": "Direct Answer",
            "recommendations": fallback_recommendations,
            "reasons_title": "Why",
            "reasons": why_points or _context_bullets(evidence, limit=3),
            "insights_title": "Key Insights",
            "insights": "\n".join(f"- {item}" for item in (key_points or evidence_points or _context_bullets(evidence, limit=2))),
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [{"title": "Conclusion", "items": [conclusion or fallback_recommendations[0]]}],
            "raw_answer": _clean_text(text),
            "_debug_parse_success": True,
        }

    return {
        "primary_title": "Direct Answer",
        "recommendations": direct_answer_bullets or [direct_answer],
        "reasons_title": "Why",
        "reasons": why_points or _context_bullets(evidence, limit=3),
        "insights_title": "Key Insights",
        "insights": "\n".join(f"- {item}" for item in (key_points or evidence_points or _context_bullets(evidence, limit=2))),
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [{"title": "Conclusion", "items": [conclusion or direct_answer]}],
        "raw_answer": _clean_text(text),
        "_debug_parse_success": True,
    }


def _payload_text(payload):
    values = []
    for key in ("primary_title", "reasons_title", "insights_title", "insights"):
        value = payload.get(key, "")
        if isinstance(value, str):
            values.append(value)
    for key in ("recommendations", "reasons"):
        for item in payload.get(key, []):
            if isinstance(item, str):
                values.append(item)
    for section in payload.get("extra_sections", []):
        if isinstance(section, dict):
            values.append(section.get("title", ""))
            for item in section.get("items", []):
                if isinstance(item, str):
                    values.append(item)
    return "\n".join(values)


def _answer_addresses_query(payload, user_query: str):
    if not isinstance(payload, dict):
        return False
    direct_answers = payload.get("recommendations", [])
    if not direct_answers:
        return False

    combined_text = _payload_text(payload)
    if _contains_generic_meta_language(combined_text):
        return False

    first_answer = _clean_text(direct_answers[0] if isinstance(direct_answers[0], str) else "")
    if not first_answer:
        return False

    query_terms = _query_terms(user_query)
    lowered_text = combined_text.lower()
    recommendation_query = _is_recommendation_query(user_query)
    if query_terms:
        matches = sum(1 for term in query_terms if term in lowered_text)
        if matches >= 1:
            return True
        if recommendation_query:
            list_items = []
            for key in ("recommendations", "reasons"):
                for item in payload.get(key, []):
                    if isinstance(item, str) and item.strip():
                        list_items.append(item)
            if len(list_items) >= 3:
                return True

    if any(token in first_answer.lower() for token in ["likely", "unlikely", "yes", "no", "risk", "more likely"]):
        return True

    return False


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    question = research_data.get("question") or research_data.get("user_query", "")
    mode_label = research_data.get("mode", "Web Research")
    mode_key = research_data.get("query_type", "web")
    recommendation_query = bool(research_data.get("recommendation_query")) or _is_recommendation_query(question)
    evidence_items = research_data.get("evidence_items") or research_data.get("retrieved_context", [])
    evidence_text = research_data.get("evidence") or ""
    evidence_block = _build_evidence_block(evidence_text or evidence_items)
    sources = research_data.get("sources", [])

    system_prompt = """You are an expert analyst.

Your job is to answer the user's question clearly, directly, and usefully using the retrieved evidence.

RULES:
- Give the actual answer first
- Do not give instructions on how to answer
- Do not use generic filler
- Do not output methodology language
- If the evidence is limited, say so briefly and stay concrete
- Use simple, strong, human-readable language
"""
    if recommendation_query:
        system_prompt += """
- For recommendation or list queries, lead with the best 3 to 5 options from the evidence
- Prefer concrete product, tool, or app names over abstract analysis
- Keep the answer practical and scannable
"""
    system_prompt += """

OUTPUT FORMAT:

DIRECT ANSWER:
<actual answer to the question>

WHY:
- reason 1
- reason 2
- reason 3

KEY INSIGHTS:
- concise supporting point
- concise supporting point

CONCLUSION:
<practical final takeaway>"""

    def _messages(strict_retry: bool = False):
        retry_block = ""
        if strict_retry:
            retry_block = (
                "\nRETRY RULES:\n"
                "- Answer the question directly in plain language. Do not describe how to answer.\n"
                "- Start with the real conclusion and keep it specific.\n"
                "- Avoid meta phrases such as 'the answer should', 'the response should', 'the clearest response', 'if reporting is mixed', or 'best supported by'.\n"
            )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""QUESTION:
{question}

MODE:
{mode_label}

SOURCE URLS:
{chr(10).join(f"- {url}" for url in sources) if sources else "- No source URLs available."}

PLAN:
{plan_text}
{context_block}

RETRIEVED EVIDENCE:
{evidence_block}{retry_block}
""",
            },
        ]

    for attempt in range(2):
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=_messages(strict_retry=attempt == 1),
        )
        writer_response = response.choices[0].message.content
        print("WRITER RAW OUTPUT:", writer_response)
        print("WRITER OUTPUT:", writer_response)
        parsed = _parse_writer_output(writer_response, question, mode_key, evidence_items)
        print("WRITER PARSED OUTPUT:", parsed)
        if _answer_addresses_query(parsed, question):
            return parsed

    fallback_payload = _fallback_answer(question, mode_key, evidence_items)
    fallback_payload["raw_answer"] = _clean_text(writer_response or "")
    fallback_payload["_debug_parse_success"] = False
    return fallback_payload
