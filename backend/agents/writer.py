import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

GENERIC_META_PHRASES = [
    "depends on the evidence",
    "focus on the strongest source",
    "focus on",
    "state the factors",
    "the most useful answer is",
    "strongest source-backed answer",
    "the answer should",
    "how to answer",
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
    pattern = rf"{section_name}:\s*(.*-)(-=\n[A-Z ]+:|\Z)"
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


def _build_evidence_block(evidence, limit: int = 5):
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
    evidence_bullets = _context_bullets(evidence, limit=3)
    if mode_key == "resume":
        answer = f"The best role fit for '{question}' is whichever path is most strongly supported by your actual project outcomes, tools, and measurable impact."
        conclusion = "A narrower role target will produce a sharper resume recommendation than a broad career question."
    elif mode_key == "study":
        answer = f"The clearest way to answer '{question}' is to start with the core concept, then move to the most exam-relevant points and examples."
        conclusion = "The strongest study answer is the one that makes the topic easy to recall under exam pressure."
    elif mode_key == "interview":
        answer = f"The strongest interview response to '{question}' is the one backed by a concrete example, a decision you made, and a clear outcome."
        conclusion = "Specific examples beat broad textbook-style answers in interview settings."
    else:
        answer = f"Based on the retrieved reporting, the most likely answer to '{question}' is the conclusion best supported by the strongest current evidence, even if some uncertainty remains."
        conclusion = "The conclusion should stay grounded in current reporting and avoid overstating certainty when the evidence is mixed."

    reasons = evidence_bullets or [
        "Recent reporting and analysis were used to build the answer.",
        "The conclusion was kept cautious where certainty was limited.",
        "The response was shaped around the user's exact question rather than a generic topic summary.",
    ]
    evidence_lines = evidence_bullets[:2] or ["Retrieved evidence was limited, so this conclusion should be treated cautiously."]
    return {
        "primary_title": "Direct Answer",
        "recommendations": [answer],
        "reasons_title": "Key Points",
        "reasons": reasons[:5],
        "insights_title": "Evidence",
        "insights": "\n".join(f"- {item}" for item in evidence_lines),
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [
            {
                "title": "Conclusion",
                "items": [conclusion],
            }
        ],
    }


def _parse_writer_output(text: str, question: str, mode_key: str, evidence):
    final_answer = _clean_text(_extract_section_block(text, "FINAL ANSWER"))
    key_points = _extract_bullets(_extract_section_block(text, "KEY POINTS"), limit=5)
    evidence_points = _extract_bullets(_extract_section_block(text, "EVIDENCE"), limit=4)
    conclusion = _clean_text(_extract_section_block(text, "CONCLUSION"))

    if not final_answer:
        return _fallback_answer(question, mode_key, evidence)

    return {
        "primary_title": "Direct Answer",
        "recommendations": [final_answer],
        "reasons_title": "Key Points",
        "reasons": key_points or _context_bullets(evidence, limit=3),
        "insights_title": "Evidence",
        "insights": "\n".join(f"- {item}" for item in evidence_points) if evidence_points else "\n".join(
            f"- {item}" for item in (_context_bullets(evidence, limit=2) or ["Retrieved evidence was limited."])
        ),
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [
            {
                "title": "Conclusion",
                "items": [conclusion or final_answer],
            }
        ],
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
    if query_terms:
        matches = sum(1 for term in query_terms if term in lowered_text)
        if matches >= 1:
            return True

    if any(token in first_answer.lower() for token in ["likely", "unlikely", "yes", "no", "uncertain", "risk", "more likely"]):
        return True

    return False


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    question = research_data.get("question") or research_data.get("user_query", "")
    mode_label = research_data.get("mode", "Web Research")
    mode_key = research_data.get("query_type", "web")
    evidence = research_data.get("evidence") or research_data.get("retrieved_context", [])
    evidence_block = _build_evidence_block(evidence)

    system_prompt = """You are an expert analyst.

Your job is to answer the user's question clearly, directly, and usefully using the retrieved evidence.

RULES:
- Give the actual answer first
- Do not give instructions on how to answer
- Do not use generic filler
- Do not output methodology language
- If uncertain, provide the most likely answer and explain briefly why
- Use simple, strong, human-readable language

OUTPUT FORMAT:

FINAL ANSWER:
<clear direct answer in 1–2 lines>

KEY POINTS:
- Bullet 1
- Bullet 2
- Bullet 3

EVIDENCE:
- Source-backed point 1
- Source-backed point 2

CONCLUSION:
<short final takeaway>"""

    def _messages(strict_retry: bool = False):
        retry_block = ""
        if strict_retry:
            retry_block = (
                "\nRETRY RULES:\n"
                "- The previous answer was too generic, meta, or indirect.\n"
                "- Answer the question itself in the first 1-2 lines.\n"
                "- Do not use phrases like 'depends on the evidence', 'focus on', 'the answer should', or 'the most useful answer is'.\n"
            )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""QUESTION:
{question}

MODE:
{mode_label}

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
        print("WRITER OUTPUT:", writer_response)
        parsed = _parse_writer_output(writer_response, question, mode_key, evidence)
        if _answer_addresses_query(parsed, question):
            return parsed

    return _fallback_answer(question, mode_key, evidence)
