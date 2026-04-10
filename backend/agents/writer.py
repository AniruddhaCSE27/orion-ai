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
        direct_answer = f"The strongest role fit for '{question}' is most likely the role family that matches your clearest projects, technical depth, and measurable outcomes."
        why_points = [
            "Role fit is strongest where your skills and outcomes are easiest to prove.",
            "Market demand matters, but profile evidence matters more than broad claims.",
            "A narrower role target usually produces a much stronger resume strategy.",
        ]
        key_insights = evidence_bullets[:2] or [
            "Your best projects and results should drive the recommendation.",
            "A sharper target role usually improves resume quality and interview conversion.",
        ]
        conclusion = "Choose the role path where your evidence is strongest rather than aiming too broadly."
    elif mode_key == "study":
        direct_answer = f"For '{question}', the clearest starting point is the core concept, what it means, and the most important points connected to it."
        why_points = [
            "Study answers work best when they explain the topic before expanding into detail.",
            "Repeated concepts and likely question angles matter most for revision.",
            "Short, memorable explanations are more useful than broad summaries.",
        ]
        key_insights = evidence_bullets[:2] or [
            "Start with definitions and scope before examples.",
            "Focus on the points most likely to appear in recall-based questions.",
        ]
        conclusion = "The topic becomes easier to remember when reduced to the core idea, examples, and likely exam points."
    elif mode_key == "interview":
        direct_answer = f"For '{question}', the most convincing direction is a concrete example that shows what you did, why you chose it, and what result it produced."
        why_points = [
            "Interviewers trust examples more than abstract claims.",
            "Clear ownership and decision-making make answers stronger.",
            "Outcomes and trade-offs usually separate average answers from strong ones.",
        ]
        key_insights = evidence_bullets[:2] or [
            "Specific examples are more persuasive than generic statements.",
            "Trade-offs and results are usually the most important follow-up areas.",
        ]
        conclusion = "A specific example with clear trade-offs and outcomes is usually the most persuasive direction."
    else:
        direct_answer = f"The most likely answer to '{question}' is uncertain, but the current evidence points to a conditional outcome rather than a simple clear-cut one."
        why_points = [
            "Recent reporting does not support a simple one-sided conclusion.",
            "Major outcomes are usually shaped by several political and strategic pressures at once.",
            "Where evidence is mixed, the safest path is to state the likeliest conclusion with clear uncertainty.",
        ]
        key_insights = evidence_bullets[:2] or [
            "Mixed reporting usually means the final outcome remains uncertain.",
            "Political constraints and external actors often matter as much as raw capability.",
        ]
        conclusion = "The likeliest outcome is usually a contested or uncertain one, not a simple decisive result."

    return {
        "primary_title": "Direct Answer",
        "recommendations": [direct_answer],
        "reasons_title": "Why",
        "reasons": why_points[:5],
        "insights_title": "Key Insights",
        "insights": "\n".join(f"- {item}" for item in key_insights[:3]),
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [
            {"title": "Conclusion", "items": [conclusion]}
        ],
    }


def _parse_writer_output(text: str, question: str, mode_key: str, evidence):
    direct_answer = _clean_text(_extract_section_block(text, "DIRECT ANSWER") or _extract_section_block(text, "FINAL ANSWER"))
    why_points = _extract_bullets(_extract_section_block(text, "WHY"), limit=5)
    key_points = _extract_bullets(_extract_section_block(text, "KEY INSIGHTS") or _extract_section_block(text, "KEY POINTS"), limit=5)
    evidence_points = _extract_bullets(_extract_section_block(text, "EVIDENCE"), limit=4)
    conclusion = _clean_text(_extract_section_block(text, "CONCLUSION"))

    if not direct_answer:
        return _fallback_answer(question, mode_key, evidence)

    return {
        "primary_title": "Direct Answer",
        "recommendations": [direct_answer],
        "reasons_title": "Why",
        "reasons": why_points or _context_bullets(evidence, limit=3),
        "insights_title": "Key Insights",
        "insights": "\n".join(f"- {item}" for item in (key_points or evidence_points or _context_bullets(evidence, limit=2))),
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [{"title": "Conclusion", "items": [conclusion or direct_answer]}],
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

    if any(token in first_answer.lower() for token in ["likely", "unlikely", "yes", "no", "uncertain", "stalemate", "risk", "more likely"]):
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
                "- Start with the real conclusion, even if uncertainty remains.\n"
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
