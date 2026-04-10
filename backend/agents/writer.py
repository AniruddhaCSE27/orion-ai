import json
import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

GENERIC_META_PHRASES = [
    "use the strongest source",
    "use the highest-relevance sources",
    "focus on the comparison",
    "take the next practical step",
    "lead with the direct answer",
    "end with the most practical next step",
    "follow-up prompts",
    "refine best-fit",
    "answer the query first",
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


def _clean_snippet(text: str, limit: int = 180):
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "..."
    return cleaned


def _context_bullets(retrieved_context, limit: int = 3):
    bullets = []
    for item in retrieved_context[:limit]:
        title = item.get("title", "Retrieved source")
        snippet = _clean_snippet(item.get("content", ""))
        if snippet:
            bullets.append(f"{title}: {snippet}")
    return bullets


def _contains_generic_meta_language(text: str):
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in GENERIC_META_PHRASES)


def _payload_text(payload):
    values = []
    for key in ("primary_title", "reasons_title", "insights_title", "insights", "improvement_title"):
        value = payload.get(key, "")
        if isinstance(value, str):
            values.append(value)
    for key in ("recommendations", "reasons", "improvement_tips"):
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

    recommendations = payload.get("recommendations", [])
    if not recommendations or not isinstance(recommendations, list):
        return False

    combined_text = _payload_text(payload)
    if _contains_generic_meta_language(combined_text):
        return False

    query_terms = _query_terms(user_query)
    lowered_text = combined_text.lower()
    if not query_terms:
        return True

    matches = sum(1 for term in query_terms if term in lowered_text)
    if matches >= 1:
        return True

    # Allow direct answers for short yes/no style questions if they avoid meta language.
    short_query = len(query_terms) <= 2
    first_answer = recommendations[0].lower() if recommendations and isinstance(recommendations[0], str) else ""
    if short_query and any(token in first_answer for token in ["likely", "unlikely", "uncertain", "depends", "yes", "no"]):
        return True

    return False


def _build_evidence_block(retrieved_context):
    if not retrieved_context:
        return "- No retrieved evidence available."
    lines = []
    for item in retrieved_context[:5]:
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


def _mode_defaults(mode_key: str, user_query: str, has_resume_context: bool, retrieved_context):
    context_bullets = _context_bullets(retrieved_context, limit=3)
    best_context = context_bullets[0] if context_bullets else f"Retrieved evidence on '{user_query}' is limited, so the answer should be treated cautiously."
    if mode_key == "resume":
        insights = (
            "These role suggestions combine the current query, recent chat memory, and live market-facing web evidence."
            if has_resume_context
            else "These role suggestions stay practical even when profile context is still limited."
        )
        return {
            "primary_title": "Direct Answer",
            "recommendations": context_bullets or [
                f"For the resume-focused query '{user_query}', the strongest role fit depends on the specific skills and outcomes visible in your profile plus current market demand.",
                "The most credible answer is usually the role family where your projects, tools, and measurable results are strongest.",
                "A narrower target role will produce a more accurate recommendation than a broad resume review prompt.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Resume analysis works best when profile clues are combined with current role expectations from the web.",
                f"The strongest retrieved signal was: {best_context}",
                "The answer is optimized for role fit, credibility, and practical next steps.",
            ],
            "insights_title": "Key Insights",
            "insights": insights,
            "improvement_title": "Improvement Tips",
            "improvement_tips": [
                "Add metrics, impact, and ownership to your best projects.",
                "Tailor your headline and summary to the exact roles you want next.",
            ],
            "extra_sections": [
                {
                    "title": "Suggested Next Steps",
                    "items": [
                        "Sharpen one primary role narrative before sending applications.",
                        "Prepare concise examples that prove technical judgment and results.",
                    ],
                }
            ],
        }

    if mode_key == "study":
        return {
            "primary_title": "Direct Answer",
            "recommendations": context_bullets or [
                f"For '{user_query}', the clearest starting point is the core concept, its scope, and why it matters.",
                "The strongest answer should emphasize the recurring facts, explanations, and likely exam-useful takeaways from the retrieved material.",
                "If you need more depth, the next turn should narrow the topic into definitions, short answers, or long answers.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Study Mode is optimized for teaching clarity, retention, and exam usefulness.",
                f"The strongest retrieved signal was: {best_context}",
                "The answer is grounded in web evidence and then compressed into revision-friendly points.",
            ],
            "insights_title": "Key Insights",
            "insights": "Use the quick bullets first, then ask follow-up questions on any weak concept for deeper understanding.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [
                {
                    "title": "Revision Focus",
                    "items": [
                        "Memorize definitions, classifications, and high-frequency examples.",
                        "Practice one short-answer and one long-answer explanation from the topic.",
                    ],
                }
            ],
        }

    if mode_key == "interview":
        return {
            "primary_title": "Direct Answer",
            "recommendations": context_bullets or [
                f"For the interview-focused query '{user_query}', the most useful answer is the one that ties likely questions to concrete examples and trade-offs.",
                "Strong interview answers should show what you did, why you chose that approach, and what result it produced.",
                "You should expect follow-up questions that test ownership, technical depth, and decision-making.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Interview Prep is optimized for question quality, answer depth, and follow-up readiness.",
                f"The strongest retrieved signal was: {best_context}",
                "The answer stays practical instead of becoming a generic theory dump.",
            ],
            "insights_title": "Key Insights",
            "insights": "Strong interview answers usually combine context, your decision, the trade-off, and the outcome in one compact story.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [
                "Keep answers specific instead of broad or textbook-like.",
                "Prepare one stronger example for system design, collaboration, and debugging.",
            ],
            "extra_sections": [
                {
                    "title": "Follow-up Questions",
                    "items": [
                        "What trade-offs did you make and why?",
                        "How would you improve the solution today?",
                    ],
                }
            ],
        }

    return {
        "primary_title": "Direct Answer",
        "recommendations": context_bullets or [
            f"Based on the retrieved evidence, the answer to '{user_query}' remains uncertain and depends on the strongest factors highlighted in current reporting and analysis.",
            "The best-supported conclusion comes from weighing the top retrieved evidence rather than making a definitive claim beyond the sources.",
            "Where the evidence conflicts or is incomplete, the answer should stay conditional instead of overstated.",
        ],
        "reasons_title": "Why This Answer",
        "reasons": [
            f"The strongest retrieved signal was: {best_context}",
            "Only source-grounded claims should be surfaced in the top section.",
            "The structure is designed for fast reading and multi-turn follow-up."
        ],
        "insights_title": "Key Insights",
        "insights": f"This answer is optimized for the current web-grounded query: {user_query}",
        "improvement_title": "Improvement Tips",
        "improvement_tips": [],
        "extra_sections": [],
    }


def _fallback_answer(user_query: str, mode_key: str, has_resume_context: bool, retrieved_context):
    return _mode_defaults(mode_key, user_query, has_resume_context, retrieved_context)


def _extract_json_object(text: str):
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_extra_sections(value):
    normalized = []
    if not isinstance(value, list):
        return normalized

    for section in value[:6]:
        if not isinstance(section, dict):
            continue
        title = section.get("title", "")
        title = title.strip() if isinstance(title, str) else ""
        items = [
            item.strip()
            for item in section.get("items", [])
            if isinstance(item, str) and item.strip()
        ][:6]
        if title and items:
            normalized.append({"title": title, "items": items})
    return normalized


def _normalize_answer_payload(payload, user_query: str, mode_key: str, has_resume_context: bool, retrieved_context):
    fallback = _fallback_answer(user_query, mode_key, has_resume_context, retrieved_context)
    if not isinstance(payload, dict):
        return fallback

    recommendations = [
        item.strip()
        for item in payload.get("recommendations", [])
        if isinstance(item, str) and item.strip()
    ][:5]
    reasons = [
        item.strip()
        for item in payload.get("reasons", [])
        if isinstance(item, str) and item.strip()
    ][:5]
    insights = payload.get("insights", "")
    insights = insights.strip() if isinstance(insights, str) else ""
    improvement_tips = [
        item.strip()
        for item in payload.get("improvement_tips", [])
        if isinstance(item, str) and item.strip()
    ][:5]

    normalized = {
        "primary_title": payload.get("primary_title", "Direct Answer"),
        "recommendations": recommendations or fallback["recommendations"],
        "reasons_title": payload.get("reasons_title", fallback["reasons_title"]),
        "reasons": reasons or fallback["reasons"],
        "insights_title": payload.get("insights_title", fallback["insights_title"]),
        "insights": insights or fallback["insights"],
        "improvement_title": payload.get("improvement_title", fallback["improvement_title"]),
        "improvement_tips": improvement_tips,
        "extra_sections": _normalize_extra_sections(payload.get("extra_sections", fallback["extra_sections"])),
    }

    if not normalized["improvement_tips"]:
        normalized["improvement_tips"] = fallback.get("improvement_tips", [])
    if not normalized["extra_sections"]:
        normalized["extra_sections"] = fallback.get("extra_sections", [])

    if normalized["primary_title"].strip().lower() != "direct answer":
        normalized["primary_title"] = "Direct Answer"

    return normalized


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    user_query = research_data.get("user_query", "")
    mode_key = research_data.get("query_type", "web")
    mode_label = research_data.get("mode", "Web Research")
    has_resume_context = research_data.get("has_resume_context", False)
    retrieved_context = research_data.get("retrieved_context", []) if isinstance(research_data, dict) else []
    evidence_block = _build_evidence_block(retrieved_context)

    mode_guidance = {
        "resume": (
            "Produce resume-analyzer output with Direct Answer, Why This Answer, Key Insights, "
            "Improvement Tips, and concise next steps."
        ),
        "study": (
            "Produce study-assistant output with concise teaching points, practical revision help, "
            "and exam-useful framing."
        ),
        "interview": (
            "Produce interview-prep output with likely questions, answer angles, and follow-up pressure points."
        ),
        "web": (
            "Produce web-research output with factual or analytical direct answers, concise reasoning, and source-grounded insights."
        ),
    }.get(mode_key, "Produce web-research output with direct factual or analytical answers.")

    def _writer_messages(strict_retry: bool = False):
        retry_block = ""
        if strict_retry:
            retry_block = (
                "\nSTRICT RETRY RULES:\n"
                "- The previous attempt did not answer the user's actual question directly enough.\n"
                "- Your first bullet must be the actual answer, not advice about answering.\n"
                "- Do not say things like 'use the strongest source', 'focus on the comparison', or 'take the next step'.\n"
                "- If the evidence is uncertain, say what the answer most likely is and why it remains uncertain.\n"
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are an answer engine, not a research report generator. "
                    "Return only valid JSON. "
                    "Never use report sections such as Objective, Data Collection, Analysis Framework, Framework, or Implementation Plan. "
                    "Do not write a generic introduction. Answer the query first. "
                    "Use bullets over long paragraphs. Keep the output practical, compact, and scannable. "
                    "Use source-grounded phrasing and avoid unsupported claims. "
                    "Return this exact JSON shape: "
                    "{\"primary_title\":\"...\",\"recommendations\":[...],\"reasons_title\":\"...\",\"reasons\":[...],"
                    "\"insights_title\":\"...\",\"insights\":\"...\",\"improvement_title\":\"...\",\"improvement_tips\":[...],"
                    "\"extra_sections\":[{\"title\":\"...\",\"items\":[...]}]}. "
                    "recommendations must contain 3 to 5 non-empty bullets. "
                    "reasons must contain 2 to 5 non-empty bullets. "
                    "insights must be a short useful paragraph or sentence. "
                    "extra_sections should be used only when they materially help the mode. "
                    "The 'recommendations' list must contain the direct answer itself, never advice about how to answer."
                ),
            },
            {
                "role": "user",
                "content": f"""
Generate a mode-aware answer payload.
Mode: {mode_label}
Mode guidance: {mode_guidance}

Rules:
- Answer the exact user query, not a generic topic.
- The first section must be 'Direct Answer'.
- Keep output practical and concise.
- Use the retrieved evidence and recent memory where relevant.
- If the query is current/latest, prioritize current web evidence.
- In Web Research mode, give a factual or analytical answer with uncertainty only when needed.
- Do not output meta advice, answering instructions, or generic filler.
- Do not use the words objective, framework, analysis framework, or data collection as sections.
- Keep the structure answer-first and follow-up friendly.

USER QUERY:
{user_query}

PLAN:
{plan_text}
{context_block}

RESEARCH DATA:
{research_data}

RETRIEVED EVIDENCE:
{evidence_block}{retry_block}
""",
            },
        ]

    parsed = None
    normalized = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=_writer_messages(strict_retry=attempt == 1),
        )
        writer_response = response.choices[0].message.content
        print("WRITER OUTPUT:", writer_response)
        parsed = _extract_json_object(writer_response)
        normalized = _normalize_answer_payload(
            parsed,
            user_query,
            mode_key,
            has_resume_context,
            retrieved_context,
        )
        if _answer_addresses_query(normalized, user_query):
            return normalized

    return _fallback_answer(user_query, mode_key, has_resume_context, retrieved_context)
