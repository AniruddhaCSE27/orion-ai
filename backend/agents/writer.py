import json
import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def _mode_defaults(mode_key: str, user_query: str, has_resume_context: bool):
    if mode_key == "resume":
        insights = (
            "These role suggestions combine the current query, recent chat memory, and live market-facing web evidence."
            if has_resume_context
            else "These role suggestions stay practical even when profile context is still limited."
        )
        return {
            "primary_title": "Top Recommendations",
            "recommendations": [
                "Prioritize roles that match your strongest technical evidence and the market demand visible in the retrieved sources.",
                "Lead with measurable projects, tools used, and outcomes instead of generic skills-only claims.",
                "Refine your positioning toward one or two role families before applying broadly.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Resume analysis works best when profile clues are combined with current role expectations from the web.",
                "The answer is optimized for role fit, credibility, and practical next steps.",
                "It stays useful for follow-up questions about strengths, gaps, and targeting strategy.",
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
            "primary_title": "Top Recommendations",
            "recommendations": [
                "Start with the core definition, scope, and why the topic matters before memorizing details.",
                "Focus on recurring concepts, likely exam angles, and cause-effect relationships from the retrieved sources.",
                "Turn the answer into short revision bullets and one or two probable questions for recall practice.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Study Mode is optimized for teaching clarity, retention, and exam usefulness.",
                "The answer is grounded in web evidence and then compressed into revision-friendly points.",
                "It supports follow-up questions such as summaries, explanations, and probable questions.",
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
            "primary_title": "Top Recommendations",
            "recommendations": [
                "Prepare a concise answer for the most likely role-specific question first.",
                "Back each answer with one concrete project, decision, or trade-off you personally handled.",
                "Practice follow-up responses that show ownership, judgment, and measurable impact.",
            ],
            "reasons_title": "Why This Answer",
            "reasons": [
                "Interview Prep is optimized for question quality, answer depth, and follow-up readiness.",
                "The answer stays practical instead of becoming a generic theory dump.",
                "It works well for iterative mock-interview style follow-ups in chat.",
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
        "primary_title": "Top Recommendations",
        "recommendations": [
            "Lead with the direct answer and the strongest evidence-backed takeaway.",
            "Use the highest-relevance sources to compare options or validate the recommendation.",
            "End with the most practical next step based on the retrieved evidence.",
        ],
        "reasons_title": "Why This Answer",
        "reasons": [
            "The answer is ranked around query relevance instead of broad background information.",
            "Only source-grounded claims should be surfaced in the top section.",
            "The structure is designed for fast reading and multi-turn follow-up."
        ],
        "insights_title": "Key Insights",
        "insights": f"This answer is optimized for the current web-grounded query: {user_query}",
        "improvement_title": "Improvement Tips",
        "improvement_tips": [],
        "extra_sections": [],
    }


def _fallback_answer(user_query: str, mode_key: str, has_resume_context: bool):
    return _mode_defaults(mode_key, user_query, has_resume_context)


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


def _normalize_answer_payload(payload, user_query: str, mode_key: str, has_resume_context: bool):
    fallback = _fallback_answer(user_query, mode_key, has_resume_context)
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
        "primary_title": payload.get("primary_title", fallback["primary_title"]),
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

    return normalized


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    user_query = research_data.get("user_query", "")
    mode_key = research_data.get("query_type", "web")
    mode_label = research_data.get("mode", "Web Research")
    has_resume_context = research_data.get("has_resume_context", False)

    mode_guidance = {
        "resume": (
            "Produce resume-analyzer output with Top Recommendations, Why This Answer, Key Insights, "
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
            "Produce web-research output with direct recommendations, concise reasoning, and source-grounded insights."
        ),
    }.get(mode_key, "Produce web-research output with direct recommendations and practical next steps.")

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
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
                    "extra_sections should be used only when they materially help the mode."
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
- Keep output practical and concise.
- Use the retrieved evidence and recent memory where relevant.
- If the query is current/latest, prioritize current web evidence.
- Do not use the words objective, framework, analysis framework, or data collection as sections.
- Keep the structure answer-first and follow-up friendly.

USER QUERY:
{user_query}

PLAN:
{plan_text}
{context_block}

RESEARCH DATA:
{research_data}
""",
            },
        ],
    )

    writer_response = response.choices[0].message.content
    print("WRITER OUTPUT:", writer_response)
    parsed = _extract_json_object(writer_response)
    return _normalize_answer_payload(parsed, user_query, mode_key, has_resume_context)
