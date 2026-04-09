import json
import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def _mode_defaults(mode_key: str, user_query: str, has_resume_context: bool):
    if mode_key == "resume":
        insights = (
            "Based on general assumptions, here are the best roles."
            if not has_resume_context
            else f"These role suggestions stay tied to the profile-oriented query: {user_query}"
        )
        return {
            "primary_title": "Best Roles for You",
            "recommendations": [
                "Product-focused software roles",
                "Data-heavy analyst roles",
                "Automation and AI-assisted implementation roles",
            ],
            "reasons_title": "Why These Roles",
            "reasons": [
                "They align broadly with technical problem-solving, project work, and transferable digital skills.",
                "They offer practical entry paths while leaving room to specialize later.",
                "They stay useful even when resume context is limited or incomplete.",
            ],
            "insights_title": "Strengths in Your Profile",
            "insights": insights,
            "improvement_title": "Improvement Suggestions",
            "improvement_tips": [
                "Add measurable project impact and concrete tools used.",
                "Highlight the strongest domain or role focus more clearly.",
            ],
            "extra_sections": [
                {
                    "title": "Skill Gaps",
                    "items": [
                        "Clarify depth in one target stack or domain.",
                        "Show stronger evidence of impact, ownership, or outcomes.",
                    ],
                },
                {
                    "title": "Suggested Next Steps",
                    "items": [
                        "Tailor the resume to one role family at a time.",
                        "Prepare 2 to 3 project stories with outcomes and metrics.",
                    ],
                },
            ],
        }

    if mode_key == "study":
        return {
            "primary_title": "Chapter Summary",
            "recommendations": [
                "Main ideas and high-priority subtopics from the material",
                "Probable exam-style questions based on the strongest chunks",
                "Fast revision points to review before practice",
            ],
            "reasons_title": "Important Topics",
            "reasons": [
                "These points are chosen from the most relevant study-oriented document chunks.",
                "They are the areas most likely to produce direct short-answer or long-answer questions.",
                "The output stays concise so it is useful for revision, not just reading.",
            ],
            "insights_title": "Revision Points",
            "insights": "Focus first on concepts, definitions, and repeated themes that appear across the retrieved chapter content.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [
                {"title": "2-mark Questions", "items": ["Document not properly indexed. Try re-indexing."]},
                {"title": "5-mark Questions", "items": ["Document not properly indexed. Try re-indexing."]},
                {"title": "10-mark Questions", "items": ["Document not properly indexed. Try re-indexing."]},
            ],
        }

    if mode_key == "document":
        return {
            "primary_title": "Summary",
            "recommendations": [
                "Core points extracted from the uploaded material",
                "Most relevant supporting details tied to the query",
                "A concise next step for deeper review or questioning",
            ],
            "reasons_title": "Important Topics",
            "reasons": [
                "The answer is grounded in uploaded content before external sources.",
                "The selected points are ranked by closeness to the exact question.",
                "The output is designed to stay practical and easy to scan.",
            ],
            "insights_title": "Key Definitions / Concepts",
            "insights": "Document not properly indexed. Try re-indexing.",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [
                {"title": "Probable Questions", "items": ["Document not properly indexed. Try re-indexing."]},
                {"title": "Short Answer Questions", "items": ["Document not properly indexed. Try re-indexing."]},
                {"title": "Long Answer Questions", "items": ["Document not properly indexed. Try re-indexing."]},
            ],
        }

    if mode_key == "interview":
        return {
            "primary_title": "Top Interview Questions",
            "recommendations": [
                "Tell me about yourself and your most relevant project.",
                "What technical decisions did you make, and why?",
                "Where would you improve your approach if you repeated the work today?",
            ],
            "reasons_title": "Focus Areas",
            "reasons": [
                "These questions test understanding, ownership, and communication.",
                "They are useful even when resume or document context is limited.",
                "They can be expanded into a mock interview quickly.",
            ],
            "insights_title": "Difficulty Level",
            "insights": "Start with foundational questions, then move into project depth and follow-ups.",
            "improvement_title": "Follow-up Questions",
            "improvement_tips": [
                "What trade-offs did you face?",
                "How would you measure success?",
            ],
            "extra_sections": [
                {
                    "title": "Short Model Answers",
                    "items": [
                        "Use one concise project example with your role, action, and outcome.",
                        "Keep answers specific and tied to tools, decisions, and impact.",
                    ],
                }
            ],
        }

    if mode_key == "web":
        return {
            "primary_title": "Top Recommendations",
            "recommendations": [
                "Most relevant direct answer based on current web-backed evidence",
                "Strongest supporting comparison or takeaway",
                "Most practical next step based on the evidence",
            ],
            "reasons_title": "Why These",
            "reasons": [
                "The answer is prioritized using web sources for current or comparative queries.",
                "Only the most query-relevant evidence is surfaced in the top section.",
                "The output stays concise instead of expanding into a generic report.",
            ],
            "insights_title": "Quick Insights",
            "insights": f"This answer is optimized for the web-oriented query: {user_query}",
            "improvement_title": "Improvement Tips",
            "improvement_tips": [],
            "extra_sections": [],
        }

    return {
        "primary_title": "Direct Answer",
        "recommendations": [
            "Most relevant answer point based on the available evidence",
            "Most useful supporting point tied to the query",
            "Most practical follow-up direction",
        ],
        "reasons_title": "Why This Answer",
        "reasons": [
            "The answer is kept aligned to the exact query rather than broad background.",
            "Retrieved evidence is ranked by relevance before writing.",
            "Fallback stays useful without drifting into unrelated recommendations.",
        ],
        "insights_title": "Personalized Insight",
        "insights": f"This response stays aligned to the general query: {user_query}",
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
    mode_key = research_data.get("query_type", "general")
    mode_label = research_data.get("mode", "General Mode")
    has_resume_context = research_data.get("has_resume_context", False)
    has_document_context = research_data.get("has_document_context", False)
    strong_document_match = research_data.get("strong_document_match", False)

    mode_guidance = {
        "resume": (
            "Produce resume-coach style output with sections such as Best Roles for You, Why These Roles, "
            "Strengths in Your Profile, Skill Gaps, Improvement Suggestions, and Suggested Next Steps."
        ),
        "study": (
            "Produce study-assistant output with Chapter Summary, Important Topics, 2-mark Questions, "
            "5-mark Questions, 10-mark Questions, and Revision Points."
        ),
        "document": (
            "Produce document-analysis output with Summary, Important Topics, Probable Questions, "
            "Short Answer Questions, Long Answer Questions, and Key Definitions / Concepts."
        ),
        "interview": (
            "Produce interview-prep output with Top Interview Questions, Short Model Answers, "
            "Difficulty Level, Focus Areas, and Follow-up Questions."
        ),
        "web": (
            "Produce web-research output with Top Recommendations, Why These, and Quick Insights."
        ),
        "general": (
            "Produce a concise direct answer with supporting reasons and practical next steps."
        ),
    }.get(mode_key, "Produce a concise direct answer.")

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
                    "extra_sections should be used for mode-specific sections."
                )
            },
            {
                "role": "user",
                "content": f"""
Generate a mode-aware answer payload.
Mode: {mode_label}
Mode guidance: {mode_guidance}

Rules:
- Answer the exact user query, not a generic topic.
- Keep output practical and recruiter-grade or exam-useful where relevant.
- If uploaded document context is strong, prefer it first.
- If the query is current/latest, web evidence can lead.
- If context is weak, produce a mode-aware fallback only.
- Do not use the words objective, framework, analysis framework, or data collection as sections.

USER QUERY:
{user_query}

PLAN:
{plan_text}
{context_block}

RESEARCH DATA:
{research_data}

DOCUMENT CONTEXT STATUS:
- has_document_context: {has_document_context}
- strong_document_match: {strong_document_match}
"""
            }
        ]
    )

    writer_response = response.choices[0].message.content
    print("WRITER OUTPUT:", writer_response)
    parsed = _extract_json_object(writer_response)
    return _normalize_answer_payload(parsed, user_query, mode_key, has_resume_context)
