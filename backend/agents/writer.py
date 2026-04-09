import json
import re

from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def _fallback_answer(user_query: str, resume_query: bool, has_resume_context: bool):
    recommendations = [
        "Software Developer",
        "Data Analyst",
        "Machine Learning Engineer",
    ]
    reasons = [
        "These roles stay broadly aligned with technical problem-solving and common software skill paths.",
        "They remain in strong demand across many industries and hiring pipelines.",
        "They provide flexible entry points for building toward more specialized AI or data roles.",
    ]

    if resume_query and not has_resume_context:
        insights = "Based on general assumptions, here are the best roles."
    else:
        insights = f"These recommendations are prioritized to answer the query directly: {user_query}"

    return {
        "recommendations": recommendations[:5],
        "reasons": reasons[:5],
        "insights": insights,
        "improvement_tips": [],
    }


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


def _normalize_answer_payload(payload, user_query: str, resume_query: bool, has_resume_context: bool):
    fallback = _fallback_answer(user_query, resume_query, has_resume_context)
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

    if not recommendations:
        recommendations = fallback["recommendations"]
    if not reasons:
        reasons = fallback["reasons"]
    if not insights:
        insights = fallback["insights"]

    return {
        "recommendations": recommendations,
        "reasons": reasons,
        "insights": insights,
        "improvement_tips": improvement_tips,
    }


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    user_query = research_data.get("user_query", "")
    resume_query = research_data.get("resume_query", False)
    has_resume_context = research_data.get("has_resume_context", False)
    resume_guidance = ""
    if resume_query and not has_resume_context:
        resume_guidance = (
            "\nThe user is asking about their resume/profile/skills, but no resume-specific context was retrieved. "
            "You must explicitly say: 'Based on general assumptions, here are the best roles.'\n"
        )

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an answer engine, not a research report generator. "
                    "Return only valid JSON. "
                    "Never use report sections such as Objective, Data Collection, Analysis Framework, "
                    "Framework, or Implementation Plan. "
                    "Use the retrieved context and supplied sources to create source-grounded responses "
                    "designed to reduce hallucinations. "
                    "Do not invent facts, citations, or unsupported claims. "
                    "Answer the user's exact question first, before giving supporting context. "
                    "Avoid generic introductions and avoid drifting into broad background unless it directly helps answer the query. "
                    "Prefer concise, relevant takeaways over long general summaries. "
                    "Keep the answer crisp, practical, and actionable. "
                    "Use this exact JSON shape: "
                    "{\"recommendations\": [..], \"reasons\": [..], \"insights\": \"..\", \"improvement_tips\": [..]}. "
                    "Recommendations must contain 3 to 5 non-empty strings. "
                    "Reasons must contain short practical bullets. "
                    "Insights must be a short user-focused string. "
                    "improvement_tips is optional but must be an array if present."
                )
            },
            {
                "role": "user",
                "content": f"""
Use the plan and research data below to generate the final answer payload.
Ground the writing in the retrieved context when available.
Do not add claims that are not supported by the provided materials.
Return only valid JSON.
The retrieved context may contain both web sources and indexed document chunks.
Prefer evidence-backed phrasing and keep the answer source-grounded and designed to reduce hallucinations.
If the query asks for ranking, best options, recommendations, or comparisons, answer in that form directly.
If the query is narrow, keep the response narrow.
Do not output any section named Objective, Data Collection, Analysis Framework, Framework, or Implementation Plan.
recommendations must never be empty.
If context is weak, still produce useful fallback recommendations.
{resume_guidance}

USER QUERY:
{user_query}

PLAN:
{plan_text}
{context_block}

RESEARCH DATA:
{research_data}
"""
            }
        ]
    )

    writer_response = response.choices[0].message.content
    print("WRITER OUTPUT:", writer_response)
    parsed = _extract_json_object(writer_response)
    return _normalize_answer_payload(parsed, user_query, resume_query, has_resume_context)
