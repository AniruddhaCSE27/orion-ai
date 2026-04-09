from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


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
                    "Never use report sections such as Objective, Data Collection, Analysis Framework, "
                    "Framework, or Implementation Plan. "
                    "Use the retrieved context and supplied sources to create source-grounded responses "
                    "designed to reduce hallucinations. "
                    "Do not invent facts, citations, or unsupported claims. "
                    "Do not include a separate Sources section in your answer. "
                    "Answer the user's exact question first, before giving supporting context. "
                    "Avoid generic introductions and avoid drifting into broad background unless it directly helps answer the query. "
                    "Prefer concise, relevant takeaways over long general summaries. "
                    "Keep the answer crisp, practical, and actionable. "
                    "Always use this markdown structure exactly when possible: "
                    "'## Top Recommendations', '## Why These Recommendations', '## Personalized Insight', "
                    "and optionally '## Improvement Tips'. "
                    "The Top Recommendations section should contain 3 to 5 direct answer bullets immediately. "
                    "Follow clean markdown conventions exactly: "
                    "put a space after heading markers like '#', '##', and '###'; "
                    "put a space after numbered items like '1.' and '2.'; "
                    "format bullets as '- item'; "
                    "use bold sparingly and only as '**text**' with proper spacing; "
                    "avoid malformed markdown such as '###Heading' or '4.**Title**'."
                )
            },
            {
                "role": "user",
                "content": f"""
Use the plan and research data below to generate only the answer content.
Ground the writing in the retrieved context when available.
Do not add claims that are not supported by the provided materials.
Return clean markdown only.
Start with the top recommendations answering the user's question.
Use short titled subsections only when helpful.
Keep list formatting valid and consistent.
The retrieved context may contain both web sources and indexed document chunks.
Prefer evidence-backed phrasing and keep the answer source-grounded and designed to reduce hallucinations.
If the query asks for ranking, best options, recommendations, or comparisons, answer in that form directly.
If the query is narrow, keep the response narrow.
Do not output any section named Objective, Data Collection, Analysis Framework, Framework, or Implementation Plan.
Keep paragraphs very short or prefer bullets.
Do not produce long report-style prose.
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

    return response.choices[0].message.content
