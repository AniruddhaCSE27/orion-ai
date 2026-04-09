from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def write(plan_text: str, research_data, conversation_context: str = ""):
    context_block = ""
    if conversation_context:
        context_block = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n"

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert report writer. "
                    "Write a polished, compact, professional final report with markdown headings, "
                    "short paragraphs, key insights, and a practical conclusion. "
                    "Use the retrieved context and supplied sources to create source-grounded responses "
                    "designed to reduce hallucinations. "
                    "Do not invent facts, citations, or unsupported claims. "
                    "Do not include a separate Sources section in your answer. "
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
Use the plan and research data below to generate only the Final Report section.
Ground the writing in the retrieved context when available.
Do not add claims that are not supported by the provided materials.
Return clean markdown only.
Use short titled subsections when helpful.
Keep list formatting valid and consistent.

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
