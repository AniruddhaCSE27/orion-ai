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
                    "Write a polished, compact, professional final report with headings, "
                    "short paragraphs, key insights, and a practical conclusion."
                )
            },
            {
                "role": "user",
                "content": f"""
Use the plan and research data below to generate the final report.

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
