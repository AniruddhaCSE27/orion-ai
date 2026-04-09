from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def plan(query: str, conversation_context: str = ""):
    user_prompt = f"Create a research plan for: {query}"
    if conversation_context:
        user_prompt = (
            "Use the recent conversation context below when it meaningfully improves continuity.\n\n"
            f"RECENT CONTEXT:\n{conversation_context}\n\n"
            f"CURRENT QUERY:\n{query}"
        )

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional research planner. "
                    "Create a concise, structured research plan with headings and bullet points."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content
