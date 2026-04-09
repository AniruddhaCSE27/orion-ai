from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def plan(query: str, conversation_context: str = "", query_type: str = "general"):
    user_prompt = (
        "Interpret the query precisely and create a focused research plan.\n\n"
        f"QUERY TYPE:\n{query_type}\n\n"
        f"USER QUERY:\n{query}"
    )
    if conversation_context:
        user_prompt = (
            "Use the recent conversation context below when it meaningfully improves continuity.\n\n"
            f"RECENT CONTEXT:\n{conversation_context}\n\n"
            "Interpret the current query precisely and create a focused research plan.\n\n"
            f"QUERY TYPE:\n{query_type}\n\n"
            f"CURRENT QUERY:\n{query}"
        )

    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                "You are a professional research planner. "
                    "Create a concise, structured plan that stays tightly aligned to the user's exact ask. "
                    "First identify the user's intent, scope, desired output type, and the best product mode. "
                    "If the query type is 'study', focus on important topics, revision priorities, and exam-style questions. "
                    "If the query type is 'document', focus on summarization, document-grounded explanation, and likely questions. "
                    "If the query type is 'resume', focus on profile fit, role ranking, strengths, gaps, and next steps. "
                    "If the query type is 'interview', focus on likely interview questions, model answers, and focus areas. "
                    "If the query type is 'web', focus on current comparisons, latest information, and direct recommendations. "
                    "If the user wants recommendations, rankings, comparisons, or a direct answer, reflect that explicitly. "
                    "Avoid broad generic outlines unless the user clearly asked for broad background research. "
                    "Return clean markdown with these sections when relevant: "
                    "'## Intent', '## Scope', '## Desired Output', and '## Research Plan'. "
                    "Under the plan, use short bullets focused only on information needed to answer the query."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content
