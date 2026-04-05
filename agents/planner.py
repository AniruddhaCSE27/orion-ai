import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan(query: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional research planner. "
                    "Create a concise, structured research plan with headings and bullet points. "
                    "Keep it useful and not excessively long."
                )
            },
            {
                "role": "user",
                "content": f"Create a research plan for: {query}"
            }
        ]
    )
    return response.choices[0].message.content