import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def write(plan_text: str, research_data):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

RESEARCH DATA:
{research_data}
"""
            }
        ]
    )

    return response.choices[0].message.content