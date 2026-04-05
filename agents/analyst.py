import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze(data) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a research analyst. Analyze the given research data and produce clear, structured insights."
            },
            {
                "role": "user",
                "content": str(data)
            }
        ]
    )

    return response.choices[0].message.content