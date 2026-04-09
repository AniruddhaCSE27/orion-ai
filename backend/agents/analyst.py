from openai import OpenAI

from backend.core.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def analyze(data) -> str:
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
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
