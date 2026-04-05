import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def research(plan_text: str):
    short_query = plan_text[:300]

    result = tavily.search(
        query=short_query,
        max_results=4
    )

    return result