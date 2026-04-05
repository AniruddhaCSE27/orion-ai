import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def research(plan: str):
    short_plan = plan[:300]

    result = tavily.search(
        query=short_plan,
        max_results=4
    )
    return result