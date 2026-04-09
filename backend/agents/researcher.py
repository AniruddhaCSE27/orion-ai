from tavily import TavilyClient

from backend.core.config import config

tavily = TavilyClient(api_key=config.TAVILY_API_KEY)

def research(plan_text: str):
    short_query = plan_text[:300]

    result = tavily.search(
        query=short_query,
        max_results=4
    )

    return result
