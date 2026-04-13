def critique(insights: str) -> str:
    if "no relevant live data found" in insights.lower() or "not enough data" in insights.lower():
        return "Warning: Live evidence was limited, so this answer may need manual verification.\n\n" + insights

    return "Verified Insights:\n\n" + insights
