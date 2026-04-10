def critique(insights: str) -> str:
    if "i couldn't find enough reliable live data for this query right now." in insights.lower() or "not enough data" in insights.lower():
        return "Warning: Some insights may need manual verification.\n\n" + insights

    return "Verified Insights:\n\n" + insights
