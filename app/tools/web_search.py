"""Web search tool. Day 1 returns mock data; Day 2 will call Tavily."""
from typing import Any

from app.tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for current information about events, opening hours, prices, "
        "or anything not in the agent's internal knowledge. Returns 3-5 snippets."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query, 3-8 words."},
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        # MOCK — Day 2 swap with real Tavily call.
        return (
            f"[MOCK] Top results for '{query}':\n"
            f"1. Result A — short snippet about {query}.\n"
            f"2. Result B — another snippet mentioning {query} highlights.\n"
            f"3. Result C — local tip relevant to {query}."
        )