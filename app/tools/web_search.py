"""Web search tool — real Tavily API.

Tavily is purpose-built for LLM agents: it returns clean text snippets
(no HTML stripping needed) and lets you control depth and snippet count.
"""
import logging
from typing import Any

from tavily import TavilyClient

from app.config import get_settings
from app.tools.base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the live web for current information: events, opening hours, "
        "prices, news, restaurant reviews, recent changes. "
        "Use this when the user asks about something time-sensitive or specific "
        "that the model may not know from training."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, 3-10 words. Be specific.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return. Default 5, max 10.",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._client = TavilyClient(api_key=get_settings().tavily_api_key)

    def run(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        max_results = min(int(kwargs.get("max_results", 5)), 10)

        if not query:
            return "ERROR: query is required."

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",  # "advanced" costs 2 credits; basic is enough
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Tavily search failed for query=%s", query)
            return f"ERROR: web search failed: {e}"

        results = response.get("results", [])
        if not results:
            return f"No web results found for '{query}'."

        # Format compactly so the LLM can read efficiently.
        lines = [f"Web results for '{query}':"]
        for i, r in enumerate(results, start=1):
            title = r.get("title", "(no title)")
            content = r.get("content", "")[:300]  # cap each snippet
            url = r.get("url", "")
            lines.append(f"{i}. {title}\n   {content}\n   Source: {url}")

        return "\n".join(lines)