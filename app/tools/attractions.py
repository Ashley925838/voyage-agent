"""Attraction search tool — backed by per-city FAISS RAG over Wikipedia."""
import logging
from typing import Any

from app.rag.store import CityVectorStore
from app.tools.base import Tool

logger = logging.getLogger(__name__)


class AttractionsTool(Tool):
    name = "search_attractions"
    description = (
        "Retrieve passages about attractions, neighbourhoods, food, and culture "
        "for a city, grounded in Wikipedia. "
        "Use this for sightseeing planning. "
        "You can call this multiple times with different interest queries "
        "(e.g. 'ramen shops', 'street photography spots', 'history museums') "
        "to gather material for a personalised plan."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "interest": {
                "type": "string",
                "description": (
                    "What aspect of the city to retrieve. Be specific: "
                    "'ramen and izakaya', 'photography spots', "
                    "'family-friendly museums', etc."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to return (default 5, max 8).",
            },
        },
        "required": ["city", "interest"],
    }

    def __init__(self) -> None:
        self._store = CityVectorStore()

    def run(self, **kwargs: Any) -> str:
        city = kwargs.get("city", "").strip()
        interest = kwargs.get("interest", "").strip()
        top_k = min(int(kwargs.get("top_k", 5)), 8)

        if not city or not interest:
            return "ERROR: both 'city' and 'interest' are required."

        try:
            results = self._store.retrieve(city, interest, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            logger.exception("Retrieval failed for city=%s interest=%s", city, interest)
            return f"ERROR: attraction retrieval failed: {e}"

        if not results:
            return (
                f"No Wikipedia content available for {city}. "
                "Try web_search for current information instead."
            )

        lines = [f"Retrieved passages for '{interest}' in {city}:"]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"{i}. [{r.source_title}] (relevance {r.score:.2f})\n"
                f"   {r.text}\n"
                f"   Source: {r.source_url}"
            )
        return "\n".join(lines)