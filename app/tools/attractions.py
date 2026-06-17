"""Attraction search tool. Day 1 returns mock data;
Day 2 will retrieve from a FAISS index built over Wikipedia city pages.
"""
from typing import Any

from app.tools.base import Tool


class AttractionsTool(Tool):
    name = "search_attractions"
    description = (
        "Retrieve well-known attractions, neighbourhoods, or activities for a city. "
        "Use this for sightseeing planning. Returns 5 attractions with short descriptions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "interest": {
                "type": "string",
                "description": (
                    "Optional interest filter, e.g. 'food', 'history', 'nightlife', "
                    "'family-friendly'. Omit if user has no preference."
                ),
            },
        },
        "required": ["city"],
    }

    def run(self, **kwargs: Any) -> str:
        city = kwargs.get("city", "unknown")
        interest = kwargs.get("interest")
        suffix = f" matching interest '{interest}'" if interest else ""
        # MOCK — Day 2 swap with FAISS-backed Wikipedia retrieval.
        return (
            f"[MOCK] Top attractions in {city}{suffix}:\n"
            f"1. Famous Landmark A — iconic spot, allow 2 hours.\n"
            f"2. Neighbourhood B — walkable area, good for evenings.\n"
            f"3. Museum C — best visited on weekday mornings.\n"
            f"4. Local Eatery D — known for a regional dish.\n"
            f"5. Park E — relaxing, central location."
        )