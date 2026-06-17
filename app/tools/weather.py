"""Weather tool. Day 1 returns mock data; Day 2 will call OpenWeatherMap."""
from typing import Any

from app.tools.base import Tool


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get the weather forecast for a city on a given date. "
        "Use this when the user asks about weather, packing, or outdoor planning."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. Use today if user did not specify.",
            },
        },
        "required": ["city", "date"],
    }

    def run(self, **kwargs: Any) -> str:
        city = kwargs.get("city", "unknown")
        date = kwargs.get("date", "unknown")
        # MOCK — Day 2 swap with real OpenWeatherMap call.
        return (
            f"[MOCK] Weather for {city} on {date}: "
            f"partly cloudy, 22°C high / 15°C low, 20% chance of rain."
        )