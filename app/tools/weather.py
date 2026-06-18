"""Weather tool — real OpenWeatherMap 5-day / 3-hour forecast."""
import logging
from datetime import date
from typing import Any

import httpx

from app.config import get_settings
from app.tools._dates import parse_loose_date
from app.tools.base import Tool

logger = logging.getLogger(__name__)

# OpenWeatherMap free tier: 5-day forecast in 3-hour steps + current weather.
_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


class WeatherTool(Tool):
    name = "get_weather"
    description = (
        "Get weather for a city on a given date. "
        "Returns temperature range, conditions, and rain probability. "
        "Use this whenever the user asks about weather, packing, or outdoor planning."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'."},
            "date": {
                "type": "string",
                "description": (
                    "Target date. Prefer YYYY-MM-DD, but 'today', 'tomorrow', "
                    "'next weekend' are also accepted."
                ),
            },
        },
        "required": ["city"],
    }

    def __init__(self) -> None:
        self._api_key = get_settings().openweather_api_key

    def run(self, **kwargs: Any) -> str:
        city = kwargs.get("city", "").strip()
        target_date = parse_loose_date(kwargs.get("date"))

        if not city:
            return "ERROR: city is required."

        try:
            lat, lon, resolved_name = self._geocode(city)
        except Exception as e:  # noqa: BLE001
            logger.exception("Geocoding failed for %s", city)
            return f"ERROR: could not look up city '{city}': {e}"

        try:
            forecast = self._forecast(lat, lon)
        except Exception as e:  # noqa: BLE001
            logger.exception("Forecast fetch failed for %s", city)
            return f"ERROR: could not fetch forecast for '{city}': {e}"

        return self._summarise(forecast, target_date, resolved_name)

    # ---------- internals ----------

    def _geocode(self, city: str) -> tuple[float, float, str]:
        r = httpx.get(
            _GEOCODE_URL,
            params={"q": city, "limit": 1, "appid": self._api_key},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError(f"no results for '{city}'")
        top = data[0]
        return top["lat"], top["lon"], f"{top['name']}, {top.get('country', '')}"

    def _forecast(self, lat: float, lon: float) -> dict[str, Any]:
        r = httpx.get(
            _FORECAST_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self._api_key,
                "units": "metric",
            },
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _summarise(forecast: dict[str, Any], target: date, resolved_name: str) -> str:
        # forecast["list"] is a list of 3-hour buckets across 5 days.
        same_day = [
            entry
            for entry in forecast.get("list", [])
            if entry.get("dt_txt", "").startswith(target.isoformat())
        ]

        if not same_day:
            available_dates = sorted(
                {e["dt_txt"][:10] for e in forecast.get("list", [])}
            )
            return (
                f"No forecast available for {resolved_name} on {target.isoformat()}. "
                f"Forecast covers: {', '.join(available_dates)}."
            )

        temps = [e["main"]["temp"] for e in same_day]
        rain_probs = [e.get("pop", 0) for e in same_day]  # 0..1
        conditions = [e["weather"][0]["description"] for e in same_day]
        # Pick the most frequent condition.
        dominant = max(set(conditions), key=conditions.count)

        return (
            f"Weather in {resolved_name} on {target.isoformat()}: "
            f"{dominant}, "
            f"{round(min(temps))}°C to {round(max(temps))}°C, "
            f"max rain probability {round(max(rain_probs) * 100)}%."
        )