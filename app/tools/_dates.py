"""Date parsing helpers — defensively normalise loose date strings the LLM passes in."""
from __future__ import annotations

from datetime import date, datetime, timedelta


def parse_loose_date(value: str | None) -> date:
    """Best-effort parse of whatever the LLM hands us as a date.

    Handles:
    - "2026-06-20"  → exact
    - "today" / "tomorrow"
    - "next weekend" → upcoming Saturday
    - empty / None  → today

    Anything we cannot parse falls back to today and we let the caller log it.
    """
    if not value:
        return date.today()

    v = value.strip().lower()
    today = date.today()

    if v in {"today", "now"}:
        return today
    if v == "tomorrow":
        return today + timedelta(days=1)
    if "weekend" in v:
        # Upcoming Saturday. weekday(): Monday=0 ... Sunday=6, Saturday=5.
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0:  # today is Saturday
            days_until_saturday = 7
        return today + timedelta(days=days_until_saturday)

    # Try strict ISO date.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue

    # Give up; let caller decide. We return today so the API call still succeeds.
    return today