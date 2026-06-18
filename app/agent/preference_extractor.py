"""Use the LLM to extract stable user preferences from a conversation turn.

We deliberately make this its own small LLM call rather than mixing it into
the main loop — it keeps the main loop predictable and lets us swap the
extractor (or batch it) later without touching agent code.
"""
import json
import logging

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


_EXTRACTOR_PROMPT = """You extract stable, reusable USER PREFERENCES from a travel-planning exchange.

Stable preferences are things that will likely be true in future trips too, e.g.:
- "Vegetarian"
- "Prefers slow travel over packed itineraries"
- "Allergic to shellfish"
- "Budget around 200 SGD per day"
- "Travels with a partner"
- "Loves street photography"

NOT stable preferences (do NOT extract these):
- Specific destinations being asked about right now
- One-off questions ("What's the weather Saturday?")
- Acknowledgements ("Sounds good", "Thanks")

Return ONLY a JSON object: {"preferences": ["short fact 1", "short fact 2", ...]}
If nothing stable was said, return {"preferences": []}.

Be conservative. It is better to miss a preference than to invent one."""


def extract_preferences(
    provider: LLMProvider,
    user_message: str,
    assistant_reply: str,
) -> list[str]:
    payload = (
        f"USER: {user_message}\n\n"
        f"ASSISTANT: {assistant_reply}\n\n"
        "Extract any stable user preferences from the USER message above."
    )
    try:
        response = provider.chat(
            messages=[
                {"role": "system", "content": _EXTRACTOR_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.0,
        )
        raw = (response.get("content") or "").strip()
        # Tolerate models that wrap JSON in ```json fences.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        prefs = data.get("preferences", [])
        return [p for p in prefs if isinstance(p, str) and len(p) <= 120]
    except Exception:  # noqa: BLE001
        logger.exception("Preference extraction failed; skipping this turn.")
        return []