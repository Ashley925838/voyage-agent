"""Abstract LLM provider. Strategy pattern so we can swap Groq <-> OpenAI <-> Gemini."""
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """An LLM provider that supports OpenAI-style chat + tool calling."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a chat request. Returns the raw assistant message dict.

        The returned dict has shape:
        {
            "role": "assistant",
            "content": str | None,
            "tool_calls": list[...] | None,
        }
        """
        ...