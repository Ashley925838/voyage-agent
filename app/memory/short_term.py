"""Short-term memory: a simple conversation buffer.

Day 2 will add long-term memory (FAISS-backed user preferences).
"""
from typing import Any


class ConversationBuffer:
    """Holds the running list of messages for one session.

    Stores OpenAI/Groq-style message dicts:
    {"role": "system" | "user" | "assistant" | "tool", "content": ..., ...}
    """

    def __init__(self, system_prompt: str, max_messages: int = 30) -> None:
        self._system_prompt = system_prompt
        self._max_messages = max_messages
        self._messages: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        # Trim oldest non-system messages if we exceed the cap.
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    def as_list(self) -> list[dict[str, Any]]:
        """Return system prompt + buffered messages, ready for the LLM."""
        return [{"role": "system", "content": self._system_prompt}, *self._messages]