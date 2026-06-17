"""Groq implementation of LLMProvider, using the OpenAI-compatible API."""
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self._model = settings.groq_model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        # Normalise to plain dict so the agent loop doesn't depend on SDK types.
        return {
            "role": "assistant",
            "content": choice.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ]
                if choice.tool_calls
                else None
            ),
        }