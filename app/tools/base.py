"""Tool abstract base class. Each tool exposes:
- a name (LLM sees this)
- a JSON schema describing its parameters
- a run() that takes the parsed kwargs and returns a string for the LLM.
"""
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]  # OpenAI/Groq tool schema (JSON schema)

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool. Returns a string the LLM will read as the tool result."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to the format Groq/OpenAI expects in the tools= parameter."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }