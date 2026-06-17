"""Agent loop: hand-written tool-calling reasoning loop.

Flow per user turn:
  1. Append the user message to the buffer.
  2. Call the LLM with the full buffer + available tool schemas.
  3. If the LLM returns plain text => return it to the user (turn done).
  4. If the LLM returns tool_calls => execute each, append tool results,
     and loop back to step 2.
  5. Stop after MAX_ITERATIONS to prevent runaway loops.

This is intentionally hand-written (no LangChain) for full control and
easier debugging — we can log every step and reason about failures.
"""
import json
import logging
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
from app.llm.base import LLMProvider
from app.memory.short_term import ConversationBuffer
from app.tools.base import Tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6  # Hard cap on tool-calling rounds per user turn.


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, Tool],
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._tool_schemas = [t.to_openai_schema() for t in tools.values()]
        self._buffer = ConversationBuffer(system_prompt=SYSTEM_PROMPT)

    def chat(self, user_message: str) -> str:
        """Run one user turn end-to-end. Returns the final assistant text."""
        self._buffer.add({"role": "user", "content": user_message})

        for iteration in range(MAX_ITERATIONS):
            logger.info("Agent iteration %d", iteration)

            assistant_msg = self._provider.chat(
                messages=self._buffer.as_list(),
                tools=self._tool_schemas,
            )

            # Persist the assistant message (with tool_calls, if any) to the buffer.
            self._buffer.add(self._assistant_msg_for_buffer(assistant_msg))

            tool_calls = assistant_msg.get("tool_calls")

            # No tool calls => the LLM is done reasoning, return its text.
            if not tool_calls:
                return assistant_msg.get("content") or ""

            # Otherwise, execute each requested tool and feed results back.
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                tool_result = self._execute_tool(fn_name, raw_args)

                self._buffer.add(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )

        # Safety net: model kept looping. Force a final answer.
        logger.warning("Agent hit MAX_ITERATIONS without producing a final answer.")
        return (
            "I gathered some information but ran out of reasoning steps. "
            "Could you narrow down what you'd like me to focus on?"
        )

    # ---------- helpers ----------

    @staticmethod
    def _assistant_msg_for_buffer(assistant_msg: dict[str, Any]) -> dict[str, Any]:
        """Shape the assistant message the way the chat API expects when replayed."""
        msg: dict[str, Any] = {"role": "assistant", "content": assistant_msg.get("content")}
        if assistant_msg.get("tool_calls"):
            msg["tool_calls"] = assistant_msg["tool_calls"]
        return msg

    def _execute_tool(self, name: str, raw_args: str) -> str:
        """Look up the tool by name, parse JSON args, run it, return a string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: tool '{name}' is not registered."

        try:
            kwargs = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return f"ERROR: could not parse arguments for tool '{name}': {e}"

        try:
            result = tool.run(**kwargs)
            logger.info("Tool %s ran with args=%s", name, kwargs)
            return result
        except Exception as e:  # noqa: BLE001 — surface to LLM
            logger.exception("Tool %s failed", name)
            return f"ERROR: tool '{name}' raised an exception: {e}"