"""Agent loop with short-term (in-session) + long-term (cross-session) memory."""
import json
import logging
from typing import Any

from app.agent.preference_extractor import extract_preferences
from app.agent.prompts import SYSTEM_PROMPT
from app.llm.base import LLMProvider
from app.memory.long_term import UserPreferenceStore
from app.memory.short_term import ConversationBuffer
from app.tools.base import Tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, Tool],
        user_id: str = "anon",
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._tool_schemas = [t.to_openai_schema() for t in tools.values()]
        self._user_id = user_id
        self._prefs = UserPreferenceStore(user_id=user_id)
        # Buffer gets primed per-turn so we can inject relevant preferences each time.
        self._buffer = ConversationBuffer(system_prompt=SYSTEM_PROMPT)

    def chat(self, user_message: str) -> str:
        # 1. Retrieve relevant long-term preferences and inject as a transient note.
        relevant_prefs = self._prefs.retrieve(user_message, top_k=3)
        if relevant_prefs:
            note = "Known user preferences (from prior conversations): " + "; ".join(relevant_prefs)
            self._buffer.add({"role": "system", "content": note})
            logger.info("Injected %d preferences into context", len(relevant_prefs))

        self._buffer.add({"role": "user", "content": user_message})

        final_reply = ""
        for iteration in range(MAX_ITERATIONS):
            logger.info("Agent iteration %d", iteration)
            assistant_msg = self._provider.chat(
                messages=self._buffer.as_list(),
                tools=self._tool_schemas,
            )
            self._buffer.add(self._assistant_msg_for_buffer(assistant_msg))

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                final_reply = assistant_msg.get("content") or ""
                break

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
        else:
            logger.warning("Agent hit MAX_ITERATIONS without a final answer.")
            final_reply = (
                "I gathered some information but ran out of reasoning steps. "
                "Could you narrow down what you'd like me to focus on?"
            )

        # 2. Extract stable preferences from this exchange for future sessions.
        try:
            new_prefs = extract_preferences(
                provider=self._provider,
                user_message=user_message,
                assistant_reply=final_reply,
            )
            if new_prefs:
                self._prefs.add(new_prefs)
        except Exception:  # noqa: BLE001
            logger.exception("Preference write-back failed; continuing.")

        return final_reply

    # ---------- helpers ----------

    @staticmethod
    def _assistant_msg_for_buffer(assistant_msg: dict[str, Any]) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": assistant_msg.get("content")}
        if assistant_msg.get("tool_calls"):
            msg["tool_calls"] = assistant_msg["tool_calls"]
        return msg

    def _execute_tool(self, name: str, raw_args: str) -> str:
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
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return f"ERROR: tool '{name}' raised an exception: {e}"