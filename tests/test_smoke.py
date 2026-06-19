"""Smoke tests that don't require external APIs.

These verify the wiring (imports, tool registration, schema generation) is
intact. End-to-end tests are run manually via the FastAPI endpoint.
"""
from app.agent.loop import AgentLoop
from app.tools.base import Tool
from app.tools.registry import build_registry


def test_registry_returns_three_tools():
    registry = build_registry()
    assert set(registry.keys()) == {"get_weather", "web_search", "search_attractions"}


def test_every_tool_exposes_valid_openai_schema():
    registry = build_registry()
    for name, tool in registry.items():
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert "parameters" in schema["function"]
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_every_tool_inherits_from_tool_abc():
    registry = build_registry()
    for tool in registry.values():
        assert isinstance(tool, Tool)


def test_agent_loop_initialises_with_anon_user():
    # Doesn't make any API calls — just verifies wiring.
    registry = build_registry()

    class _DummyProvider:
        def chat(self, messages, tools=None, temperature=0.3):
            return {"role": "assistant", "content": "ok", "tool_calls": None}

    agent = AgentLoop(provider=_DummyProvider(), tools=registry, user_id="test_anon")
    assert agent._user_id == "test_anon"
    assert len(agent._tool_schemas) == 3
