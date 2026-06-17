"""Tool registry: a single place that lists all available tools.
The agent loop reads from here. Adding a new tool = 1 line.
"""
from app.tools.attractions import AttractionsTool
from app.tools.base import Tool
from app.tools.weather import WeatherTool
from app.tools.web_search import WebSearchTool


def build_registry() -> dict[str, Tool]:
    tools: list[Tool] = [
        WeatherTool(),
        WebSearchTool(),
        AttractionsTool(),
    ]
    return {tool.name: tool for tool in tools}