"""Unit tests for the system prompt builder and tool schema converter."""

from llm_surface.prompts import (
    FINAL_ANSWER_TOOL,
    build_system_prompt,
    build_tool_schemas_for_openai,
)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


def test_build_system_prompt_contains_goal() -> None:
    prompt = build_system_prompt("Find RAG papers", memory_context=[])
    assert "Find RAG papers" in prompt


def test_build_system_prompt_contains_memory() -> None:
    memories = ["RAG was introduced in 2020", "Dense retrieval improved recall"]
    prompt = build_system_prompt("Summarise RAG history", memory_context=memories)
    for mem in memories:
        assert mem in prompt


def test_build_system_prompt_no_memory_shows_placeholder() -> None:
    prompt = build_system_prompt("Do something", memory_context=[])
    assert "No relevant memories" in prompt


def test_build_system_prompt_memory_as_bullet_list() -> None:
    memories = ["fact one", "fact two"]
    prompt = build_system_prompt("goal", memory_context=memories)
    assert "- fact one" in prompt
    assert "- fact two" in prompt


# ---------------------------------------------------------------------------
# build_tool_schemas_for_openai
# ---------------------------------------------------------------------------


def test_build_tool_schemas_for_openai_format() -> None:
    xola_schemas = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]
    result = build_tool_schemas_for_openai(xola_schemas)

    # The first entry should be the converted tool.
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "web_search"
    assert tool["function"]["description"] == "Search the web"
    assert tool["function"]["parameters"]["required"] == ["query"]


def test_build_tool_schemas_includes_final_answer() -> None:
    result = build_tool_schemas_for_openai([])

    # Even with no real tools, final_answer must be present.
    assert len(result) == 1
    assert result[-1] == FINAL_ANSWER_TOOL


def test_build_tool_schemas_preserves_order() -> None:
    schemas = [
        {"name": "a", "description": "", "parameters": {}},
        {"name": "b", "description": "", "parameters": {}},
    ]
    result = build_tool_schemas_for_openai(schemas)

    names = [t["function"]["name"] for t in result]
    assert names == ["a", "b", "final_answer"]


def test_build_tool_schemas_handles_missing_optional_fields() -> None:
    """Schemas missing 'description' or 'parameters' should not crash."""
    schemas = [{"name": "minimal"}]
    result = build_tool_schemas_for_openai(schemas)
    tool = result[0]
    assert tool["function"]["name"] == "minimal"
    assert tool["function"]["description"] == ""
    assert tool["function"]["parameters"] == {}
