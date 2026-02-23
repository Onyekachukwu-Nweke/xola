"""Unit tests for the ReAct reasoning step and /reason endpoint."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from llm_surface.react import reason_step
from llm_surface.server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Find RAG papers."},
]

SAMPLE_TOOL_SCHEMAS: list[dict[str, Any]] = [
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


def _make_tool_call(
    name: str = "web_search",
    arguments: str = '{"query": "RAG papers 2024"}',
) -> MagicMock:
    """Build a mock tool_call object matching OpenAI's response shape."""
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc = MagicMock()
    tc.function = func
    return tc


def _mock_openai_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> AsyncMock:
    """Return an ``AsyncOpenAI`` mock whose ``chat.completions.create`` yields
    a response with the given content and/or tool_calls."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    client = AsyncMock()
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# reason_step — tool call path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_step_returns_tool_call() -> None:
    tc = _make_tool_call("web_search", '{"query": "RAG papers 2024"}')
    client = _mock_openai_response(content="I should search for papers.", tool_calls=[tc])

    result = await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="Find RAG papers",
        client=client,
    )

    assert result["action"] == "web_search"
    assert result["action_input"] == {"query": "RAG papers 2024"}
    assert result["is_final"] is False
    assert result["final_answer"] is None
    assert result["thought"] == "I should search for papers."


@pytest.mark.asyncio
async def test_reason_step_returns_final_answer() -> None:
    tc = _make_tool_call("final_answer", '{"answer": "The top 3 RAG papers are..."}')
    client = _mock_openai_response(content="Done.", tool_calls=[tc])

    result = await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="Find RAG papers",
        client=client,
    )

    assert result["is_final"] is True
    assert result["final_answer"] == "The top 3 RAG papers are..."
    assert result["action"] is None
    assert result["action_input"] == {}


# ---------------------------------------------------------------------------
# reason_step — plain text fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_step_plain_text_fallback() -> None:
    """No tool call, but model returned text → treated as final answer."""
    client = _mock_openai_response(content="I already know the answer: 42.", tool_calls=None)

    result = await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="What is the answer?",
        client=client,
    )

    assert result["is_final"] is True
    assert result["final_answer"] == "I already know the answer: 42."
    assert result["action"] is None


# ---------------------------------------------------------------------------
# reason_step — error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_step_empty_response_raises() -> None:
    """Empty content and no tool call → ValueError."""
    client = _mock_openai_response(content=None, tool_calls=None)

    with pytest.raises(ValueError, match="empty response"):
        await reason_step(
            SAMPLE_MESSAGES,
            SAMPLE_TOOL_SCHEMAS,
            memory_context=[],
            task_goal="goal",
            client=client,
        )


@pytest.mark.asyncio
async def test_reason_step_malformed_arguments_raises() -> None:
    """Tool call with invalid JSON in arguments → ValueError."""
    tc = _make_tool_call("web_search", "not valid json {{{")
    client = _mock_openai_response(content=None, tool_calls=[tc])

    with pytest.raises(ValueError, match="Malformed tool call arguments"):
        await reason_step(
            SAMPLE_MESSAGES,
            SAMPLE_TOOL_SCHEMAS,
            memory_context=[],
            task_goal="goal",
            client=client,
        )


# ---------------------------------------------------------------------------
# reason_step — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_step_tool_call_with_no_content() -> None:
    """OpenAI sometimes returns content=None with a tool call — thought defaults to ''."""
    tc = _make_tool_call("web_search", '{"query": "test"}')
    client = _mock_openai_response(content=None, tool_calls=[tc])

    result = await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="goal",
        client=client,
    )

    assert result["thought"] == ""
    assert result["action"] == "web_search"


@pytest.mark.asyncio
async def test_reason_step_empty_tool_arguments() -> None:
    """Tool call with empty arguments string → empty dict."""
    tc = _make_tool_call("web_search", "")
    client = _mock_openai_response(content="thinking", tool_calls=[tc])

    result = await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="goal",
        client=client,
    )

    assert result["action_input"] == {}


@pytest.mark.asyncio
async def test_reason_step_includes_memory_context() -> None:
    """Memory context should be forwarded to the system prompt."""
    memories = ["RAG was published in 2020"]
    tc = _make_tool_call("web_search", '{"query": "test"}')
    client = _mock_openai_response(content="thinking", tool_calls=[tc])

    await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=memories,
        task_goal="Find papers",
        client=client,
    )

    # Verify the system prompt (first message) contains the memory.
    call_args = client.chat.completions.create.call_args
    sent_messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
    system_msg = sent_messages[0]["content"]
    assert "RAG was published in 2020" in system_msg


@pytest.mark.asyncio
async def test_reason_step_passes_tool_schemas_to_openai() -> None:
    """Tool schemas should be converted to OpenAI format and passed."""
    tc = _make_tool_call("web_search", '{"query": "test"}')
    client = _mock_openai_response(content="thinking", tool_calls=[tc])

    await reason_step(
        SAMPLE_MESSAGES,
        SAMPLE_TOOL_SCHEMAS,
        memory_context=[],
        task_goal="goal",
        client=client,
    )

    call_args = client.chat.completions.create.call_args
    tools = call_args.kwargs.get("tools", [])
    names = [t["function"]["name"] for t in tools]
    assert "web_search" in names
    assert "final_answer" in names


# ---------------------------------------------------------------------------
# /reason HTTP endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def reason_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with reason_step monkeypatched."""

    async def _fake_reason_step(
        messages: Any,
        tool_schemas: Any,
        memory_context: Any,
        task_goal: Any,
        *,
        client: Any,
        model: Any = "gpt-4o",
    ) -> dict[str, Any]:
        return {
            "thought": "I should search.",
            "action": "web_search",
            "action_input": {"query": "test"},
            "is_final": False,
            "final_answer": None,
        }

    monkeypatch.setattr("llm_surface.server.reason_step", _fake_reason_step)
    with TestClient(app) as c:
        yield c


def test_reason_endpoint_happy_path(reason_client: TestClient) -> None:
    resp = reason_client.post(
        "/reason",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "tool_schemas": [],
            "memory_context": [],
            "task_goal": "Find papers",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["thought"] == "I should search."
    assert body["action"] == "web_search"
    assert body["is_final"] is False


def test_reason_endpoint_rejects_empty_goal(reason_client: TestClient) -> None:
    resp = reason_client.post(
        "/reason",
        json={
            "messages": [],
            "tool_schemas": [],
            "memory_context": [],
            "task_goal": "",
        },
    )
    assert resp.status_code == 422


def test_reason_endpoint_502_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APIStatusError

    async def _failing_reason(*args: Any, **kwargs: Any) -> None:
        raise APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}, text="rate limited"),
            body=None,
        )

    monkeypatch.setattr("llm_surface.server.reason_step", _failing_reason)
    with TestClient(app) as c:
        resp = c.post(
            "/reason",
            json={
                "messages": [],
                "tool_schemas": [],
                "memory_context": [],
                "task_goal": "goal",
            },
        )
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Integration test (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reason_step_real_api() -> None:
    """Smoke test against the real OpenAI API."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)
    try:
        result = await reason_step(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            tool_schemas=[],
            memory_context=[],
            task_goal="Answer the arithmetic question",
            client=client,
            model="gpt-4o-mini",
        )
        # The model should return a final answer (no tools available except final_answer).
        assert result["is_final"] is True
        assert result["final_answer"]
    finally:
        await client.close()
