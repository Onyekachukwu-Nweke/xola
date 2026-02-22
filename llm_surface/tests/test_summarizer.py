"""Tests for the summarizer module and POST /summarize endpoint.

Unit tests
----------
- ``summarize_messages`` returns a non-empty string and a positive token count
- ``summarize_messages`` raises ``ValueError`` on empty messages list
- ``summarize_messages`` raises ``RuntimeError`` when LLM returns empty content
- ``summarize_messages`` passes max_tokens to the API
- ``summarize_messages`` uses temperature=0
- ``POST /summarize`` returns 200 with summary + token_count
- ``POST /summarize`` rejects empty messages list (422)
- ``POST /summarize`` accepts an explicit model field
- ``POST /summarize`` returns 502 on OpenAI API errors
- ``POST /summarize`` returns 502 when LLM returns empty content

Integration test (requires OPENAI_API_KEY)
------------------------------------------
- ``summarize_messages`` against the real API returns a non-empty summary
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from llm_surface.server import app
from llm_surface.summarizer import SUMMARIZE_MODEL, summarize_messages

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "What is the capital of France?", "token_count": "8"},
    {"role": "assistant", "content": "The capital of France is Paris.", "token_count": "7"},
    {"role": "user", "content": "What is its population?", "token_count": "6"},
    {
        "role": "assistant",
        "content": "Paris has a population of about 2.1 million in the city proper.",
        "token_count": "16",
    },
]

# A plausible summary — kept short to stay under the 100-char line limit.
FAKE_SUMMARY = "Paris is the capital of France with ~2.1 million city-proper residents."


# ---------------------------------------------------------------------------
# Unit tests — summarize_messages helper
# ---------------------------------------------------------------------------


def _mock_openai_chat(summary_text: str = FAKE_SUMMARY) -> AsyncMock:
    """Return an ``AsyncOpenAI`` mock whose chat.completions.create returns *summary_text*."""
    client = AsyncMock()
    choice = MagicMock()
    choice.message.content = summary_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


@pytest.mark.asyncio
async def test_summarize_returns_text_and_count() -> None:
    """summarize_messages must return a non-empty summary and a positive token count."""
    mock = _mock_openai_chat()
    text, count = await summarize_messages(SAMPLE_MESSAGES, client=mock, model=SUMMARIZE_MODEL)

    assert text == FAKE_SUMMARY
    assert isinstance(count, int)
    assert count > 0


@pytest.mark.asyncio
async def test_summarize_passes_max_tokens_to_api() -> None:
    """max_summary_tokens must be forwarded as max_tokens in the API call."""
    mock = _mock_openai_chat()
    await summarize_messages(SAMPLE_MESSAGES, client=mock, max_summary_tokens=128)

    _, kwargs = mock.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 128


@pytest.mark.asyncio
async def test_summarize_raises_on_empty_messages() -> None:
    mock = _mock_openai_chat()
    with pytest.raises(ValueError, match="non-empty"):
        await summarize_messages([], client=mock)


@pytest.mark.asyncio
async def test_summarize_uses_temperature_zero() -> None:
    """Summaries must be deterministic (temperature=0)."""
    mock = _mock_openai_chat()
    await summarize_messages(SAMPLE_MESSAGES, client=mock)

    _, kwargs = mock.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_summarize_raises_on_empty_llm_response() -> None:
    """An empty LLM response must raise RuntimeError, not return ('', 1).

    This guards against silent STM corruption: returning ('', 1) would replace
    the entire buffer with an empty message carrying token_count=1.
    """
    mock = _mock_openai_chat(summary_text="")
    with pytest.raises(RuntimeError, match="empty response"):
        await summarize_messages(SAMPLE_MESSAGES, client=mock)


# ---------------------------------------------------------------------------
# HTTP endpoint tests — TestClient with mocked summarize_messages
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with ``summarize_messages`` replaced by a stub."""

    async def _fake_summarize(messages: list[dict[str, str]], **_kwargs: Any) -> tuple[str, int]:
        return FAKE_SUMMARY, 18

    monkeypatch.setattr("llm_surface.server.summarize_messages", _fake_summarize)
    with TestClient(app) as c:
        yield c  # type: ignore[misc]


def test_summarize_endpoint_happy_path(client: TestClient) -> None:
    resp = client.post("/summarize", json={"messages": SAMPLE_MESSAGES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == FAKE_SUMMARY
    assert body["token_count"] == 18


def test_summarize_endpoint_with_explicit_model(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={"messages": SAMPLE_MESSAGES, "model": "gpt-4o", "max_summary_tokens": 256},
    )
    assert resp.status_code == 200
    assert "summary" in resp.json()


def test_summarize_endpoint_rejects_empty_messages(client: TestClient) -> None:
    """Pydantic min_length=1 on messages must result in 422."""
    resp = client.post("/summarize", json={"messages": []})
    assert resp.status_code == 422


def test_summarize_endpoint_rejects_missing_messages(client: TestClient) -> None:
    resp = client.post("/summarize", json={})
    assert resp.status_code == 422


def test_summarize_endpoint_502_on_openai_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When summarize_messages raises APIStatusError the endpoint must return 502."""
    import httpx
    from openai import APIStatusError

    async def _raise(messages: list[dict[str, str]], **_kwargs: Any) -> tuple[str, int]:
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(429, request=req)
        raise APIStatusError("rate limited", response=resp, body=None)

    monkeypatch.setattr("llm_surface.server.summarize_messages", _raise)
    with TestClient(app) as c:
        resp = c.post("/summarize", json={"messages": SAMPLE_MESSAGES})
    assert resp.status_code == 502


def test_summarize_endpoint_502_on_empty_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """When summarize_messages raises RuntimeError (empty LLM response) the endpoint returns 502.

    This is a distinct code path from the APIError path above — it exercises the
    ``(APIError, RuntimeError)`` combined except clause added in L2-08 review.
    """

    async def _raise(messages: list[dict[str, str]], **_kwargs: Any) -> tuple[str, int]:
        raise RuntimeError("Summarization returned an empty response — safety filter")

    monkeypatch.setattr("llm_surface.server.summarize_messages", _raise)
    with TestClient(app) as c:
        resp = c.post("/summarize", json={"messages": SAMPLE_MESSAGES})
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Integration test — requires real OPENAI_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_summarize_real_api() -> None:
    """Call the real OpenAI API and verify the returned summary is non-trivial."""
    import os

    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration test")

    openai_client = AsyncOpenAI(api_key=api_key)
    try:
        text, count = await summarize_messages(
            SAMPLE_MESSAGES,
            client=openai_client,
            model=SUMMARIZE_MODEL,
        )
        assert text, "summary must be non-empty"
        assert count > 0, "token count must be positive"
        # Sanity: a reasonable summary of 4 messages should be > 5 tokens
        assert count > 5, f"summary seems too short ({count} tokens): {text!r}"
    finally:
        await openai_client.close()
