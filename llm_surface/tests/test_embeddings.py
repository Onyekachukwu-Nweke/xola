"""Tests for the embedding helper, token counting, and /embed HTTP endpoint.

Unit tests
----------
- ``embed_text`` returns the vector from the OpenAI response
- ``embed_text`` raises ``ValueError`` on empty / whitespace input
- ``count_tokens`` returns a positive int for valid text
- ``count_tokens`` falls back to cl100k_base for unknown models
- ``count_tokens`` raises ``ValueError`` on empty text
- ``POST /embed`` returns a vector and token_count (IPC contract)
- ``POST /embed`` accepts an explicit model field
- ``POST /embed`` rejects empty text (422)
- ``POST /embed`` rejects a missing ``text`` field (422)
- ``POST /embed`` returns 502 on OpenAI API errors
- ``GET /health`` returns ``{"status": "ok"}``

Integration test (requires OPENAI_API_KEY)
------------------------------------------
- ``POST /embed`` against the real API returns a 1536-dim vector

Run integration tests::

    uv run pytest -m integration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from llm_surface.embeddings import EMBEDDING_MODEL, count_tokens, embed_text
from llm_surface.server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_DIM = 1536
FAKE_VECTOR: list[float] = [0.5] * FAKE_DIM


def _mock_openai(vector: list[float] = FAKE_VECTOR) -> AsyncMock:
    """Return an ``AsyncOpenAI`` mock whose ``embeddings.create`` yields *vector*."""
    client = AsyncMock()
    client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=vector)])
    return client


# ---------------------------------------------------------------------------
# Unit tests — embed_text helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_text_returns_vector() -> None:
    """embed_text must return the embedding contained in the OpenAI response."""
    mock = _mock_openai()
    result = await embed_text("hello world", client=mock, model=EMBEDDING_MODEL)

    assert result == FAKE_VECTOR
    mock.embeddings.create.assert_awaited_once_with(model=EMBEDDING_MODEL, input="hello world")


@pytest.mark.asyncio
async def test_embed_text_uses_default_model() -> None:
    """embed_text must default to EMBEDDING_MODEL when model is not passed."""
    mock = _mock_openai()
    await embed_text("hello", client=mock)

    _, kwargs = mock.embeddings.create.call_args
    assert kwargs["model"] == EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_embed_text_raises_on_empty_string() -> None:
    mock = _mock_openai()
    with pytest.raises(ValueError, match="non-empty"):
        await embed_text("", client=mock)


@pytest.mark.asyncio
async def test_embed_text_raises_on_whitespace_only() -> None:
    mock = _mock_openai()
    with pytest.raises(ValueError, match="non-empty"):
        await embed_text("   \t\n", client=mock)


# ---------------------------------------------------------------------------
# Unit tests — count_tokens helper
# ---------------------------------------------------------------------------


def test_count_tokens_returns_positive_int() -> None:
    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_longer_text_returns_more_tokens() -> None:
    short = count_tokens("hi")
    long = count_tokens("this is a much longer sentence with many more words")
    assert long > short


def test_count_tokens_raises_on_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        count_tokens("")


def test_count_tokens_raises_on_whitespace_only() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        count_tokens("   \t\n")


def test_count_tokens_unknown_model_falls_back() -> None:
    """Unknown models should fall back to cl100k_base, not raise."""
    result = count_tokens("hello world", model="nonexistent-model-xyz")
    assert isinstance(result, int)
    assert result > 0


# ---------------------------------------------------------------------------
# HTTP endpoint tests — TestClient with mocked embed_text
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with ``embed_text`` replaced by a stub that returns FAKE_VECTOR."""

    async def _fake_embed(text: str, *, client: Any, model: str) -> list[float]:
        return FAKE_VECTOR

    monkeypatch.setattr("llm_surface.server.embed_text", _fake_embed)
    with TestClient(app) as c:
        yield c  # type: ignore[misc]


def test_embed_endpoint_happy_path(client: TestClient) -> None:
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["vector"]) == FAKE_DIM
    assert isinstance(body["token_count"], int)
    assert body["token_count"] > 0


def test_embed_endpoint_with_explicit_model(client: TestClient) -> None:
    """The request body should accept a model field per IPC contract."""
    resp = client.post("/embed", json={"text": "hello world", "model": "text-embedding-3-small"})
    assert resp.status_code == 200
    body = resp.json()
    assert "vector" in body
    assert "token_count" in body


def test_embed_endpoint_rejects_empty_text(client: TestClient) -> None:
    """FastAPI/Pydantic must reject empty strings before reaching embed_text."""
    resp = client.post("/embed", json={"text": ""})
    assert resp.status_code == 422


def test_embed_endpoint_rejects_missing_text_field(client: TestClient) -> None:
    resp = client.post("/embed", json={})
    assert resp.status_code == 422


def test_embed_endpoint_502_on_openai_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When embed_text raises APIError the endpoint must return 502."""
    import httpx
    from openai import APIStatusError

    async def _raise(*args: Any, **kwargs: Any) -> list[float]:
        # APIStatusError requires response.request to be set.
        req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        resp = httpx.Response(429, request=req)
        raise APIStatusError("quota exceeded", response=resp, body=None)

    monkeypatch.setattr("llm_surface.server.embed_text", _raise)
    # _mock_api_key autouse fixture has already set OPENAI_API_KEY, so
    # TestClient can complete the lifespan without hitting the real API.
    with TestClient(app) as c:
        resp = c.post("/embed", json={"text": "some text"})
    assert resp.status_code == 502


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_reason_rejects_empty_body(client: TestClient) -> None:
    """POST /reason without a body returns 422 (validation error)."""
    resp = client.post("/reason")
    assert resp.status_code == 422


def test_parse_returns_501(client: TestClient) -> None:
    resp = client.post("/parse")
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# Integration test — requires real OPENAI_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_text_real_api() -> None:
    """Call the real OpenAI API and verify the returned vector shape.

    Skip gracefully when ``OPENAI_API_KEY`` is not present (e.g. in CI without
    the secret configured).
    """
    import os

    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration test")

    openai_client = AsyncOpenAI(api_key=api_key)
    try:
        vector = await embed_text("the quick brown fox", client=openai_client)
        assert len(vector) == FAKE_DIM, f"expected {FAKE_DIM} dims, got {len(vector)}"
        assert all(isinstance(x, float) for x in vector)
    finally:
        await openai_client.close()
