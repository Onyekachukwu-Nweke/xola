"""FastAPI IPC server — exposes HTTP endpoints toward the Rust runtime.

Endpoints
---------
POST /embed   — embed text, return a float vector          (L2-06, implemented)
POST /reason  — run the ReAct loop, return next action     (L3-01, placeholder)
POST /parse   — validate/coerce raw LLM output to JSON     (L3-02, placeholder)
GET  /health  — liveness / readiness check

Run locally::

    cd llm_surface
    uv run uvicorn llm_surface.server:app --uds /tmp/agent.sock

Or over TCP for development::

    uv run uvicorn llm_surface.server:app --host 127.0.0.1 --port 8001 --reload
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field

from llm_surface.embeddings import EMBEDDING_MODEL, count_tokens, embed_text

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Body for ``POST /embed``.

    Matches the IPC contract in ``AGENTS.md``::

        {"text": "string", "model": "text-embedding-3-small"}
    """

    text: str = Field(..., min_length=1, description="Text to embed.")
    model: str = Field(
        default=EMBEDDING_MODEL,
        description="Embedding model. Defaults to text-embedding-3-small.",
    )


class EmbedResponse(BaseModel):
    """Successful response from ``POST /embed``.

    Matches the IPC contract in ``AGENTS.md``::

        {"vector": [0.0, ...], "token_count": 42}
    """

    vector: list[float] = Field(..., description="Embedding vector.")
    token_count: int = Field(..., description="Token count of the input text.")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources; inject them into ``app.state``."""
    # Load repo-root .env for local development. find_dotenv() walks up the
    # directory tree from the CWD, so this works regardless of where uvicorn
    # is invoked from. override=False means real env vars always take priority
    # (e.g. CI secrets injected by the pipeline).
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)

    app.state.openai = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        # Allow overriding the base URL for self-hosted / proxy setups.
        base_url=os.environ.get("OPENAI_API_BASE"),
    )
    app.state.embedding_model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    yield
    await app.state.openai.close()


app = FastAPI(
    title="llm-surface",
    version="0.1.0",
    description="Python IPC surface for the Xola agent runtime.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse, tags=["memory"])
async def embed(request: Request, body: EmbedRequest) -> EmbedResponse:
    """Embed *body.text* and return the float vector with token count.

    - **422** — Pydantic validation error (empty string, missing field).
    - **502** — OpenAI API error (network issue, invalid key, quota exceeded).
    """
    try:
        vector = await embed_text(
            body.text,
            client=request.app.state.openai,
            model=body.model,
        )
        token_count = count_tokens(body.text, model=body.model)
    except ValueError as exc:
        # embed_text raises ValueError on empty/whitespace — surface as 422.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    return EmbedResponse(vector=vector, token_count=token_count)


@app.post("/reason", tags=["planning"], status_code=501)
async def reason() -> dict[str, str]:
    """ReAct loop endpoint — not yet implemented (L3-01)."""
    return {"detail": "not implemented"}


@app.post("/parse", tags=["planning"], status_code=501)
async def parse() -> dict[str, str]:
    """Output parser endpoint — not yet implemented (L3-02)."""
    return {"detail": "not implemented"}
