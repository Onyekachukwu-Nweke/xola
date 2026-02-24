"""FastAPI IPC server — exposes HTTP endpoints toward the Rust runtime.

Endpoints
---------
POST /embed      — embed text, return a float vector          (L2-06, implemented)
POST /summarize  — condense short-term buffer into summary    (L2-08, implemented)
POST /reason     — run the ReAct loop, return next action     (L3-01, implemented)
POST /parse      — validate/coerce raw LLM output to JSON     (L4-03, implemented)
GET  /health     — liveness / readiness check

Run locally::

    cd llm_surface
    uv run uvicorn llm_surface.server:app --uds /tmp/agent.sock

Or over TCP for development::

    uv run uvicorn llm_surface.server:app --host 127.0.0.1 --port 8001 --reload
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field

from llm_surface.embeddings import EMBEDDING_MODEL, count_tokens, embed_text
from llm_surface.parser import ParseRequest, ParseResponse, validate_against_schema
from llm_surface.prompts import REASON_MODEL
from llm_surface.react import reason_step
from llm_surface.summarizer import SUMMARIZE_MODEL, SUMMARY_MAX_TOKENS, summarize_messages

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


class SummarizeRequest(BaseModel):
    """Body for ``POST /summarize``.

    Mirrors the Rust ``Message`` struct so the runtime can POST the current
    ``ShortTermMemory`` buffer directly::

        {"messages": [{"role": "user", "content": "...", "token_count": 12}, ...]}

    The ``token_count`` field on each message is accepted but ignored here —
    Python recomputes the count for the returned summary via tiktoken.
    """

    messages: list[dict[str, str]] = Field(
        ...,
        min_length=1,
        description="Conversation messages to condense ({role, content}).",
    )
    model: str = Field(
        default=SUMMARIZE_MODEL,
        description="Chat model used for summarisation.",
    )
    max_summary_tokens: int = Field(
        default=SUMMARY_MAX_TOKENS,
        ge=1,
        description="Upper bound on summary length in tokens.",
    )


class SummarizeResponse(BaseModel):
    """Successful response from ``POST /summarize``.

    The Rust runtime passes ``summary`` and ``token_count`` directly into
    ``ShortTermMemory::replace_with_summary``.
    """

    summary: str = Field(..., description="Condensed summary paragraph.")
    token_count: int = Field(..., description="Tiktoken count of the summary text.")


class ReasonMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="Message role: user, assistant, or system.")
    content: str = Field(..., description="Message content.")


class ToolSchema(BaseModel):
    """A tool descriptor matching ``ToolRegistry::list_schemas()`` output."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ReasonRequest(BaseModel):
    """Body for ``POST /reason``.

    Matches the IPC contract in ``AGENTS.md``::

        {"messages": [...], "tool_schemas": [...], "memory_context": [...], "task_goal": "..."}
    """

    messages: list[ReasonMessage] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    memory_context: list[str] = Field(default_factory=list)
    task_goal: str = Field(..., min_length=1, description="The high-level task goal.")


class ReasonResponse(BaseModel):
    """Successful response from ``POST /reason``.

    Matches the IPC contract in ``AGENTS.md``::

        {"thought": "...", "action": "...", "action_input": {...},
         "is_final": false, "final_answer": null}
    """

    thought: str
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    is_final: bool = False
    final_answer: str | None = None


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


@app.post("/summarize", response_model=SummarizeResponse, tags=["memory"])
async def summarize(request: Request, body: SummarizeRequest) -> SummarizeResponse:
    """Condense the short-term buffer into a single summary paragraph.

    Called by the Rust runtime when ``ShortTermMemory::needs_summarization``
    returns ``true`` (buffer ≥ 80 % of its token budget). The response is
    passed directly to ``ShortTermMemory::replace_with_summary``.

    - **422** — Pydantic validation error (empty messages list).
    - **502** — OpenAI API error.
    """
    try:
        summary_text, token_count = await summarize_messages(
            body.messages,
            client=request.app.state.openai,
            model=body.model,
            max_summary_tokens=body.max_summary_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (APIError, RuntimeError) as exc:
        # RuntimeError: empty LLM response (safety filter / malformed message)
        # APIError: network issue, quota exceeded, invalid key, etc.
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}") from exc

    return SummarizeResponse(summary=summary_text, token_count=token_count)


@app.post("/reason", response_model=ReasonResponse, tags=["planning"])
async def reason(request: Request, body: ReasonRequest) -> ReasonResponse:
    """Execute one ReAct reasoning step.

    Called by the Rust ``PlanExecutor`` on every iteration of the planning
    loop. Constructs a prompt from the conversation history, available tools,
    and memory context, then calls the LLM to decide the next action.

    - **422** — Pydantic validation error (empty goal, malformed request).
    - **502** — OpenAI API error or unparseable LLM response.
    """
    try:
        result = await reason_step(
            messages=[m.model_dump() for m in body.messages],
            tool_schemas=[t.model_dump() for t in body.tool_schemas],
            memory_context=body.memory_context,
            task_goal=body.task_goal,
            client=request.app.state.openai,
            model=os.environ.get("REASON_MODEL", REASON_MODEL),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    return ReasonResponse(**result)


@app.post("/parse", response_model=ParseResponse, tags=["planning"])
async def parse(body: ParseRequest) -> ParseResponse:
    """Validate and parse LLM output against a JSON Schema.

    This endpoint takes raw LLM output (which may be malformed JSON or not
    match the expected structure) and validates it against a provided JSON
    Schema. If validation fails, the response includes the error message.

    For retry logic with corrective prompting, use ``parse_with_retry`` from
    the parser module directly (L4-04).

    - **422** — Pydantic validation error (malformed request).
    - **200** — Parse attempt completed (check ``success`` field in response).

    Parameters
    ----------
    body : ParseRequest
        Contains raw LLM output, JSON Schema, and attempt number.

    Returns
    -------
    ParseResponse
        Parse result with success flag, parsed data, or error message.
    """
    result = validate_against_schema(body.raw, body.schema)
    return result
