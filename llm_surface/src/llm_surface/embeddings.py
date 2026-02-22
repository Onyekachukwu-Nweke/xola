"""Embedding and token-counting utilities.

This module contains the embedding call logic and a thin tiktoken wrapper.
FastAPI wiring lives in ``server.py``. Keeping them separate makes the helpers
trivially unit-testable with an injected mock client.

The default model is ``text-embedding-3-small``, which produces 1536-dimensional
vectors. This **must** match the ``EMBEDDING_DIM`` constant in the Rust
``LongTermMemory`` implementation and the ``vector(1536)`` column type in
migration ``0001_create_memories_table.sql``.
"""

import logging

import tiktoken
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

#: OpenAI model used for embeddings.
#: Must match ``EMBEDDING_DIM = 1536`` in ``runtime/src/memory/long_term.rs``
#: and the ``vector(1536)`` column in ``migrations/0001_create_memories_table.sql``.
EMBEDDING_MODEL: str = "text-embedding-3-small"


def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Return the number of tokens in *text* for the given *model*.

    Uses ``tiktoken`` for canonical token counting. Falls back to
    ``cl100k_base`` if the model name is not recognised by tiktoken.

    Raises:
        ValueError: If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("tiktoken has no encoding for model %r, falling back to cl100k_base", model)
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


async def embed_text(
    text: str,
    *,
    client: AsyncOpenAI,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Return the embedding vector for *text*.

    Args:
        text:   Text to embed. Must be non-empty (after stripping whitespace).
        client: Injected ``AsyncOpenAI`` client, managed by the server lifespan.
        model:  OpenAI embedding model. Defaults to ``text-embedding-3-small``
                (1536 dimensions).

    Returns:
        Flat list of floats with length equal to the model's embedding dimension
        (1536 for the default model).

    Raises:
        ValueError:       If *text* is empty or whitespace-only.
        openai.APIError:  On any error from the OpenAI API.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
