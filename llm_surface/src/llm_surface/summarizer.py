"""Summarization helper — condenses a short-term message buffer via the LLM.

Called by the Rust runtime (``PlanExecutor``) when ``ShortTermMemory::needs_summarization``
fires (buffer ≥ 80 % of its token budget). The summary replaces the full buffer
via ``ShortTermMemory::replace_with_summary`` so context is preserved at low cost.

Flow
----
1. Rust detects ``stm.needs_summarization() == true``.
2. Rust calls ``POST /summarize`` with the current buffer messages.
3. This module condenses them using OpenAI chat completion.
4. Rust receives ``{summary, token_count}`` and calls ``stm.replace_with_summary``.

The summary is always placed as a ``system`` role message so the LLM treats it
as authoritative background context in subsequent turns.
"""

import logging

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_surface.embeddings import count_tokens

logger = logging.getLogger(__name__)

#: System prompt injected before the messages to be summarised.
_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a concise summarizer. The user will provide a conversation transcript. "
    "Produce a single, dense paragraph that preserves every key fact, decision, "
    "tool result, and error from the transcript. "
    "Do NOT add commentary. Do NOT omit errors or failures. "
    "Output only the summary paragraph — no headings, no bullet points."
)

#: Default chat model used for summarisation.
#: Intentionally separate from the embedding model; can be overridden via env.
SUMMARIZE_MODEL: str = "gpt-4o-mini"

#: Hard cap on the summary length in tokens. The Rust buffer is typically
#: 4 000–8 000 tokens; a summary over 512 tokens would eat too much headroom.
SUMMARY_MAX_TOKENS: int = 512


async def summarize_messages(
    messages: list[dict[str, str]],
    *,
    client: AsyncOpenAI,
    model: str = SUMMARIZE_MODEL,
    max_summary_tokens: int = SUMMARY_MAX_TOKENS,
    count_model: str = SUMMARIZE_MODEL,
) -> tuple[str, int]:
    """Condense *messages* into a single summary paragraph.

    Args:
        messages:           List of ``{"role": ..., "content": ...}`` dicts,
                            matching the Rust ``Message`` struct (JSON-serialised).
                            The ``token_count`` field is ignored here; Python
                            recomputes it for the summary.
        client:             Injected ``AsyncOpenAI`` client from the server lifespan.
        model:              Chat model to use for summarisation.
        max_summary_tokens: Target upper bound for the returned summary's token
                            count. Passed as ``max_tokens`` to the API so the
                            model self-truncates rather than running over budget.
        count_model:        Model name supplied to ``count_tokens`` for the final
                            token count of the returned summary.

    Returns:
        A ``(summary_text, token_count)`` 2-tuple. ``token_count`` is computed
        with tiktoken and must be passed back to Rust so it can update
        ``ShortTermMemory``'s budget accounting.

    Raises:
        ValueError:      If *messages* is empty.
        openai.APIError: On any error from the OpenAI chat completion API.
    """
    if not messages:
        raise ValueError("messages must be non-empty")

    # Build the transcript for the summarisation prompt.
    transcript_lines = [
        f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')}" for m in messages
    ]
    transcript = "\n".join(transcript_lines)

    chat_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarise the following conversation:\n\n{transcript}"},
    ]

    logger.debug(
        "summarizer: requesting summary for %d messages via model=%s",
        len(messages),
        model,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=chat_messages,
        max_tokens=max_summary_tokens,
        temperature=0.0,  # Deterministic — no creative variance in summaries.
    )

    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        raise RuntimeError(
            "Summarization returned an empty response — the model may have triggered "
            "a safety filter or returned a malformed message. "
            "The short-term buffer was NOT modified."
        )

    token_count = count_tokens(summary, model=count_model)

    logger.debug(
        "summarizer: summary produced (%d tokens, %d source messages)",
        token_count,
        len(messages),
    )

    return summary, token_count
