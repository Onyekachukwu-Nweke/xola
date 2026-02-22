# PROPOSAL: Add `POST /summarize` to the IPC Contract

**Status:** Human-approved (approved inline, 2026-02-22)
**Author:** LLM_SURFACE_AGENT
**Task:** L2-08 — Summarization fallback when short-term buffer fills

---

## Problem

`ShortTermMemory::needs_summarization()` (Rust, `runtime/src/memory/short_term.rs`) fires when
the token buffer reaches 80 % of its budget. At that point the Rust planning loop needs to
compress the current context window into a single dense paragraph — but all LLM calls must go
through the Python IPC surface (CLAUDE.md, process-boundary rule).

The original IPC contract defined three endpoints (`/reason`, `/embed`, `/parse`). None of them
is suitable for summarisation:

| Endpoint | Why unsuitable |
|----------|---------------|
| `/reason` | Returns a ReAct action JSON, not a plain-text summary |
| `/embed`  | Returns a vector, not text |
| `/parse`  | Validates existing LLM output, does not generate new text |

A dedicated `POST /summarize` endpoint is required.

---

## Proposed Change

### New endpoint: `POST /summarize`

**Request:**
```json
{
  "messages": [
    {"role": "user",      "content": "string", "token_count": 8},
    {"role": "assistant", "content": "string", "token_count": 12}
  ],
  "model":              "gpt-4o-mini",
  "max_summary_tokens": 512
}
```

- `messages` — the current `ShortTermMemory` buffer serialised to JSON (≥ 1 item).
- `model` — optional; defaults to `gpt-4o-mini`.
- `max_summary_tokens` — optional; defaults to 512. Passed as `max_tokens` to the chat
  completion API so the model self-truncates rather than exceeding the Rust budget.

**Response:**
```json
{
  "summary":     "Condensed single-paragraph summary of the conversation.",
  "token_count": 42
}
```

- `summary` — non-empty condensed paragraph. An empty LLM response raises a 502.
- `token_count` — tiktoken count of `summary` using the same model encoding. Rust passes
  this directly to `ShortTermMemory::replace_with_summary`, so budget accounting is accurate.

**Error codes:**
- `422` — `messages` is empty or missing.
- `502` — OpenAI API error or empty response from model (safety filter / malfunction).

### Rust call site (future, L3-03 / PlanExecutor)

```rust
if stm.needs_summarization() {
    let resp = ipc.post_json("/summarize", &SummarizeRequest {
        messages: stm.messages().iter().map(|m| m.into()).collect(),
        ..Default::default()
    }).await?;
    stm.replace_with_summary(Message::new(
        Role::System,
        resp.summary,
        resp.token_count,
    ));
}
```

---

## Impact

- **Rust:** No changes required now. `PlanExecutor` (L3-03) will need to add the call site.
- **Python:** New module `summarizer.py` + `POST /summarize` route in `server.py`.
- **Schema drift risk:** Low — the request mirrors the existing `Message` struct shape already
  used by `/reason`. The response is a strict subset of `/reason`'s response schema.
- **Cost:** One additional `gpt-4o-mini` call per summarisation event (~$0.0001 for 512 tokens).

---

## Alternatives Considered

| Alternative | Rejected because |
|-------------|-----------------|
| Squash into `/reason` body | Conflates planning and memory compression; complicates Rust call site |
| Rust-side sliding-window eviction only | Already implemented (`push` evicts oldest). Summarisation is a soft alternative that preserves context better. |
| Extend `/embed` to return a summary | `/embed` is a deterministic vector operation; adding LLM generation breaks that contract |
