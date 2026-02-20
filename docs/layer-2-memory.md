# Layer 2 — Memory Architecture

This is the most underappreciated part of agent systems. Agents fail not because the LLM is bad but because they **lose context at the wrong moment** or **retrieve the wrong memories**. Memory architecture determines reasoning quality.

**Related tasks:** L2-01 through L2-09 in [AGENTS.md](../AGENTS.md)

---

## Three Memory Types

Xola implements three distinct memory systems, each serving a different purpose:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Turn                               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Short-Term       │  │  Long-Term        │  │  Episodic     │ │
│  │  Memory           │  │  Memory           │  │  Memory       │ │
│  │                   │  │                   │  │               │ │
│  │  Recent messages  │  │  Semantic facts   │  │  Past task    │ │
│  │  in a circular    │  │  stored as        │  │  completions  │ │
│  │  buffer with      │  │  embeddings in    │  │  stored as    │ │
│  │  token budget     │  │  pgvector         │  │  structured   │ │
│  │                   │  │                   │  │  records      │ │
│  │  Scope: current   │  │  Scope: all time  │  │  Scope: all   │ │
│  │  conversation     │  │                   │  │  tasks        │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Short-Term Memory

The sliding context window — what happened recently in this conversation.

### Design

- **Data structure:** `VecDeque<Message>` acting as a circular buffer
- **Budget:** Capped by the model's token limit (configurable per model)
- **Eviction:** When the buffer fills, the oldest messages are evicted
- **Summarization fallback:** Before evicting, a summarization call compresses old messages into a single summary message — preserving the gist while freeing token budget

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| Circular buffer, not growing list | Token limits are hard — you can't send 100k tokens to a 32k context model |
| Summarize before evict | Naive eviction loses important early context; summarization preserves it |
| Token counting via `tiktoken` | Exact count, not character-based estimation |
| Token budget set per model | Different models have different context windows |

### Interface

```rust
pub struct ShortTermMemory {
    messages: VecDeque<Message>,
    token_count: usize,
    max_tokens: usize,
}

impl ShortTermMemory {
    pub fn push(&mut self, msg: Message);       // May trigger eviction
    pub fn messages(&self) -> &[Message];       // Current window
    pub fn token_count(&self) -> usize;         // Current usage
    pub fn needs_summarization(&self) -> bool;  // Over threshold?
}
```

---

## Long-Term Memory

Important facts that persist across conversations, stored as embeddings and retrieved by semantic similarity.

### Design

- **Storage:** Postgres with `pgvector` extension
- **Schema:** `memories` table with `id`, `content`, `embedding` (vector), `metadata` (JSONB), `created_at`
- **Write path:** Content → `/embed` endpoint → vector → INSERT into Postgres
- **Read path:** Query text → `/embed` → vector → cosine similarity search → top-K results

### When to Write

Not every message gets stored in long-term memory. The runtime writes to long-term memory when:

- A task completes successfully (key findings)
- The user explicitly states a fact the agent should remember
- The agent discovers something that will be useful across tasks

### When to Read

At the **start of each turn**, the runtime queries long-term memory with the current goal/context and prepends relevant memories to the prompt. This gives the agent access to knowledge it acquired in previous sessions.

### Interface

```rust
pub struct LongTermMemory {
    pool: PgPool,
}

impl LongTermMemory {
    pub async fn store(&self, content: &str, embedding: Vec<f32>, metadata: Value) -> Result<()>;
    pub async fn query(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<MemoryRecord>>;
    pub async fn delete(&self, id: Uuid) -> Result<()>;
}
```

### SQL

```sql
CREATE TABLE memories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(1536) NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Episodic Memory

A structured log of past task completions — what was tried, what tools were called, what succeeded or failed.

### Design

- **Storage:** Postgres `episodes` table
- **Schema:** `id`, `task_goal`, `steps` (JSONB array), `outcome` (success/failure), `error`, `duration_ms`, `created_at`
- **Purpose:** The agent can look up how it previously solved similar problems
- **Retrieval:** By embedding similarity on `task_goal`, or by structured query

### What Gets Logged

Each episode records:

```json
{
  "task_goal": "find the three most cited RAG papers from 2024",
  "steps": [
    { "action": "web_search", "input": {"query": "most cited RAG papers 2024"}, "success": true },
    { "action": "url_fetch", "input": {"url": "..."}, "success": true },
    { "action": "url_fetch", "input": {"url": "..."}, "success": false, "error": "timeout" }
  ],
  "outcome": "success",
  "duration_ms": 45200,
  "created_at": "2026-03-15T10:30:00Z"
}
```

### Interface

```rust
pub struct EpisodicLog {
    pool: PgPool,
}

impl EpisodicLog {
    pub async fn record(&self, episode: Episode) -> Result<()>;
    pub async fn query_similar(&self, goal_embedding: Vec<f32>, top_k: usize) -> Result<Vec<Episode>>;
}
```

---

## The Embedding Pipeline

All semantic operations flow through the Python `/embed` endpoint:

```
Content (text) → POST /embed → Python → OpenAI text-embedding-3-small → Vec<f32> → Rust
```

| Field | Detail |
|-------|--------|
| **Model** | `text-embedding-3-small` (configurable) |
| **Dimensions** | 1536 |
| **Token counting** | `tiktoken` — exact count returned alongside vector |
| **Batching** | Single text per call (batch support future) |

---

## What You'll Learn

Building this layer teaches:

- **Embedding models** — how text becomes vectors, what similarity means
- **Vector similarity search** — pgvector, cosine distance, index types (IVFFlat vs HNSW)
- **Token counting** — why exact counts matter and how `tiktoken` works
- **Summarization as compression** — using the LLM to reduce context without losing meaning
- **Read vs write trade-offs** — when to persist knowledge and when it's noise

---

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Unit tests | Short-term memory eviction, token counting | `runtime/src/memory/` |
| Unit tests | Summarization trigger logic | `llm_surface/` |
| Integration tests | Store embedding → retrieve by similarity | `tests/` |
| Integration tests | Full turn with memory read + write | `tests/` |
