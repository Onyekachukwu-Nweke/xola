# AGENTS.md

This file is for autonomous coding agents (Claude Code, GPT-4o with code tools, etc.) operating in this repository. It defines task ownership, execution rules, what agents are allowed to do, and what they must escalate to a human.

Read `CLAUDE.md` first for project context and conventions. This file assumes you have.

---

## Agent Roles

Three agent personas are defined for this project. An agent should identify which role it is operating in at the start of a session and restrict itself accordingly.

---

### Role: `RUNTIME_AGENT`

**Scope:** The `runtime/` Rust crate only.

**Permitted actions:**
- Implement or modify anything under `runtime/src/`
- Add or update Cargo dependencies with justification in the commit message
- Write and run unit tests (`cargo test`)
- Run `cargo clippy` and `cargo fmt`
- Add sqlx migrations under `migrations/`
- Read (never write) files in `llm_surface/` for interface reference

**Prohibited actions:**
- Modifying `llm_surface/` in any way
- Changing the IPC contract (socket protocol, endpoint schemas) without creating a `PROPOSAL.md` for human review
- Adding any Python code
- Calling LLM APIs directly from Rust (all LLM calls go through the Python IPC server)
- Removing observability instrumentation (tracing spans, metrics)

**Escalate to human when:**
- The IPC schema needs to change
- A new external service dependency is required (new Docker socket, new third-party API)
- A circuit breaker threshold needs tuning (business decision, not engineering)
- Any `unsafe` block is required

---

### Role: `LLM_SURFACE_AGENT`

**Scope:** The `llm_surface/` Python package only.

**Permitted actions:**
- Implement or modify anything under `llm_surface/llm_surface/`
- Add or update Python dependencies with `uv add`
- Write and run pytest tests
- Run `ruff check`, `ruff format`, `mypy`
- Modify the FastAPI server routes — but not their URL paths or Pydantic schemas without a `PROPOSAL.md`

**Prohibited actions:**
- Modifying `runtime/` in any way
- Adding state to the Python server. It must remain stateless between requests.
- Adding new LLM API providers without human approval (cost implication)
- Hardcoding prompt templates that should be configurable — use `config/prompts.toml`
- Swallowing exceptions in the retry loop without logging the raw failure

**Escalate to human when:**
- The ReAct prompt template needs structural changes (affects agent behavior globally)
- A new LLM provider is needed
- Token budget logic changes (cost implication)
- The IPC schema needs to change

---

### Role: `INTEGRATION_AGENT`

**Scope:** `tests/`, `docker/`, `config/`, `migrations/`. Read access to both `runtime/` and `llm_surface/`.

**Permitted actions:**
- Write and run integration tests in `tests/`
- Modify Docker Compose files
- Add or modify config files under `config/`
- Add sqlx migrations (additive only — no destructive schema changes without human approval)
- Run the full test suite

**Prohibited actions:**
- Modifying source code in `runtime/` or `llm_surface/`
- Running integration tests marked `#[ignore]` / `pytest.mark.integration` in CI without explicit instruction
- Changing database schema destructively

**Escalate to human when:**
- An integration test reveals a cross-process contract mismatch
- A migration would require data transformation on existing rows
- A new external service needs to be added to the Docker Compose stack

---

## Task Board

Tasks are listed by layer. Each task has an ID, description, owning role, status, and any blockers.

### Layer 1: Tool Registry & Execution Sandbox

| ID | Task | Role | Status | Blocker |
|----|------|------|--------|---------|
| L1-01 | Define `Tool` trait with `execute(&self, input: Value) -> Result<Value>` | RUNTIME | 🔲 todo | — |
| L1-02 | Implement `ToolRegistry` as `HashMap<String, Arc<dyn Tool>>` | RUNTIME | 🔲 todo | L1-01 |
| L1-03 | JSON Schema validation on tool inputs via `jsonschema` crate | RUNTIME | 🔲 todo | L1-02 |
| L1-04 | Per-tool `tokio::time::timeout` with configurable duration | RUNTIME | 🔲 todo | L1-02 |
| L1-05 | Implement `UrlFetchTool` (reqwest, returns page text) | RUNTIME | 🔲 todo | L1-03 |
| L1-06 | Implement `WebSearchTool` (Serper or Brave API) | RUNTIME | 🔲 todo | L1-03 |
| L1-07 | Implement `CodeExecTool` via Docker SDK | RUNTIME | 🔲 todo | L1-03 |
| L1-08 | Execution trace: log inputs, outputs, latency, errors per call | RUNTIME | 🔲 todo | L1-04 |
| L1-09 | Integration test: call each tool end-to-end | INTEGRATION | 🔲 todo | L1-05, L1-06, L1-07 |

### Layer 2: Memory Architecture

| ID | Task | Role | Status | Blocker |
|----|------|------|--------|---------|
| L2-01 | Postgres schema: `memories`, `episodes` tables | INTEGRATION | 🔲 todo | — |
| L2-02 | pgvector extension and `embedding` column on `memories` | INTEGRATION | 🔲 todo | L2-01 |
| L2-03 | `ShortTermMemory`: VecDeque with token budget; push evicts oldest | RUNTIME | 🔲 todo | — |
| L2-04 | `LongTermMemory`: write embedding + metadata; semantic query via pgvector | RUNTIME | 🔲 todo | L2-02 |
| L2-05 | `EpisodicLog`: structured insert per task completion | RUNTIME | 🔲 todo | L2-01 |
| L2-06 | `/embed` endpoint: accepts text, returns `Vec<f32>` | LLM_SURFACE | 🔲 todo | — |
| L2-07 | tiktoken budget helper: `count_tokens(text: str, model: str) -> int` | LLM_SURFACE | 🔲 todo | — |
| L2-08 | Summarization fallback when short-term buffer fills | LLM_SURFACE | 🔲 todo | L2-07 |
| L2-09 | Integration test: store and retrieve a memory by semantic similarity | INTEGRATION | 🔲 todo | L2-04, L2-06 |

### Layer 3: Task Planning & Orchestration

| ID | Task | Role | Status | Blocker |
|----|------|------|--------|---------|
| L3-01 | `/reason` endpoint: accepts context + tool schemas, returns action JSON | LLM_SURFACE | 🔲 todo | — |
| L3-02 | ReAct loop parser: extract `Thought`, `Action`, `Action Input`, `Observation` | LLM_SURFACE | 🔲 todo | L3-01 |
| L3-03 | `PlanExecutor` in Rust: sequential step runner over action list | RUNTIME | 🔲 todo | L1-02 |
| L3-04 | Parallel branch support: `tokio::task::JoinSet` fan-out | RUNTIME | 🔲 todo | L3-03 |
| L3-05 | Replan trigger: on tool error, call `/reason` with failure context | RUNTIME | 🔲 todo | L3-03, L3-01 |
| L3-06 | Max replanning attempts config; escalate to error after N failures | RUNTIME | 🔲 todo | L3-05 |
| L3-07 | Integration test: multi-step research task end-to-end | INTEGRATION | 🔲 todo | L3-05, L2-04 |

### Layer 4: Reliability & Fault Tolerance

| ID | Task | Role | Status | Blocker |
|----|------|------|--------|---------|
| L4-01 | `CircuitBreaker` struct: Closed/Open/HalfOpen FSM with atomic state | RUNTIME | 🔲 todo | — |
| L4-02 | Wire `CircuitBreaker` per tool in registry | RUNTIME | 🔲 todo | L4-01, L1-02 |
| L4-03 | `/parse` endpoint: validate LLM output against Pydantic schema | LLM_SURFACE | 🔲 todo | — |
| L4-04 | Corrective re-prompt on parse failure; up to N retries | LLM_SURFACE | 🔲 todo | L4-03 |
| L4-05 | `LoopDetector`: sliding window of last K tool call hashes; abort on cycle | RUNTIME | 🔲 todo | — |
| L4-06 | Hierarchical timeout: tool < step < task; all via `CancellationToken` tree | RUNTIME | 🔲 todo | L1-04, L3-03 |
| L4-07 | Failure taxonomy: distinguish timeout / parse_error / tool_error / loop | RUNTIME | 🔲 todo | L4-06 |
| L4-08 | Integration test: inject tool failure, verify replan and recovery | INTEGRATION | 🔲 todo | L4-02, L3-05 |

### Layer 5: Observability

| ID | Task | Role | Status | Blocker |
|----|------|------|--------|---------|
| L5-01 | Add `tracing` spans to tool dispatch, memory read/write, plan steps | RUNTIME | 🔲 todo | L1-08 |
| L5-02 | OTel OTLP exporter → Jaeger | RUNTIME | 🔲 todo | L5-01 |
| L5-03 | Prometheus: tool success rate, avg plan steps, memory hit rate, token spend | RUNTIME | 🔲 todo | — |
| L5-04 | Cost accounting: token count × per-model price attached to each LLM span | LLM_SURFACE | 🔲 todo | — |
| L5-05 | axum SSR web UI: live run list, step inspector, memory browser | RUNTIME | 🔲 todo | L5-01 |
| L5-06 | Jaeger in Docker Compose for local dev | INTEGRATION | 🔲 todo | — |
| L5-07 | Integration test: run agent task, assert trace contains expected span types | INTEGRATION | 🔲 todo | L5-02 |

---

## Execution Rules for Agents

These apply to all roles.

**Before starting any task:**
1. Read `CLAUDE.md` in full.
2. Run the existing test suite and confirm it passes. Do not start on a red suite.
3. Identify the task ID you are working on and state it in your first commit message.

**While working:**
- Make commits at logical checkpoints — do not accumulate an entire layer in one commit.
- If a task requires changing code outside your role's scope, stop and write a `PROPOSAL.md` describing the change needed and why. Do not make the change yourself.
- If you encounter a failing test you did not write, investigate before proceeding. Do not delete tests to make the suite green.
- Do not add `println!` or `print()` debug statements to committed code. Use `tracing::debug!` in Rust and Python's `logging` module.

**Before marking a task done:**
- All tests pass (`cargo test`, `pytest`)
- Lint passes (`cargo clippy -- -D warnings`, `ruff check`, `mypy --strict`)
- The task's integration test (if listed) passes with `-- --include-ignored` / `-m integration`
- No secrets, no `unwrap()` on LLM output paths, no stateful Python server handlers

**Never do without human approval:**
- Change the IPC contract between Rust and Python
- Add a new external API dependency (search provider, LLM provider, etc.)
- Modify database schema destructively
- Add `unsafe` code in Rust
- Change token budget or cost accounting logic
- Modify the ReAct prompt template structurally

---

## IPC Contract (Canonical)

This is the source of truth for the Rust ↔ Python boundary. Both `RUNTIME_AGENT` and `LLM_SURFACE_AGENT` must treat this as immutable without a human-approved `PROPOSAL.md`.

**`POST /reason`**

Request:
```json
{
  "messages": [{"role": "user"|"assistant", "content": "string"}],
  "tool_schemas": [{"name": "string", "description": "string", "parameters": {}}],
  "memory_context": ["string"],
  "task_goal": "string"
}
```

Response:
```json
{
  "thought": "string",
  "action": "string | null",
  "action_input": {},
  "is_final": false,
  "final_answer": "string | null"
}
```

**`POST /embed`**

Request: `{"text": "string", "model": "text-embedding-3-small"}`
Response: `{"vector": [0.0, ...], "token_count": 42}`

**`POST /parse`**

Request: `{"raw": "string", "schema": {}, "attempt": 1}`
Response: `{"parsed": {}, "success": true, "error": null}`

---

## Definition of MVP

The project is at MVP when all of the following are true:

- [ ] Three tools registered and working: `url_fetch`, `web_search`, `code_exec`
- [ ] Short-term and long-term memory wired up and queried each turn
- [ ] ReAct loop completes a multi-step research task without human intervention
- [ ] Replanning fires correctly on a tool failure (verified by integration test)
- [ ] Every run produces a visible trace in Jaeger with tool call spans
- [ ] `cargo clippy -- -D warnings` and `mypy --strict` both pass on main
- [ ] README has a working quickstart that a new developer can follow in under 20 minutes