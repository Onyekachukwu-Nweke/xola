# CLAUDE.md

This file is the orientation document for Claude when working in this repository. Read it before touching any code.

---

## What This Project Is

A **two-process AI agent runtime** built in Rust and Python. It is not a wrapper around LangChain or any existing agent framework. It is the infrastructure layer that sits between a raw LLM and a reliable autonomous system — handling tool dispatch, memory, planning, fault tolerance, and observability.

The analogy is a JVM: the LLM is one component inside the runtime, not the runtime itself.

---

## Repository Layout

```
xola/
├── runtime/              # Rust crate — the core engine
│   ├── src/
│   │   ├── main.rs
│   │   ├── tools/        # Tool trait, registry, dispatcher, sandbox
│   │   ├── memory/       # Short-term buffer, pgvector store, episodic log
│   │   ├── planning/     # Plan executor, DAG runner, JoinSet fan-out
│   │   ├── reliability/  # Circuit breakers, loop detector, timeout hierarchy
│   │   ├── ipc/          # Unix socket / gRPC server toward Python
│   │   └── observe/      # tracing, OTel, Prometheus
│   ├── Cargo.toml
│   └── Cargo.lock
│
├── llm_surface/          # Python package — LLM-facing logic
│   ├── src/
│   │   ├── client.py     # OpenAI / Anthropic SDK wrapper
│   │   ├── prompts.py    # System prompt builder, ReAct template
│   │   ├── react.py      # ReAct loop: Thought/Action/Observation parser
│   │   ├── parser.py     # Pydantic output validation, corrective retry
│   │   ├── embeddings.py # Embedding API calls, tiktoken budgeting
│   │   └── server.py     # FastAPI server exposing /reason /embed /parse
│   ├── pyproject.toml
│   └── uv.lock
│
├── docker/               # Sandbox container definitions
├── migrations/           # sqlx migration files for Postgres
├── config/               # Runtime config (TOML)
├── tests/                # Integration tests spanning both processes
└── CLAUDE.md             # This file
```

---

## The Process Boundary

Rust and Python run as **two separate processes** communicating over a Unix socket (default) or gRPC. Do not collapse this into a single process unless there is a compelling performance reason — the separation keeps the borrow checker from fighting async Python and keeps the Python surface independently testable.

Python exposes three endpoints:

| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /reason` | Context window, tool schemas, last observation | Next action JSON |
| `POST /embed` | Text string | Float vector |
| `POST /parse` | Raw LLM string, target schema | Validated JSON or error |

Rust calls these endpoints. Python never calls into Rust. Data flows one way at the IPC boundary.

---

## Language Ownership

**Rust owns:**
- Everything concurrent (tokio, JoinSet, CancellationToken)
- Tool trait, registry, dispatcher
- Memory storage (sqlx, pgvector queries)
- Circuit breakers, loop detector, timeout trees
- OpenTelemetry spans, Prometheus metrics
- The Unix socket / gRPC server

**Python owns:**
- Prompt construction and the ReAct template
- Parsing `Thought / Action / Observation` from raw LLM output
- Corrective re-prompting on malformed JSON
- Calling the embedding API
- tiktoken-based token counting
- All LLM SDK calls

**Never mix these.** If you find yourself writing async Rust that needs to evaluate prompt logic, stop and move the logic to Python. If you find Python trying to manage timeouts or concurrency, stop and move it to Rust.

---

## Key Crates (Rust)

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime. Everything async lives here. |
| `axum` | Web UI server (SSR HTML) and internal HTTP if needed |
| `serde` / `serde_json` | All structured data serialization |
| `sqlx` | Postgres async driver with compile-time query checking |
| `tracing` | Structured logging and span instrumentation |
| `opentelemetry` | OTel SDK; export to Jaeger via OTLP |
| `tokio-util` | `CancellationToken` for the timeout hierarchy |
| `jsonschema` | Validate tool inputs before dispatch |

Add crates deliberately. Every new dependency needs a clear reason in the PR description.

---

## Key Packages (Python)

| Package | Purpose |
|---------|---------|
| `openai` / `anthropic` | LLM API clients |
| `tiktoken` | Token counting (canonical; do not reimplement) |
| `pydantic` | Output schema validation and corrective error context |
| `fastapi` + `uvicorn` | IPC server exposing the three endpoints |
| `numpy` | Embedding math if needed locally |
| `httpx` | Async HTTP client for any outbound calls |

Use `uv` for all dependency management. Do not use pip directly.

---

## Development Conventions

**Rust:**
- `cargo clippy -- -D warnings` must pass before any commit
- `cargo fmt` is mandatory
- Prefer `Result<T, E>` with `thiserror`-derived errors; no `unwrap()` in production paths
- Every `tokio::spawn` must have its `JoinHandle` accounted for — no fire-and-forget without a rationale comment
- Timeout every external call. Nothing blocks forever.

**Python:**
- `ruff check` and `ruff format` before commit
- `mypy --strict` on the `llm_surface` package
- All IPC endpoint handlers are typed with Pydantic models — no raw dicts crossing the API boundary
- Do not swallow exceptions silently in the retry loop — always log the raw LLM response that failed parsing

**Both:**
- No secrets in source. Use environment variables loaded via `config/` at startup.
- Every new tool must have: a JSON Schema descriptor, a unit test with mocked output, and an integration test against the real service with `#[ignore]` / `pytest.mark.integration`

---

## Running Locally

```bash
# 1. Start Postgres with pgvector
docker compose up -d postgres

# 2. Run migrations
cd runtime && cargo sqlx migrate run

# 3. Start the Python IPC server
cd llm_surface && uv run uvicorn llm_surface.server:app --uds /tmp/agent.sock

# 4. Start the Rust runtime
cd runtime && cargo run -- --config config/local.toml

# 5. Send a task
cargo run --bin cli -- --goal "find the three most cited RAG papers from 2024"
```

Traces appear in Jaeger at `http://localhost:16686`.

---

## What Not To Do

- Do not add LangChain, LlamaIndex, or any agent framework as a dependency. The point of this project is to build what those abstract over.
- Do not use `unwrap()` or `expect()` on paths that can be reached by bad LLM output.
- Do not add a new tool without adding it to the tool registry documentation in `runtime/src/tools/README.md`.
- Do not let the Python server become stateful. All state lives in Rust / Postgres. Python is stateless between requests.
- Do not use `async_std` — the project is `tokio` throughout.

---

## Current Status

See `AGENTS.md` for the task breakdown, layer ownership, and what is in progress.