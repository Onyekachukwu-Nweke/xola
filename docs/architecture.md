# Architecture

Xola is a **two-process AI agent runtime**. It is not a wrapper around LangChain or any existing framework — it is the infrastructure layer that makes an LLM-powered agent reliable. Think of it the way a JVM sits between bytecode and hardware: the LLM is just one component inside the runtime.

---

## Design Philosophy

Most AI engineering work is spent calling APIs. The hard problem — and the one that determines whether an agent works in production — is the **infrastructure around the LLM**: execution safety, memory management, planning with recovery, fault tolerance, and observability. Xola exists to solve that problem.

Agents fail not because the LLM is bad, but because they lose context at the wrong moment, repeat failed actions, or crash on malformed output. Every architectural decision in Xola is a response to one of these failure modes.

---

## The Two-Process Model

Xola runs as two cooperating processes:

```
┌─────────────────────────┐       Unix Socket / gRPC       ┌──────────────────────────┐
│      Rust Runtime       │ ──────────────────────────────▶ │    Python LLM Surface    │
│                         │                                 │                          │
│  • Tool dispatch        │         POST /reason            │  • Prompt construction   │
│  • Memory storage       │         POST /embed             │  • ReAct parsing         │
│  • Plan execution       │         POST /parse             │  • Output validation     │
│  • Circuit breakers     │                                 │  • Embedding calls       │
│  • Timeout hierarchy    │ ◀────────────────────────────── │  • LLM API calls         │
│  • Observability        │        Structured JSON          │  • Token counting        │
└─────────────────────────┘                                 └──────────────────────────┘
```

**Why two processes?**

- The borrow checker and async Python don't mix well in a single binary.
- The Python surface is independently testable and deployable.
- The Rust runtime owns all state and concurrency — Python remains stateless between requests.
- Data flows **one direction** at the IPC boundary: Rust calls Python. Python never calls into Rust.

---

## The Five Layers

Xola is organized into five architectural layers, each building on the one below it. This is both the code structure and the build order.

```
┌───────────────────────────────────────────────────┐
│  Layer 5: Observability & Runtime Introspection   │
├───────────────────────────────────────────────────┤
│  Layer 4: Reliability & Fault Tolerance           │
├───────────────────────────────────────────────────┤
│  Layer 3: Task Planning & Orchestration           │
├───────────────────────────────────────────────────┤
│  Layer 2: Memory Architecture                     │
├───────────────────────────────────────────────────┤
│  Layer 1: Tool Registry & Execution Sandbox       │
└───────────────────────────────────────────────────┘
```

| Layer | Responsibility | Key Abstractions |
|-------|---------------|-----------------|
| **1 — Tools** | Register, validate, execute, and trace tool calls | `Tool` trait, `ToolRegistry`, JSON Schema, Docker sandbox |
| **2 — Memory** | Maintain context across turns and across tasks | Short-term buffer, pgvector long-term store, episodic log |
| **3 — Planning** | Break goals into steps, execute, observe, replan | ReAct loop, `PlanExecutor`, fan-out/fan-in, replan triggers |
| **4 — Reliability** | Handle the weird, slow failures agents produce | Circuit breakers, loop detector, output enforcement, timeout tree |
| **5 — Observability** | Make everything debuggable and monitorable | OpenTelemetry traces, Prometheus metrics, web UI |

Each layer has a dedicated guide: [Layer 1](layer-1-tools.md) · [Layer 2](layer-2-memory.md) · [Layer 3](layer-3-planning.md) · [Layer 4](layer-4-reliability.md) · [Layer 5](layer-5-observability.md)

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
│   │   ├── ipc/          # Unix socket / gRPC client toward Python
│   │   └── observe/      # tracing, OTel, Prometheus
│   ├── Cargo.toml
│   └── Cargo.lock
│
├── llm_surface/          # Python package — LLM-facing logic
│   ├── src/
│   │   ├── client.py     # OpenAI / Anthropic SDK wrapper
│   │   ├── prompts.py    # System prompt builder, ReAct template
│   │   ├── react.py      # ReAct loop parser
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
├── docs/                 # This documentation
└── README.md
```

---

## Language Ownership

The language boundary is strict. If you find yourself writing the wrong kind of logic in the wrong language, stop and move it.

**Rust owns:**
- Everything concurrent (`tokio`, `JoinSet`, `CancellationToken`)
- Tool trait, registry, and dispatch
- Memory storage (`sqlx`, pgvector queries)
- Circuit breakers, loop detection, timeout trees
- OpenTelemetry spans, Prometheus metrics
- The Unix socket / gRPC server

**Python owns:**
- Prompt construction and the ReAct template
- Parsing `Thought / Action / Observation` from raw LLM output
- Corrective re-prompting on malformed JSON
- Embedding API calls
- `tiktoken`-based token counting
- All LLM SDK calls

---

## Data Flow — A Single Agent Turn

```
1. User submits a goal via CLI
2. Rust runtime loads short-term memory (recent messages)
3. Rust queries long-term memory (pgvector semantic search) for relevant context
4. Rust serializes context + tool schemas → POST /reason
5. Python constructs prompt, calls LLM, parses ReAct output
6. Python returns { thought, action, action_input } to Rust
7. Rust validates action_input against tool's JSON Schema
8. Rust dispatches tool with context timeout
9. Tool result (observation) is appended to short-term memory
10. Repeat from step 4 until Python returns { is_final: true }
11. Final answer returned to user; episodic log written; trace exported
```

---

## Key Technology Choices

### Rust Side

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime — all concurrency lives here |
| `axum` | Web UI server (SSR HTML) |
| `serde` / `serde_json` | Structured data serialization |
| `sqlx` | Postgres async driver with compile-time query checking |
| `tracing` | Structured logging and span instrumentation |
| `opentelemetry` | OTel SDK; export to Jaeger via OTLP |
| `tokio-util` | `CancellationToken` for timeout hierarchy |
| `jsonschema` | Tool input validation |

### Python Side

| Package | Purpose |
|---------|---------|
| `openai` / `anthropic` | LLM API clients |
| `tiktoken` | Token counting (canonical — do not reimplement) |
| `pydantic` | Output schema validation |
| `fastapi` + `uvicorn` | IPC server |
| `httpx` | Async HTTP client |

---

## What This Project Teaches

This project teaches the **infrastructure that makes AI systems reliable** — the actual hard problem, and the one companies hire for. By the end you'll understand:

- Why agents fail in production and how to prevent it
- How memory architecture affects reasoning quality
- How to make non-deterministic systems observable and debuggable
- The engineering patterns behind tool dispatch, circuit breaking, and hierarchical timeouts

This is not about calling APIs. This is about building the engine that sits beneath them.
