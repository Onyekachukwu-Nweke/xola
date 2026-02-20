# Getting Started

This guide walks you through setting up the Xola development environment, running the stack locally, and executing your first agent task.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Rust** | 1.75+ | Runtime engine |
| **Python** | 3.11+ | LLM surface server |
| **Docker** + **Docker Compose** | Latest | Postgres, pgvector, Jaeger, code sandbox |
| **uv** | Latest | Python dependency management |
| **cargo-sqlx** | Latest | Database migrations |

### Install Prerequisites

```bash
# Rust (via rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# cargo-sqlx (for migrations)
cargo install sqlx-cli --no-default-features --features postgres

# Docker — install via your OS package manager or https://docs.docker.com/get-docker/
```

### Environment Variables

Create a `.env` file in the project root (never committed):

```env
# LLM API keys
OPENAI_API_KEY=sk-...

# Search API key (for WebSearchTool)
SERPER_API_KEY=...

# Database
DATABASE_URL=postgresql://xola:xola@localhost:5432/xola

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/xola.git
cd xola
```

### 2. Start Infrastructure

```bash
docker compose up -d postgres jaeger
```

This starts:
- **Postgres** with `pgvector` extension on port `5432`
- **Jaeger** on port `16686` (UI) and `4317` (OTLP receiver)

### 3. Run Database Migrations

```bash
cd runtime && cargo sqlx migrate run
```

### 4. Install Python Dependencies

```bash
cd llm_surface && uv sync
```

### 5. Start the Python LLM Surface

```bash
cd llm_surface && uv run uvicorn llm_surface.server:app --uds /tmp/agent.sock
```

The server listens on a Unix socket at `/tmp/agent.sock`.

### 6. Start the Rust Runtime

In a separate terminal:

```bash
cd runtime && cargo run -- --config config/local.toml
```

### 7. Send Your First Task

```bash
cargo run --bin cli -- --goal "find the three most cited RAG papers from 2024"
```

### 8. View the Trace

Open Jaeger at [http://localhost:16686](http://localhost:16686) to see the full execution trace.

---

## Directory Quick Reference

| Path | What's There |
|------|-------------|
| `runtime/src/` | Rust source — tools, memory, planning, reliability, observability |
| `llm_surface/src/` | Python source — prompts, ReAct parser, embeddings, FastAPI server |
| `config/` | Runtime configuration files (TOML) |
| `migrations/` | sqlx database migrations |
| `docker/` | Dockerfile definitions for sandbox containers |
| `tests/` | Integration tests spanning both processes |
| `docs/` | This documentation |

---

## Running Tests

```bash
# Rust unit tests
cd runtime && cargo test

# Rust lint
cd runtime && cargo clippy -- -D warnings

# Python tests
cd llm_surface && uv run pytest

# Python lint
cd llm_surface && uv run ruff check && uv run mypy --strict llm_surface

# Integration tests (requires infrastructure running)
cd tests && cargo test -- --include-ignored
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `connection refused` on Postgres | Ensure `docker compose up -d postgres` is running |
| `No such file: /tmp/agent.sock` | Start the Python server first |
| `OPENAI_API_KEY not set` | Create `.env` file with your API key |
| `sqlx` compile errors | Run `cargo sqlx prepare` to update query cache |
| Jaeger shows no traces | Check `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable |

---

## Next Steps

- Read the [Architecture](architecture.md) doc to understand the system design
- Check the [Roadmap](roadmap.md) to see what's being built and in what order
- See [Contributing](contributing.md) for development conventions and workflow
