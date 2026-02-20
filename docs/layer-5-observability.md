# Layer 5 — Observability & Runtime Introspection

This layer makes everything else debuggable. Without observability, agent failures are black boxes — you know *something* went wrong but not *where*, *why*, or *how much it cost*. This layer turns every agent run into a fully inspectable trace.

**Related tasks:** L5-01 through L5-07 in [AGENTS.md](../AGENTS.md)

---

## Three Pillars

```
┌─────────────────────────────────────────────────────────┐
│                    Observability                         │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Tracing   │  │   Metrics   │  │    Web UI       │ │
│  │             │  │             │  │                 │ │
│  │ OpenTelemetry│  │ Prometheus  │  │  axum SSR HTML  │ │
│  │ → Jaeger    │  │ counters    │  │  live run view  │ │
│  │             │  │ histograms  │  │  step inspector │ │
│  │ Per-run     │  │ gauges      │  │  memory browser │ │
│  │ trace tree  │  │             │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Distributed Tracing

Every agent run produces a **trace** — a tree of spans representing each operation:

```
agent_run (trace root)
├── memory_read (long-term memory query)
│   ├── embed_text (POST /embed)
│   └── pgvector_query
├── plan_step_1
│   ├── reason (POST /reason)
│   │   └── llm_call (tokens_in=1200, tokens_out=150, cost=$0.003)
│   ├── validate_input (JSON Schema check)
│   └── tool_call: web_search
│       ├── api_request (latency=450ms)
│       └── response_parse
├── plan_step_2
│   ├── reason (POST /reason)
│   │   └── llm_call (tokens_in=2400, tokens_out=200, cost=$0.005)
│   └── tool_call: url_fetch
│       └── http_request (latency=1200ms)
├── memory_write (store findings in long-term memory)
│   └── embed_text (POST /embed)
└── episodic_log_write
```

### What Every Span Records

| Attribute | Example |
|-----------|---------|
| `span.name` | `tool_call:web_search` |
| `span.duration_ms` | `450` |
| `tool.input` | `{"query": "RAG papers 2024"}` |
| `tool.output` | `{"results": [...]}` |
| `llm.model` | `gpt-4o` |
| `llm.tokens_in` | `1200` |
| `llm.tokens_out` | `150` |
| `llm.cost_usd` | `0.003` |
| `error.type` | `TOOL_TIMEOUT` |
| `error.message` | `exceeded 30s deadline` |

### Implementation

Using the `tracing` crate with OpenTelemetry export:

```rust
use tracing::{instrument, info_span, Instrument};
use opentelemetry::trace::Tracer;

#[instrument(skip(self), fields(tool.name = %name))]
pub async fn dispatch(&self, name: &str, input: Value) -> Result<Value> {
    let span = info_span!("tool_call", tool.name = %name);

    async {
        let result = self.execute_tool(name, input).await;
        tracing::info!(latency_ms = %elapsed, "tool completed");
        result
    }
    .instrument(span)
    .await
}
```

### Export Pipeline

```
tracing spans → OpenTelemetry SDK → OTLP exporter → Jaeger
```

Traces are viewable at `http://localhost:16686` (Jaeger UI) during local development.

---

## Prometheus Metrics

Operational metrics exposed at `/metrics` for scraping:

| Metric | Type | Description |
|--------|------|-------------|
| `xola_tool_calls_total` | Counter | Total tool calls, labeled by `tool_name` and `status` |
| `xola_tool_duration_seconds` | Histogram | Tool call latency distribution |
| `xola_tool_circuit_breaker_state` | Gauge | Current circuit breaker state per tool |
| `xola_plan_steps_total` | Counter | Planning steps per task |
| `xola_plan_replans_total` | Counter | Replanning events |
| `xola_memory_queries_total` | Counter | Memory reads, labeled by `memory_type` |
| `xola_memory_hit_rate` | Gauge | Percentage of memory queries returning results |
| `xola_llm_tokens_total` | Counter | Tokens consumed, labeled by `model` and `direction` |
| `xola_llm_cost_usd_total` | Counter | Cumulative LLM spend |
| `xola_task_duration_seconds` | Histogram | End-to-end task completion time |

---

## Cost Accounting

Every LLM call tracks token usage and maps it to cost:

```python
# In llm_surface — cost calculation per call
PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000},
}

def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = PRICING[model]
    return tokens_in * prices["input"] + tokens_out * prices.get("output", 0)
```

Cost is attached to every LLM span and aggregated in Prometheus. This answers: *"How much did this agent run cost?"*

---

## Web UI

A simple server-side rendered web interface built with `axum`:

### Views

| View | What It Shows |
|------|---------------|
| **Run List** | All agent runs with status, duration, cost, step count |
| **Run Detail** | Step-by-step trace of a single run with thought/action/observation |
| **Step Inspector** | Detailed view of one step: tool input/output, latency, errors |
| **Memory Browser** | Current short-term buffer, recent long-term writes, episodic log |

### Technology

- `axum` for routing and handlers
- Server-side rendered HTML (templates, no SPA framework)
- Minimal CSS for readability
- Pulls data from the same Postgres instance and trace store

> **Note:** The web UI is a post-MVP stretch goal. For MVP, Jaeger traces and CLI output are sufficient.

---

## Local Development Stack

```yaml
# docker-compose.yml additions for observability
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"   # Jaeger UI
      - "4317:4317"     # OTLP gRPC receiver
    environment:
      COLLECTOR_OTLP_ENABLED: "true"
```

Traces appear at `http://localhost:16686` after starting the stack.

---

## What You'll Learn

Building this layer teaches:

- **OpenTelemetry in Rust** — spans, exporters, context propagation
- **Structured logging vs tracing** — when to log, when to trace
- **Building introspectable systems** — designing for debuggability
- **Cost accounting for LLM calls** — tracking and controlling spend
- **Production AI monitoring** — the metrics that matter for agent systems

---

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Integration tests | Run agent task → assert trace contains expected span types | `tests/` |
| Integration tests | Verify Prometheus metrics increment correctly | `tests/` |
| Manual tests | Inspect Jaeger UI for trace completeness | Local dev |
