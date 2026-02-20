# MVP Definition

This document defines what "done" looks like for the Xola minimum viable product. The MVP is a working agent that can complete a multi-step research task autonomously.

---

## The Demo Task

The MVP agent must be able to accept a goal like:

> *"Find the three most cited papers on RAG published in 2024 and summarize their key contributions."*

…and autonomously plan, search, read, synthesize, and return a structured answer — with a full trace visible in Jaeger.

---

## Acceptance Criteria

All of the following must be true:

| # | Criterion | Layer | Status |
|---|-----------|-------|--------|
| 1 | Three tools registered and working: `url_fetch`, `web_search`, `code_exec` | L1 | ☐ |
| 2 | Short-term and long-term memory wired up and queried each turn | L2 | ☐ |
| 3 | ReAct loop completes a multi-step research task without human intervention | L3 | ☐ |
| 4 | Replanning fires correctly on a tool failure (verified by integration test) | L3 | ☐ |
| 5 | Every run produces a visible trace in Jaeger with tool call spans | L5 | ☐ |
| 6 | `cargo clippy -- -D warnings` and `mypy --strict` both pass on main | — | ☐ |
| 7 | README has a working quickstart that a new developer can follow in under 20 minutes | — | ☐ |

---

## What's In Scope (MVP)

| Component | What's Required |
|-----------|----------------|
| **Tool registry** | 3 tools registered, validated, dispatched with timeouts |
| **Short-term memory** | Circular buffer with token budget, eviction working |
| **Long-term memory** | pgvector store, semantic query on each turn |
| **ReAct loop** | Full Thought → Action → Observation cycle |
| **Replanning** | Fires on tool failure, bounded by max attempts |
| **Context timeouts** | Per-tool and per-task deadlines via `CancellationToken` |
| **Tracing** | OpenTelemetry spans exported to Jaeger |
| **CLI** | Clean command-line interface for submitting goals |

## What's Out of Scope (MVP)

| Component | Why Deferred |
|-----------|-------------|
| **Web UI** | Jaeger provides sufficient trace inspection for MVP |
| **Prometheus metrics** | Nice to have, not required for demo |
| **Full circuit breakers** | Timeouts provide sufficient protection for MVP |
| **Loop detector** | Replan limits cover the worst cases |
| **Episodic memory** | Long-term memory sufficient for MVP demo |
| **Cost accounting dashboard** | Token counts in traces are sufficient |

---

## Timeline

Completable in **8–10 weeks** at intermediate Rust level. Each layer builds on the last, so you're never blocked:

| Phase | Layers | Duration |
|-------|--------|----------|
| Phase 1 | L1 (Tools) | ~2 weeks |
| Phase 2 | L2 (Memory) | ~2 weeks |
| Phase 3 | L3 (Planning) | ~2 weeks |
| Phase 4 | L5 (Tracing only) | ~1 week |
| Polish | Integration tests, README, cleanup | ~1 week |

See [Roadmap](roadmap.md) for the detailed phased plan.

---

## What This Proves

Completing the MVP demonstrates:

1. **You can build infrastructure, not just call APIs** — tool dispatch, memory, planning, and tracing are all first-party code
2. **You understand why agents fail in production** — timeout hierarchies, replanning on failure, memory management
3. **You can make non-deterministic systems observable** — every run is fully traceable
4. **You can work across a multi-language, multi-process system** — Rust + Python, IPC boundary, shared state via Postgres

This is the combination that's rare and in demand.
