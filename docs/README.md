# Xola — Documentation

> **Xola** is an AI Agent Runtime engine — the infrastructure layer between a raw LLM and a reliable autonomous system. It handles tool dispatch, memory, planning, fault tolerance, and observability so the LLM can focus on reasoning.

---

## Table of Contents

### Overview

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System architecture, the five-layer model, and process boundary |
| [Roadmap](roadmap.md) | Phased development plan with milestones and timelines |
| [MVP Definition](mvp-definition.md) | Acceptance criteria for the portfolio-ready deliverable |

### Layer Guides

| Document | Layer | Focus |
|----------|-------|-------|
| [Layer 1 — Tools](layer-1-tools.md) | Tool Registry & Execution Sandbox | Interface design, sandboxing, tracing |
| [Layer 2 — Memory](layer-2-memory.md) | Memory Architecture | Short-term, long-term, episodic memory |
| [Layer 3 — Planning](layer-3-planning.md) | Task Planning & Orchestration | ReAct loop, DAG execution, replanning |
| [Layer 4 — Reliability](layer-4-reliability.md) | Reliability & Fault Tolerance | Circuit breakers, timeouts, loop detection |
| [Layer 5 — Observability](layer-5-observability.md) | Observability & Introspection | OpenTelemetry, Prometheus, web UI |

### Reference

| Document | Description |
|----------|-------------|
| [IPC Contract](ipc-contract.md) | Canonical Rust ↔ Python API contract |
| [Getting Started](getting-started.md) | Prerequisites, local setup, first run |
| [Contributing](contributing.md) | Conventions, agent roles, task workflow |

---

## How to Read This

- **New to the project?** Start with [Architecture](architecture.md), then [Getting Started](getting-started.md).
- **Building a specific layer?** Go directly to the relevant layer guide.
- **Planning work?** Read the [Roadmap](roadmap.md) and [MVP Definition](mvp-definition.md).
- **Working on the IPC boundary?** Read [IPC Contract](ipc-contract.md) — it's the source of truth.
