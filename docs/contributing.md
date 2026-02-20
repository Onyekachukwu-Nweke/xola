# Contributing

This document covers development conventions, the agent role system, task workflow, and rules for working in the Xola codebase.

---

## Agent Roles

Xola uses three defined agent roles. If you're an autonomous coding agent, identify which role you're operating in and restrict yourself accordingly.

### `RUNTIME_AGENT`

| | |
|---|---|
| **Scope** | `runtime/` Rust crate only |
| **Can** | Implement/modify `runtime/src/`, add Cargo deps, write unit tests, run `cargo test`/`clippy`/`fmt`, add sqlx migrations, **read** `llm_surface/` for reference |
| **Cannot** | Modify `llm_surface/`, change IPC contract, add Python code, call LLM APIs from Rust, remove observability instrumentation |
| **Escalate** | IPC schema changes, new external service deps, circuit breaker threshold tuning, `unsafe` blocks |

### `LLM_SURFACE_AGENT`

| | |
|---|---|
| **Scope** | `llm_surface/` Python package only |
| **Can** | Implement/modify `llm_surface/`, add Python deps with `uv add`, write pytest tests, run `ruff`/`mypy`, modify FastAPI routes (not paths/schemas without PROPOSAL.md) |
| **Cannot** | Modify `runtime/`, add state to Python server, add new LLM providers without approval, hardcode prompts (use `config/prompts.toml`) |
| **Escalate** | ReAct prompt structural changes, new LLM providers, token budget logic, IPC schema changes |

### `INTEGRATION_AGENT`

| | |
|---|---|
| **Scope** | `tests/`, `docker/`, `config/`, `migrations/`. Read access to both `runtime/` and `llm_surface/` |
| **Can** | Write integration tests, modify Docker Compose, add config files, add migrations (additive only) |
| **Cannot** | Modify source code in `runtime/` or `llm_surface/`, run ignored tests in CI without instruction, destructive schema changes |
| **Escalate** | Cross-process contract mismatches, data-transforming migrations, new Docker services |

---

## Task Workflow

### Before Starting

1. Read `CLAUDE.md` in full
2. Run the existing test suite and confirm it passes — don't start on a red suite
3. Identify your task ID (from [AGENTS.md](../AGENTS.md)) and state it in your first commit message

### While Working

- **Commit at logical checkpoints** — don't accumulate an entire layer in one commit
- **Stay in your lane** — if a task requires code outside your role's scope, write a `PROPOSAL.md` instead
- **Don't delete others' tests** — if a test you didn't write fails, investigate before proceeding
- **Use proper logging** — `tracing::debug!` in Rust, `logging` module in Python. No `println!` / `print()` in committed code.

### Before Marking Done

- [ ] All tests pass (`cargo test`, `pytest`)
- [ ] Lint passes (`cargo clippy -- -D warnings`, `ruff check`, `mypy --strict`)
- [ ] The task's integration test passes (if listed)
- [ ] No secrets in source
- [ ] No `unwrap()` on LLM output paths
- [ ] No stateful Python server handlers

---

## Things That Always Require Human Approval

These actions are **never** permitted without explicit human sign-off:

| Action | Reason |
|--------|--------|
| Change the IPC contract | Cross-process breaking change |
| Add a new external API dependency | Cost and security implication |
| Modify database schema destructively | Data loss risk |
| Add `unsafe` code in Rust | Memory safety risk |
| Change token budget or cost accounting | Financial implication |
| Modify the ReAct prompt template structurally | Affects all agent behavior |

To propose any of these, create a `PROPOSAL.md` in the project root describing the change, its rationale, and its impact.

---

## Code Conventions

### Rust

| Rule | Detail |
|------|--------|
| Lint | `cargo clippy -- -D warnings` must pass |
| Format | `cargo fmt` mandatory |
| Error handling | `Result<T, E>` with `thiserror` — no `unwrap()` in prod paths |
| Concurrency | Every `tokio::spawn` has its `JoinHandle` accounted for |
| Timeouts | Every external call has a timeout — nothing blocks forever |

### Python

| Rule | Detail |
|------|--------|
| Lint | `ruff check` and `ruff format` before commit |
| Types | `mypy --strict` on the `llm_surface` package |
| API boundary | All IPC handlers typed with Pydantic models — no raw dicts |
| Error handling | Never swallow exceptions silently — always log raw LLM failures |
| Deps | Use `uv` for all dependency management — no `pip` |

### Both

| Rule | Detail |
|------|--------|
| Secrets | Environment variables only — nothing in source |
| New tools | Must have: JSON Schema descriptor + unit test + integration test |
| Config | Configurable values go in `config/` TOML files |

---

## Adding a New Tool

1. Implement the `Tool` trait in `runtime/src/tools/`
2. Add a JSON Schema descriptor
3. Write a unit test with mocked external output
4. Write an integration test against the real service (mark with `#[ignore]`)
5. Register the tool in the `ToolRegistry`
6. Update `runtime/src/tools/README.md`
7. Add the tool to the CLI help text
