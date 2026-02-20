# Layer 4 — Reliability & Fault Tolerance

Agents fail in slow, weird ways — an LLM returns malformed JSON, a tool hangs indefinitely, the agent loops forever calling the same tool. Your runtime needs to handle **all of this** gracefully. This layer is the difference between a demo and a production system.

**Related tasks:** L4-01 through L4-08 in [AGENTS.md](../AGENTS.md)

---

## Failure Modes in Agent Systems

| Failure Mode | Root Cause | Impact Without Mitigation |
|--------------|-----------|--------------------------|
| **Malformed output** | LLM returns unparseable JSON or wrong schema | Tool dispatch crashes, execution halts |
| **Tool hang** | External API never responds | Agent blocks forever, resources leak |
| **Infinite loop** | Agent repeats the same tool call endlessly | Token spend explodes, no progress |
| **Cascading failure** | One broken tool causes every plan to fail | Agent becomes unusable |
| **Silent degradation** | Tool returns wrong data that *looks* valid | Agent reasons on bad information |

Xola addresses each of these with a dedicated mechanism.

---

## Circuit Breaker

Prevents cascading failures by stopping calls to a tool that has failed repeatedly.

### State Machine

```
           success
    ┌───────────────────┐
    │                   ▼
┌───────┐  failure   ┌───────┐  timeout   ┌───────────┐
│CLOSED │──────────▶ │ OPEN  │───────────▶│ HALF-OPEN │
│       │ (N times)  │       │ (wait T)   │           │
└───────┘            └───────┘            └─────┬─────┘
    ▲                    ▲                      │
    │                    │    failure            │
    │                    └──────────────────────┘
    │                         success
    └─────────────────────────────────┘
```

| State | Behavior |
|-------|----------|
| **Closed** | Normal operation. Failures are counted. |
| **Open** | All calls rejected immediately. Entered after N consecutive failures. |
| **Half-Open** | One probe call allowed after a cooldown period. Success → Closed. Failure → Open. |

### Interface

```rust
pub struct CircuitBreaker {
    state: AtomicU8,  // Closed=0, Open=1, HalfOpen=2
    failure_count: AtomicUsize,
    last_failure: AtomicU64,  // timestamp
    config: CircuitBreakerConfig,
}

pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,  // N failures before opening
    pub recovery_timeout: Duration, // Wait before half-open probe
}

impl CircuitBreaker {
    pub fn can_execute(&self) -> bool;
    pub fn record_success(&self);
    pub fn record_failure(&self);
}
```

Each tool in the registry gets its own circuit breaker instance.

---

## Structured Output Enforcement

LLMs don't always return valid JSON. The `/parse` endpoint validates LLM output against a Pydantic schema and retries with a corrective prompt on failure.

### Flow

```
LLM Output → POST /parse → Pydantic validates
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 Success             Failure
                    │                   │
              Return parsed        Retry with corrective prompt
                                        │
                                   "Your output was malformed.
                                    The error was: <validation_error>
                                    Expected schema: <schema>
                                    Please try again."
                                        │
                                   Attempt 2, 3, ... N
                                        │
                                   After N failures → escalate
```

### Configuration

```toml
[parse]
max_retries = 3
include_error_in_prompt = true
include_schema_in_prompt = true
```

---

## Loop Detector

Detects when the agent is stuck calling the same tool with the same (or similar) inputs repeatedly.

### How It Works

- Maintain a sliding window of the last K tool calls (default: K=10)
- Hash each call as `hash(tool_name + canonical(input))`
- If a hash appears more than M times in the window (default: M=3), trigger abort

```rust
pub struct LoopDetector {
    window: VecDeque<u64>,  // Hashes of recent tool calls
    window_size: usize,
    repeat_threshold: usize,
}

impl LoopDetector {
    pub fn record(&mut self, tool_name: &str, input: &Value);
    pub fn is_looping(&self) -> bool;
}
```

### On Loop Detection

1. Current execution is aborted
2. Error is classified as `FailureType::Loop`
3. Agent receives a message: `"Execution aborted: detected repeated tool calls (web_search called 3 times with identical input)"`
4. If replanning is available, the agent gets one chance to replan with this context

---

## Hierarchical Timeouts

Every execution level has a timeout, enforced through a `CancellationToken` tree:

```
Task Timeout (e.g., 5 minutes)
├── Step Timeout (e.g., 60 seconds)
│   ├── Tool Call Timeout (e.g., 30 seconds)
│   ├── Tool Call Timeout
│   └── ...
├── Step Timeout
│   ├── Tool Call Timeout
│   └── ...
└── ...
```

### Rules

- **Tool timeout** — individual tool execution. Configured per tool.
- **Step timeout** — a single planning iteration (reason + tool call + observe). Default: 60s.
- **Task timeout** — the entire goal from start to final answer. Default: 5 minutes.
- **Cancellation propagates downward** — if the task deadline hits, all active steps and tool calls are cancelled.
- **All timeouts flow through `context.Context`** — Rust's `CancellationToken` from `tokio-util`.

### Implementation

```rust
let task_token = CancellationToken::new();
let task_timeout = tokio::time::sleep(task_deadline);

tokio::select! {
    result = execute_plan(task_token.child_token()) => result,
    _ = task_timeout => {
        task_token.cancel();
        Err(TaskError::GlobalDeadlineExceeded)
    }
}
```

---

## Failure Taxonomy

Every error in the system is classified into a structured taxonomy:

| Type | Code | Description | Recovery |
|------|------|-------------|----------|
| `Timeout` | `TOOL_TIMEOUT` | Tool call exceeded its deadline | Retry or replan |
| `Timeout` | `STEP_TIMEOUT` | Planning step exceeded deadline | Replan |
| `Timeout` | `TASK_TIMEOUT` | Global task deadline exceeded | Abort |
| `ParseError` | `MALFORMED_OUTPUT` | LLM returned unparseable output | Corrective re-prompt |
| `ParseError` | `SCHEMA_MISMATCH` | Output doesn't match expected schema | Corrective re-prompt |
| `ToolError` | `EXECUTION_FAILED` | Tool returned an error | Replan |
| `ToolError` | `CIRCUIT_OPEN` | Tool's circuit breaker is open | Skip tool or replan |
| `LoopError` | `CYCLE_DETECTED` | Agent is repeating tool calls | Abort or replan |

This taxonomy is used in traces, metrics, and error messages to the LLM.

---

## What You'll Learn

Building this layer teaches:

- **Circuit breaker pattern** — protecting systems from cascading failures
- **Defensive LLM output parsing** — never trusting raw LLM output
- **Timeout hierarchies with context** — structured cancellation across async call chains
- **Failure taxonomies** — classifying errors so you can handle them programmatically

---

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Unit tests | Circuit breaker state transitions | `runtime/src/reliability/` |
| Unit tests | Loop detector with synthetic sequences | `runtime/src/reliability/` |
| Unit tests | Corrective re-prompt generation | `llm_surface/` |
| Integration tests | Inject tool failure → verify replan fires | `tests/` |
| Integration tests | Force timeout → verify cancellation propagates | `tests/` |
