# Layer 1 — Tool Registry & Execution Sandbox

Tools are functions the agent can call — search the web, run code, query a database, call an API. The runtime needs a clean way to register tools, validate inputs against a schema, execute them safely, and return structured results back to the LLM.

**Related tasks:** L1-01 through L1-09 in [AGENTS.md](../AGENTS.md)

---

## Concepts

### What Is a Tool?

A tool is any capability the agent can invoke. Each tool has:

1. **A name** — unique identifier used by the LLM to select it (e.g., `web_search`)
2. **A JSON Schema descriptor** — defines the expected input shape, validated before execution
3. **An execute function** — takes validated input, performs work, returns structured output
4. **Tracing metadata** — every invocation records inputs, outputs, latency, and errors

### Why Validation Matters

LLMs produce approximate output. An agent might call `web_search` with `{"query": 42}` instead of `{"query": "string"}`. Without input validation, this passes silently to the tool and fails in unpredictable ways. Schema validation catches this at the boundary — before execution begins — and produces a clear error the LLM can reason about.

---

## Architecture

```
                     ┌────────────────────┐
                     │    Tool Registry    │
                     │  HashMap<String,    │
                     │   Arc<dyn Tool>>    │
                     └─────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
      ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
      │  UrlFetch   │  │  WebSearch  │  │  CodeExec   │
      │  Tool       │  │  Tool       │  │  Tool       │
      │  (reqwest)  │  │  (Serper)   │  │  (Docker)   │
      └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
             │                 │                 │
      ┌──────▼─────────────────▼─────────────────▼──────┐
      │              Execution Sandbox                    │
      │  • JSON Schema validation                         │
      │  • context.Context timeout                        │
      │  • Goroutine-per-tool execution                   │
      │  • Trace recording (input, output, latency, err)  │
      └──────────────────────────────────────────────────┘
```

---

## Key Interfaces

### The `Tool` Trait

```rust
pub trait Tool: Send + Sync {
    /// Unique name for this tool (used in LLM tool selection)
    fn name(&self) -> &str;

    /// Human-readable description (included in LLM prompt)
    fn description(&self) -> &str;

    /// JSON Schema defining expected input format
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with validated input
    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError>;
}
```

### The `ToolRegistry`

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Arc<dyn Tool>);
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn list_schemas(&self) -> Vec<serde_json::Value>;
    
    /// Validate input, execute with timeout, record trace
    pub async fn dispatch(
        &self,
        name: &str,
        input: serde_json::Value,
        timeout: Duration,
    ) -> Result<ToolResult, DispatchError>;
}
```

---

## Tool Implementations

### `UrlFetchTool`

Fetches the text content of a URL using `reqwest`.

| Field | Value |
|-------|-------|
| **Input** | `{ "url": "string" }` |
| **Output** | `{ "content": "string", "status_code": 200, "content_length": 12345 }` |
| **Timeout** | Configurable, default 30s |
| **Error modes** | Network error, timeout, non-2xx status |

### `WebSearchTool`

Calls an external search API (Serper or Brave) and returns structured results.

| Field | Value |
|-------|-------|
| **Input** | `{ "query": "string", "num_results": 5 }` |
| **Output** | `{ "results": [{ "title": "...", "url": "...", "snippet": "..." }] }` |
| **Timeout** | Configurable, default 15s |
| **Error modes** | API error, rate limit, timeout |

### `CodeExecTool`

Executes code in a sandboxed Docker container using the Go Docker SDK.

| Field | Value |
|-------|-------|
| **Input** | `{ "language": "python", "code": "string", "timeout_seconds": 30 }` |
| **Output** | `{ "stdout": "string", "stderr": "string", "exit_code": 0 }` |
| **Timeout** | User-specified + hard cap |
| **Error modes** | Container creation failure, execution timeout, OOM kill |
| **Security** | No network access, memory limit, read-only filesystem |

---

## Execution Model

Every tool call follows this sequence:

1. **Lookup** — Registry resolves tool name to `Arc<dyn Tool>`
2. **Validate** — Input checked against the tool's JSON Schema; reject with error if invalid
3. **Dispatch** — Tool executes in a spawned task with `tokio::time::timeout`
4. **Trace** — Record `{ tool_name, input, output, latency_ms, error }` on every call
5. **Return** — Structured result sent back to the planning layer

```rust
// Simplified dispatch flow
let tool = registry.get(action_name)?;
let validated = validate_input(&tool.input_schema(), &input)?;

let result = tokio::time::timeout(
    tool_timeout,
    tool.execute(validated)
).await??;

tracer.record_tool_call(tool.name(), &input, &result, elapsed);
```

---

## What You'll Learn

Building this layer teaches:

- **Interface design in Rust** — trait objects, dynamic dispatch, `Arc<dyn Trait>`
- **JSON Schema validation** — runtime type checking for LLM-generated inputs
- **Context propagation** — `context.Context` timeouts flowing through async call chains
- **Container sandboxing** — the Docker SDK, image management, resource limits
- **Structured concurrency** — spawned tasks that are always accounted for

---

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Unit tests | Each tool with mocked external calls | `runtime/src/tools/` |
| Schema tests | Validation rejects bad inputs, accepts good ones | `runtime/src/tools/` |
| Integration tests | Each tool end-to-end against real services | `tests/` (marked `#[ignore]`) |

Every new tool must have: a JSON Schema descriptor, a unit test with mocked output, and an integration test against the real service.
