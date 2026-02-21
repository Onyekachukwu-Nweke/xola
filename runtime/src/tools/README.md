# Tool Registry

This directory contains the `Tool` trait definition and all tool implementations.

## Current Tools

| Tool Name | Status | Description |
|-----------|--------|-------------|
| `mock_echo` | ✓ Implemented | Test tool that echoes input (L1-01) |
| `url_fetch` | ✓ Implemented | Fetches URL content via reqwest (L1-05) |
| `web_search` | ✓ Implemented | Searches via Serper API (L1-06) |
| `code_exec` | ✓ Implemented | Executes code in Docker sandbox (L1-07) |

## Tool Registry (L1-02)

The `ToolRegistry` manages all available tools. It provides:
- **Registration**: Add tools during startup with duplicate name checking
- **Lookup**: Retrieve tools by name for execution
- **Schema listing**: Generate JSON schemas for LLM tool selection

### Usage

```rust
use std::sync::Arc;
use xola_runtime::tools::{ToolRegistry, mock::MockTool};

// Create registry
let mut registry = ToolRegistry::new();

// Register tools
registry.register(Arc::new(MockTool))
    .expect("Duplicate tool name");

// Look up a tool
let tool = registry.get("mock_echo").unwrap();

// Get schemas for LLM prompt
let schemas = registry.list_schemas();
```

### Schema Format

Schemas returned by `list_schemas()` match the IPC `/reason` endpoint format:
```json
[
  {
    "name": "mock_echo",
    "description": "A mock tool that echoes back the input message",
    "parameters": {
      "type": "object",
      "properties": {
        "message": { "type": "string" }
      },
      "required": ["message"]
    }
  }
]
```

This format is sent to the Python LLM surface for tool selection.

### Registry Error Handling

The registry returns `Result<(), RegistryError>` from `register()`:
- `RegistryError::DuplicateName(String)` - Tool name already registered

This prevents silent overwrites and helps catch configuration errors early.

## JSON Schema Validation (L1-03)

All tool inputs are validated against their JSON Schemas before execution using the `jsonschema` crate.

### Validation Flow

```
Tool Input → InputValidator.validate() → Tool.execute()
             ↓ (if invalid)
         ToolError::InvalidInput
```

### Usage

```rust
use xola_runtime::tools::{ToolRegistry, InputValidator};
use serde_json::json;

let mut registry = ToolRegistry::new();
// ... register tools ...

// Validate via registry (convenience method)
let input = json!({ "query": "search term" });
registry.validate_input("web_search", &input)?;

// Or validate directly with a schema
let schema = json!({
    "type": "object",
    "properties": {
        "query": { "type": "string" }
    },
    "required": ["query"]
});
InputValidator::validate(&schema, &input)?;
```

### Error Handling

Validation failures return `ToolError::InvalidInput` with detailed error messages:
```
Input validation failed: "message" is a required property
```

These messages help the LLM understand what went wrong and self-correct in the next attempt.

### Testing Validation

Tools should have tests for both valid and invalid inputs:
```rust
#[test]
fn test_valid_input() {
    let schema = tool.input_schema();
    let input = json!({ /* valid input */ });
    assert!(InputValidator::validate(&schema, &input).is_ok());
}

#[test]
fn test_invalid_input() {
    let schema = tool.input_schema();
    let input = json!({ /* invalid input */ });
    assert!(InputValidator::validate(&schema, &input).is_err());
}
```

## Per-Tool Timeouts (L1-04)

All tool executions are wrapped with `tokio::time::timeout` to prevent tools from blocking indefinitely.

### Timeout Flow

```
Tool Call → validate_input() → execute_with_timeout()
                                  ↓
                         tokio::time::timeout(duration)
                                  ↓
                            tool.execute(input)
                                  ↓ (if exceeds timeout)
                            ToolError::Timeout
```

### Timeout Configuration

```rust
use xola_runtime::tools::{ToolRegistry, ToolTimeoutConfig};
use std::time::Duration;

// Create config with defaults (60s for all tools)
let mut timeout_config = ToolTimeoutConfig::default();

// Set per-tool overrides
timeout_config.set_timeout("web_search", 15_000);   // 15 seconds
timeout_config.set_timeout("url_fetch", 30_000);    // 30 seconds
timeout_config.set_timeout("code_exec", 300_000);   // 5 minutes

// Execute with timeout
let timeout = timeout_config.get_timeout("web_search");
let result = registry
    .execute_with_timeout("web_search", input, timeout)
    .await?;
```

### Environment Variables

Load timeout configuration from environment variables:

```bash
export XOLA_TOOL_TIMEOUT_MS=60000  # Default: 60 seconds
```

```rust
let timeout_config = ToolTimeoutConfig::from_env();
```

### Error Handling

Timeout failures return `ToolError::Timeout` with the exceeded duration:
```
Timeout after 60000ms
```

The LLM can use this signal to:
- Retry with a simpler query
- Switch to a different tool
- Report timeout to the user

### Testing Timeouts

Tools should verify they complete within reasonable timeouts:
```rust
#[tokio::test]
async fn test_tool_completes_within_timeout() {
    let timeout = Duration::from_secs(5);
    let result = registry
        .execute_with_timeout("mock_echo", input, timeout)
        .await;

    assert!(result.is_ok());
}
```

## CodeExecTool Special Requirements

The `code_exec` tool has additional prerequisites beyond standard tools due to its Docker dependency.

### Docker Daemon Required

CodeExecTool requires a running Docker daemon accessible via the default socket:
- **Linux**: `/var/run/docker.sock`
- **macOS**: Docker Desktop must be running
- **Windows**: Docker Desktop with WSL2 backend

The user running the runtime must have permission to access Docker (typically via the `docker` group on Linux).

### Pre-pull Docker Images

**Before first use**, pull required images to avoid timeout failures:

```bash
docker pull python:3.12-alpine
docker pull node:22-alpine
docker pull alpine:3.21
```

The tool does NOT auto-pull images to avoid unpredictable latency (image pulls can take 30+ seconds).

If an image is missing, you'll see an error like:
```
Failed to create container: No such image: python:3.12-alpine
```

### Resource Limits

Every code execution runs with strict resource constraints:

| Limit | Value | Reason |
|-------|-------|--------|
| **Memory** | 128 MB | Prevents runaway allocations (OOM kill at ~128 MB) |
| **Network** | Disabled | Code cannot make HTTP requests or access external services |
| **Filesystem** | Read-only | Code cannot persist files between runs |
| **Processes** | 64 max | Prevents fork bombs |

### Timeout Behavior

- **User-specified**: The `timeout_seconds` parameter (1-30s, default 10s) applies to code execution
- **Registry timeout**: Acts as a hard cap (typically 60s by default)
- **Timeout error**: Returns `ToolError::ExecutionFailed` with "exceeded timeout" message

### Exit Codes

CodeExecTool returns different exit codes to help the LLM understand what happened:

| Exit Code | Meaning |
|-----------|---------|
| `0` | Success |
| `1-255` | Language runtime error (e.g., Python exception, Node.js crash, bash error) |
| `137` | **Memory limit exceeded (OOM kill)** - Returns error instead of normal result |
| `-1` | Container status unknown (rare Docker API failure) |

When code hits the 128 MB memory limit, Docker sends SIGKILL (exit code 137). CodeExecTool detects this and returns a clear error message: `"Code execution exceeded 128 MB memory limit (OOM kill)"`.

### Supported Languages

| Language | Image | Execution Command |
|----------|-------|-------------------|
| `python` | `python:3.12-alpine` | `python3 -c "<code>"` |
| `javascript` | `node:22-alpine` | `node -e "<code>"` |
| `bash` | `alpine:3.21` | `sh -c "<code>"` |

Requesting an unsupported language returns:
```
Unsupported language 'ruby'. Supported: python, javascript, bash
```

### Example Usage

```rust
use xola_runtime::tools::{CodeExecTool, Tool};
use serde_json::json;

let tool = CodeExecTool;
let input = json!({
    "language": "python",
    "code": "print('Hello from sandbox!')",
    "timeout_seconds": 5
});

let result = tool.execute(input).await?;
assert_eq!(result["exit_code"], 0);
assert!(result["stdout"].as_str().unwrap().contains("Hello from sandbox!"));
```

### Security Guarantees

- **Network isolation**: Containers cannot reach external networks (both `network_disabled: true` and `network_mode: "none"`)
- **No host access**: No bind mounts giving access to host paths
- **Memory cap**: Hard 128 MB limit enforced by Docker
- **Process limits**: Max 64 processes to prevent fork bombs
- **Read-only root**: Code cannot modify the container filesystem
- **Ephemeral containers**: Every execution creates a fresh container, automatically cleaned up afterward

## Adding a New Tool

See [docs/contributing.md](../../docs/contributing.md#adding-a-new-tool) for the full checklist.

Quick version:
1. Implement `Tool` trait in a new file under `src/tools/`
2. Add JSON Schema descriptor in `input_schema()`
3. Write unit tests with mocked external dependencies
4. Write integration test in `tests/` (mark with `#[ignore]`)
5. Register in `ToolRegistry` (L1-02)
6. Update this README

## Tool Trait

Defined in `mod.rs`. Every tool must:
- Have a unique `name()` in snake_case
- Provide a clear `description()` for the LLM
- Define `input_schema()` as a JSON Schema object
- Implement async `execute(&self, input: Value) -> Result<Value, ToolError>`

All tools must be `Send + Sync` because they're stored in `Arc<dyn Tool>`.

## Error Handling

Use `ToolError` variants:
- `InvalidInput`: Schema validation failed (handled by dispatcher in L1-03)
- `ExecutionFailed`: Tool-specific error (network, API, etc.)
- `Timeout`: Set by dispatcher, not by tool implementation
- `JsonError`: Serialization failure
- `NotFound`: Registry lookup failed

## Testing Strategy

- **Unit tests**: Mock external dependencies, test tool logic in isolation
- **Integration tests**: Call real services, mark with `#[ignore]`
- **Schema tests**: Validate that good inputs pass, bad inputs fail

See `mock.rs` for test examples.
