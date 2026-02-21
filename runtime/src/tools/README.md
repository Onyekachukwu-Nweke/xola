# Tool Registry

This directory contains the `Tool` trait definition and all tool implementations.

## Current Tools

| Tool Name | Status | Description |
|-----------|--------|-------------|
| `mock_echo` | ✓ Implemented | Test tool that echoes input (L1-01) |
| `url_fetch` | ✓ Implemented | Fetches URL content via reqwest (L1-05) |
| `web_search` | ✓ Implemented | Searches via Serper API (L1-06) |
| `code_exec` | ⏳ Planned | Executes code in Docker sandbox (L1-07) |

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
