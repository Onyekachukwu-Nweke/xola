# Tool Registry

This directory contains the `Tool` trait definition and all tool implementations.

## Current Tools

| Tool Name | Status | Description |
|-----------|--------|-------------|
| `mock_echo` | ✓ Implemented | Test tool that echoes input (L1-01) |
| `url_fetch` | ⏳ Planned | Fetches URL content via reqwest (L1-05) |
| `web_search` | ⏳ Planned | Searches via Serper/Brave API (L1-06) |
| `code_exec` | ⏳ Planned | Executes code in Docker sandbox (L1-07) |

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
