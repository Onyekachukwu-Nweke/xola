//! Xola Agent Runtime
//!
//! A two-process AI agent runtime built in Rust and Python.
//! This is the Rust side - handling tool dispatch, memory, planning,
//! fault tolerance, and observability.
//!
//! See CLAUDE.md and docs/ for architecture and design rationale.

mod tools;

#[tokio::main]
async fn main() {
    use serde_json::json;
    use std::sync::Arc;
    use tools::{mock::MockTool, ToolRegistry, ToolTimeoutConfig};

    println!("Xola runtime initializing...");
    println!("L1-01: Tool trait defined ✓");
    println!("L1-02: Tool registry implemented ✓");
    println!("L1-03: JSON Schema validation implemented ✓");
    println!("L1-04: Per-tool timeout implemented ✓");

    // Initialize registry and register tools
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(MockTool))
        .expect("Failed to register mock tool");

    println!("\nRegistered tools: {:?}", registry.list_names());

    // Initialize timeout configuration
    let timeout_config = ToolTimeoutConfig::default();
    println!("\nTimeout configuration:");
    println!("  Default timeout: {}ms", timeout_config.default_timeout_ms);

    // Demonstrate tool lookup
    let tool = registry.get("mock_echo").expect("Tool not found");
    println!("\nLoaded tool: {} - {}", tool.name(), tool.description());

    // Demonstrate schema generation for LLM
    let schemas = registry.list_schemas();
    println!("\nTool schemas for LLM:");
    for schema in &schemas {
        println!("  - {}: {}", schema["name"], schema["description"]);
    }

    // Demonstrate validation with valid input
    println!("\nValidation tests:");
    let valid_input = json!({ "message": "Registry working!" });
    match registry.validate_input("mock_echo", &valid_input) {
        Ok(()) => println!("  ✓ Valid input accepted"),
        Err(e) => eprintln!("  ✗ Unexpected error: {}", e),
    }

    // Demonstrate validation with invalid input
    let invalid_input = json!({ "wrong_field": "oops" });
    match registry.validate_input("mock_echo", &invalid_input) {
        Ok(()) => eprintln!("  ✗ Invalid input was accepted (should have failed)"),
        Err(e) => println!("  ✓ Invalid input rejected: {}", e),
    }

    // Demonstrate tool execution with timeout
    println!("\nTool execution with timeout:");
    let timeout = timeout_config.get_timeout("mock_echo");
    match registry
        .execute_with_timeout("mock_echo", valid_input, timeout)
        .await
    {
        Ok(result) => println!("  ✓ Execution completed: {:?}", result),
        Err(e) => eprintln!("  ✗ Execution failed: {}", e),
    }

    // Placeholder - will be expanded in future tasks:
    // - L1-05-07: Real tools (url_fetch, web_search, code_exec)
    // - L1-08: Execution trace logging
    // - L2-03+: Memory subsystem
    // - L3-01+: Planning layer
    // - L4-01+: Reliability layer (circuit breakers)
    // - L5-01+: Observability (tracing spans)

    // Future: Wrap registry in Arc for sharing across tasks
    // let shared_registry = Arc::new(registry);
}
