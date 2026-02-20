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
    use tools::{mock::MockTool, ToolRegistry};

    println!("Xola runtime initializing...");
    println!("L1-01: Tool trait defined ✓");
    println!("L1-02: Tool registry implemented ✓");
    println!("L1-03: JSON Schema validation implemented ✓");

    // Initialize registry and register tools
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(MockTool))
        .expect("Failed to register mock tool");

    println!("\nRegistered tools: {:?}", registry.list_names());

    // Demonstrate tool lookup
    let tool = registry.get("mock_echo").expect("Tool not found");
    println!("Loaded tool: {} - {}", tool.name(), tool.description());

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

    // Demonstrate tool execution with validated input
    println!("\nTool execution test:");
    match tool.execute(valid_input).await {
        Ok(result) => println!("  Result: {:?}", result),
        Err(e) => eprintln!("  Execution failed: {}", e),
    }

    // Placeholder - will be expanded in future tasks:
    // - L1-04: Dispatcher with timeout (orchestrates validate + execute)
    // - L2-03+: Memory subsystem
    // - L3-01+: Planning layer
    // - L4-01+: Reliability layer
    // - L5-01+: Observability

    // Future: Wrap registry in Arc for sharing across tasks
    // let shared_registry = Arc::new(registry);
}
