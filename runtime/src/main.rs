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

    // Demonstrate tool execution
    let input = json!({ "message": "Registry working!" });
    match tool.execute(input).await {
        Ok(result) => println!("\nTool execution test: {:?}", result),
        Err(e) => eprintln!("\nTool execution failed: {}", e),
    }

    // Placeholder - will be expanded in future tasks:
    // - L1-03: JSON Schema validation
    // - L1-04: Dispatcher with timeout
    // - L2-03+: Memory subsystem
    // - L3-01+: Planning layer
    // - L4-01+: Reliability layer
    // - L5-01+: Observability

    // Future: Wrap registry in Arc for sharing across tasks
    // let shared_registry = Arc::new(registry);
}
