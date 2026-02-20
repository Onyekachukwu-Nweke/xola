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
    println!("Xola runtime initializing...");
    println!("L1-01: Tool trait defined ✓");

    // Placeholder - will be expanded in future tasks:
    // - L1-02: ToolRegistry
    // - L1-04: Dispatcher with timeout
    // - L2-03+: Memory subsystem
    // - L3-01+: Planning layer
    // - L4-01+: Reliability layer
    // - L5-01+: Observability

    // For now, just demonstrate the tool trait works
    use serde_json::json;
    use tools::{mock::MockTool, Tool};

    let tool = MockTool;
    println!("Loaded tool: {} - {}", tool.name(), tool.description());

    let input = json!({ "message": "Runtime initialized" });
    match tool.execute(input).await {
        Ok(result) => println!("Tool execution test: {:?}", result),
        Err(e) => eprintln!("Tool execution failed: {}", e),
    }
}
