//! Xola Agent Runtime Binary
//!
//! A two-process AI agent runtime built in Rust and Python.
//! This is the Rust side - handling tool dispatch, memory, planning,
//! fault tolerance, and observability.
//!
//! See CLAUDE.md and docs/ for architecture and design rationale.

use xola_runtime::config::{self, ObservabilityConfig};
use xola_runtime::observe;
use xola_runtime::tools;

#[tokio::main]
async fn main() {
    // Load .env for local development. In production, environment variables
    // are injected directly (Docker --env-file, systemd, Kubernetes secrets),
    // so .ok() ensures this is a no-op when no .env file is present.
    dotenvy::dotenv().ok();

    // Load configuration from TOML file if available.
    let obs_config = match config::load_config("config/local.toml") {
        Ok(cfg) => cfg.observability,
        Err(e) => {
            eprintln!("Warning: could not load config ({e}), using defaults");
            ObservabilityConfig::default()
        }
    };

    // Initialise observability (tracing + metrics).
    // The guard must be held for the lifetime of the process to ensure
    // spans and metrics are flushed on shutdown.
    let obs = observe::init_observability(&obs_config).expect("failed to initialise observability");
    let _guard = obs.guard;

    // Spawn metrics server if enabled.
    if let Some(router) = obs.metrics_router {
        let port = obs_config.metrics_port;
        tokio::spawn(async move {
            let addr = format!("0.0.0.0:{port}");
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .unwrap_or_else(|e| panic!("Failed to bind metrics server on {addr}: {e}"));
            tracing::info!(port, "Metrics server listening");
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!(error = %e, "Metrics server exited with error");
            }
        });
    }

    use serde_json::json;
    use std::sync::Arc;
    use tools::{
        mock::MockTool, CodeExecTool, ToolRegistry, ToolTimeoutConfig, UrlFetchTool, WebSearchTool,
    };

    tracing::info!("Xola runtime initializing");

    // Initialize registry and register tools
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("Failed to register mock tool");
    registry
        .register(Arc::new(UrlFetchTool))
        .expect("Failed to register url_fetch tool");
    registry
        .register(Arc::new(WebSearchTool))
        .expect("Failed to register web_search tool");
    registry
        .register(Arc::new(CodeExecTool))
        .expect("Failed to register code_exec tool");

    tracing::info!(tools = ?registry.list_names(), "Tools registered");

    // Initialize timeout configuration
    let timeout_config = ToolTimeoutConfig::default();
    tracing::debug!(
        default_timeout_ms = timeout_config.default_timeout_ms,
        "Timeout configuration loaded"
    );

    // Demonstrate tool lookup
    let tool = registry.get("mock_echo").expect("Tool not found");
    tracing::debug!(tool = tool.name(), "Loaded tool");

    // Demonstrate schema generation for LLM
    let schemas = registry.list_schemas();
    tracing::debug!(count = schemas.len(), "Tool schemas generated for LLM");

    // Demonstrate validation with valid input
    let valid_input = json!({ "message": "Registry working!" });
    match registry.validate_input("mock_echo", &valid_input) {
        Ok(()) => tracing::debug!("Valid input accepted"),
        Err(e) => tracing::error!(error = %e, "Unexpected validation error"),
    }

    // Demonstrate tool execution with timeout
    let timeout = timeout_config.get_timeout("mock_echo");
    match registry
        .execute_with_timeout("mock_echo", valid_input, timeout)
        .await
    {
        Ok(result) => tracing::info!(result = %result, "Tool execution completed"),
        Err(e) => tracing::error!(error = %e, "Tool execution failed"),
    }

    tracing::info!("Xola runtime ready");
}
