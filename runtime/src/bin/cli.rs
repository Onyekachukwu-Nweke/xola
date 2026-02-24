//! Xola CLI — send a goal to the agent runtime and watch it execute.
//!
//! Usage:
//!
//! ```bash
//! cargo run --bin xola-cli -- --goal "what is the capital of France"
//! cargo run --bin xola-cli -- --goal "find recent RAG papers" --config config/local.toml
//! ```
//!
//! Prerequisites:
//!   1. Docker services running: `docker compose -f docker/docker-compose.yml up -d`
//!   2. Python IPC server: `cd llm_surface && uv run uvicorn llm_surface.server:app --host 127.0.0.1 --port 8001`

use std::env;
use std::process;
use std::sync::Arc;

use xola_runtime::config::{self, ObservabilityConfig};
use xola_runtime::ipc::IpcClient;
use xola_runtime::observe;
use xola_runtime::planning::{PlanConfig, PlanExecutor};
use xola_runtime::tools::mock::MockTool;
use xola_runtime::tools::{
    CodeExecTool, ToolRegistry, ToolTimeoutConfig, UrlFetchTool, WebSearchTool,
};

/// Parse a simple `--key value` pair from args.
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let args: Vec<String> = env::args().collect();

    let goal = match get_arg(&args, "--goal") {
        Some(g) => g,
        None => {
            eprintln!("Usage: xola-cli --goal \"your task\" [--config path/to/config.toml] [--ipc-url http://host:port]");
            process::exit(1);
        }
    };

    let config_path = get_arg(&args, "--config").unwrap_or_else(|| "config/local.toml".into());
    let ipc_url_override = get_arg(&args, "--ipc-url");

    // Load config.
    let obs_config = match config::load_config(&config_path) {
        Ok(cfg) => cfg.observability,
        Err(e) => {
            eprintln!("Warning: could not load config ({e}), using defaults");
            ObservabilityConfig::default()
        }
    };

    // Initialise observability.
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

    // Connect to the Python IPC server.
    let ipc_url = ipc_url_override.unwrap_or_else(|| "http://127.0.0.1:8001".into());
    let ipc_client = match IpcClient::new(&ipc_url) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Failed to create IPC client for {ipc_url}: {e}");
            process::exit(1);
        }
    };

    // Build tool registry.
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("Failed to register mock_echo");
    registry
        .register(Arc::new(UrlFetchTool))
        .expect("Failed to register url_fetch");
    registry
        .register(Arc::new(WebSearchTool))
        .expect("Failed to register web_search");
    registry
        .register(Arc::new(CodeExecTool))
        .expect("Failed to register code_exec");

    let registry = Arc::new(registry);
    let timeout_config = ToolTimeoutConfig::default();
    let plan_config = PlanConfig::default();

    tracing::info!(
        goal = %goal,
        tools = ?registry.list_names(),
        ipc_url = %ipc_url,
        "Starting task"
    );

    // Build and run the executor.
    let executor = PlanExecutor::new(
        ipc_client,
        registry,
        timeout_config,
        plan_config,
    );

    match executor.execute(&goal).await {
        Ok(result) => {
            println!("\n--- Task Complete ---");
            println!("Outcome:    {:?}", result.outcome);
            println!("Iterations: {}", result.iterations);
            println!("Replans:    {}", result.total_replans);
            println!("Duration:   {} ms", result.duration_ms);
            if let Some(answer) = &result.final_answer {
                println!("\nFinal Answer:\n{answer}");
            }
            println!("\nSteps:");
            for (i, step) in result.steps.iter().enumerate() {
                let status = if step.success { "ok" } else { "FAIL" };
                let detail = step.error.as_deref().unwrap_or("-");
                println!("  {}: {} [{}] {}", i + 1, step.action, status, detail);
            }
        }
        Err(e) => {
            eprintln!("\n--- Task Failed ---");
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
