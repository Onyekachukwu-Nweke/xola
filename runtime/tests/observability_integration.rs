//! Observability integration tests — verify spans and metrics are emitted.
//!
//! Uses `InMemorySpanExporter` from the OTel SDK to capture spans in-process
//! without requiring a running Jaeger instance.

use std::sync::Arc;
use std::time::Duration;

use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::trace::{
    InMemorySpanExporterBuilder, SdkTracerProvider, SimpleSpanProcessor,
};
use serde_json::{json, Value};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use xola_runtime::ipc::*;
use xola_runtime::planning::config::PlanConfig;
use xola_runtime::planning::executor::PlanExecutor;
use xola_runtime::tools::mock::MockTool;
use xola_runtime::tools::{ToolRegistry, ToolTimeoutConfig};

// ---------------------------------------------------------------------------
// Mock IPC (same pattern as executor unit tests)
// ---------------------------------------------------------------------------

struct MockIpc {
    responses: std::sync::Mutex<Vec<ReasonResponse>>,
}

impl MockIpc {
    fn new(responses: Vec<ReasonResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait::async_trait]
impl IpcTransport for MockIpc {
    async fn reason(&self, _req: &ReasonRequest) -> Result<ReasonResponse, IpcError> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(ReasonResponse {
                thought: "Done.".to_string(),
                action: None,
                action_input: json!({}),
                is_final: true,
                final_answer: Some("default".to_string()),
                usage: None,
            })
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn embed(&self, _req: &EmbedRequest) -> Result<EmbedResponse, IpcError> {
        Ok(EmbedResponse {
            vector: vec![0.0; 1536],
            token_count: 5,
            cost_usd: None,
        })
    }

    async fn summarize(&self, _req: &SummarizeRequest) -> Result<SummarizeResponse, IpcError> {
        Ok(SummarizeResponse {
            summary: "Summary.".to_string(),
            token_count: 5,
        })
    }
}

fn make_response(
    thought: &str,
    action: Option<&str>,
    action_input: Value,
    is_final: bool,
    final_answer: Option<&str>,
) -> ReasonResponse {
    ReasonResponse {
        thought: thought.to_string(),
        action: action.map(|s| s.to_string()),
        action_input,
        is_final,
        final_answer: final_answer.map(|s| s.to_string()),
        usage: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that a two-step task produces the expected span hierarchy:
/// `agent_run` → `plan_step` (×2), each containing `tool_call`.
#[tokio::test]
async fn trace_contains_expected_span_types() {
    // 1. Set up InMemorySpanExporter + TracerProvider.
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();

    let tracer = provider.tracer("xola-test");
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // Install a subscriber with the OTel layer for this test.
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let _guard = subscriber.set_default();

    // 2. Build executor with MockIpc.
    let ipc = Arc::new(MockIpc::new(vec![
        // Step 1: call mock_echo.
        make_response(
            "Let me echo.",
            Some("mock_echo"),
            json!({"message": "hello"}),
            false,
            None,
        ),
        // Step 2: final answer.
        make_response("Done.", None, json!({}), true, Some("echoed")),
    ]));

    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("register mock");
    let executor = PlanExecutor::new(
        ipc,
        Arc::new(registry),
        ToolTimeoutConfig::default(),
        PlanConfig {
            max_iterations: 10,
            stm_token_budget: 8000,
            ..Default::default()
        },
    );

    // 3. Execute the task.
    let result = executor.execute("Echo hello").await.unwrap();
    assert_eq!(result.iterations, 2);

    // 4. Force flush the provider to export all spans.
    let _ = provider.force_flush();

    // 5. Inspect exported spans.
    let spans = exporter.get_finished_spans().unwrap();
    let span_names: Vec<&str> = spans.iter().map(|s| s.name.as_ref()).collect();

    // We expect at least: agent_run, plan_step (×2), tool_call (×1)
    assert!(
        span_names.contains(&"agent_run"),
        "Expected 'agent_run' span, got: {span_names:?}"
    );
    assert!(
        span_names.contains(&"plan_step"),
        "Expected 'plan_step' span, got: {span_names:?}"
    );
    assert!(
        span_names.contains(&"tool_call"),
        "Expected 'tool_call' span, got: {span_names:?}"
    );
}

/// Verify that tool_call spans contain the expected attributes.
#[tokio::test]
async fn tool_call_span_has_attributes() {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();

    let tracer = provider.tracer("xola-test-attrs");
    let otel_layer = OpenTelemetryLayer::new(tracer);
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    let _guard = subscriber.set_default();

    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(MockTool))
        .expect("register mock");

    // Execute a single tool call directly through the registry.
    let input = json!({"message": "test"});
    let _result = registry
        .execute_with_timeout("mock_echo", input, Duration::from_secs(5))
        .await
        .unwrap();

    let _ = provider.force_flush();

    let spans = exporter.get_finished_spans().unwrap();
    let tool_span = spans
        .iter()
        .find(|s| s.name.as_ref() == "tool_call")
        .expect("tool_call span should exist");

    // Check that tool.name attribute is set.
    let attr_keys: Vec<&str> = tool_span
        .attributes
        .iter()
        .map(|kv| kv.key.as_str())
        .collect();

    assert!(
        attr_keys.contains(&"tool.name"),
        "Expected 'tool.name' attribute on tool_call span, got: {attr_keys:?}"
    );
}
