//! OTel metrics with Prometheus exporter.
//!
//! Metrics are defined using the OpenTelemetry Meter API and exposed via a
//! Prometheus scrape endpoint (`GET /metrics`) served by axum.

use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, routing::get, Router};
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry_prometheus::PrometheusExporter;
use prometheus::Registry as PrometheusRegistry;

/// All Xola runtime metrics, defined via the OTel Meter API.
///
/// Pass `Arc<Metrics>` to modules that need to record telemetry.
pub struct Metrics {
    /// Total tool calls, labeled by tool_name and status.
    pub tool_calls_total: Counter<u64>,
    /// Tool call duration in seconds.
    pub tool_duration_seconds: Histogram<f64>,
    /// Total planning steps executed.
    pub plan_steps_total: Counter<u64>,
    /// Total replanning events.
    pub plan_replans_total: Counter<u64>,
    /// Total memory queries, labeled by memory_type.
    pub memory_queries_total: Counter<u64>,
    /// Total LLM tokens consumed, labeled by model and direction.
    pub llm_tokens_total: Counter<u64>,
    /// Cumulative LLM cost in USD.
    pub llm_cost_usd_total: Counter<f64>,
    /// End-to-end task duration in seconds.
    pub task_duration_seconds: Histogram<f64>,
}

impl Metrics {
    /// Create all metric instruments from a Meter.
    pub fn new(meter: &Meter) -> Self {
        Self {
            tool_calls_total: meter
                .u64_counter("xola_tool_calls_total")
                .with_description("Total tool calls")
                .build(),
            tool_duration_seconds: meter
                .f64_histogram("xola_tool_duration_seconds")
                .with_description("Tool call duration in seconds")
                .build(),
            plan_steps_total: meter
                .u64_counter("xola_plan_steps_total")
                .with_description("Total planning steps executed")
                .build(),
            plan_replans_total: meter
                .u64_counter("xola_plan_replans_total")
                .with_description("Total replanning events")
                .build(),
            memory_queries_total: meter
                .u64_counter("xola_memory_queries_total")
                .with_description("Total memory queries")
                .build(),
            llm_tokens_total: meter
                .u64_counter("xola_llm_tokens_total")
                .with_description("Total LLM tokens consumed")
                .build(),
            llm_cost_usd_total: meter
                .f64_counter("xola_llm_cost_usd_total")
                .with_description("Cumulative LLM cost in USD")
                .build(),
            task_duration_seconds: meter
                .f64_histogram("xola_task_duration_seconds")
                .with_description("End-to-end task duration in seconds")
                .build(),
        }
    }
}

/// Shared state for the metrics HTTP server.
struct MetricsState {
    registry: PrometheusRegistry,
}

/// Build the Prometheus exporter and return both the exporter (for the OTel
/// MeterProvider) and the Prometheus Registry (for the HTTP handler).
pub fn build_prometheus_exporter(
) -> Result<(PrometheusExporter, PrometheusRegistry), MetricsInitError> {
    let registry = PrometheusRegistry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .map_err(MetricsInitError::Exporter)?;
    Ok((exporter, registry))
}

/// Build the axum Router serving `GET /metrics`.
pub fn metrics_router(registry: PrometheusRegistry) -> Router {
    let state = Arc::new(MetricsState { registry });
    Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

/// Handler that encodes the Prometheus registry into text format.
async fn metrics_handler(State(state): State<Arc<MetricsState>>) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = state.registry.gather();
    let body = encoder
        .encode_to_string(&metric_families)
        .unwrap_or_else(|e| format!("# error encoding metrics: {e}\n"));

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// Errors during metrics initialization.
#[derive(Debug, thiserror::Error)]
pub enum MetricsInitError {
    #[error("failed to build Prometheus exporter: {0}")]
    Exporter(#[source] opentelemetry_sdk::error::OTelSdkError),
}
