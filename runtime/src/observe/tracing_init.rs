//! OpenTelemetry tracer provider setup.
//!
//! Configures the OTLP gRPC span exporter and installs a `tracing-opentelemetry`
//! layer so that existing `tracing` macros automatically produce OTel spans.

use opentelemetry::trace::TracerProvider;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{trace::SdkTracerProvider, Resource};
use tracing_opentelemetry::OpenTelemetryLayer;

use crate::config::ObservabilityConfig;

/// Build an OTLP-backed `TracerProvider`.
///
/// Spans are exported via gRPC to the endpoint specified in config (typically
/// `http://localhost:4317` for a local Jaeger).
pub fn build_tracer_provider(
    config: &ObservabilityConfig,
) -> Result<SdkTracerProvider, TraceInitError> {
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build()
        .map_err(TraceInitError::Exporter)?;

    let resource = Resource::builder()
        .with_attributes([KeyValue::new(
            "service.name",
            config.service_name.clone(),
        )])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // Set the global tracer provider so instrumented libraries can find it.
    global::set_tracer_provider(provider.clone());

    Ok(provider)
}

/// Create the `tracing_opentelemetry` layer for the subscriber.
///
/// Returns a layer that is generic over the subscriber type `S`, so it
/// composes cleanly with other layers (fmt, env-filter) in any order.
pub fn otel_layer<S>(
    provider: &SdkTracerProvider,
) -> OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    let tracer = provider.tracer("xola-runtime");
    OpenTelemetryLayer::new(tracer)
}

/// Errors during trace initialization.
#[derive(Debug, thiserror::Error)]
pub enum TraceInitError {
    #[error("failed to build OTLP exporter: {0}")]
    Exporter(#[source] opentelemetry_otlp::ExporterBuildError),
}
