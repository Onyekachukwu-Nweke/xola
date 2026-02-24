//! Observability — distributed tracing, metrics, and runtime introspection.
//!
//! This module initialises the OpenTelemetry SDK for both traces (→ Jaeger via
//! OTLP) and metrics (→ Prometheus via scrape endpoint). All instrumentation
//! uses the `tracing` crate; the `tracing-opentelemetry` bridge forwards spans
//! to the OTel SDK transparently.
//!
//! # Pipeline
//!
//! ```text
//! tracing macros → tracing-opentelemetry → OTel SDK → OTLP gRPC → Jaeger
//!                                        → Prometheus exporter  → /metrics
//! ```

pub mod metrics;
pub mod tracing_init;

use std::sync::Arc;

use opentelemetry::metrics::{Meter, MeterProvider};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::config::ObservabilityConfig;
use metrics::Metrics;

/// Guard that flushes and shuts down OTel providers on drop.
///
/// Hold this in `main()` for the lifetime of the process.
pub struct ObservabilityGuard {
    _tracer_provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
    _meter_provider: Option<SdkMeterProvider>,
}

impl Drop for ObservabilityGuard {
    fn drop(&mut self) {
        if let Some(ref provider) = self._tracer_provider {
            if let Err(e) = provider.shutdown() {
                eprintln!("Failed to shutdown tracer provider: {e}");
            }
        }
        if let Some(ref provider) = self._meter_provider {
            if let Err(e) = provider.shutdown() {
                eprintln!("Failed to shutdown meter provider: {e}");
            }
        }
    }
}

/// Initialisation result containing the guard, metrics, and optional metrics router.
pub struct ObservabilityInit {
    /// Hold this for the lifetime of the process.
    pub guard: ObservabilityGuard,
    /// Shared metrics instruments (None if metrics are disabled).
    pub metrics: Option<Arc<Metrics>>,
    /// Axum router serving `/metrics` (None if metrics are disabled).
    pub metrics_router: Option<axum::Router>,
}

/// Initialise observability: tracing subscriber + optional OTel + optional metrics.
///
/// This replaces the bare `tracing_subscriber::fmt().init()` in `main.rs`.
pub fn init_observability(config: &ObservabilityConfig) -> Result<ObservabilityInit, ObsInitError> {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // --- Tracing ---
    let tracer_provider = if config.enable_tracing {
        match tracing_init::build_tracer_provider(config) {
            Ok(tp) => Some(tp),
            Err(e) => {
                eprintln!("Warning: failed to init OTel tracing, falling back to stdout: {e}");
                None
            }
        }
    } else {
        None
    };

    // --- Metrics ---
    let (meter_provider, prom_metrics, metrics_router) = if config.enable_metrics {
        match metrics::build_prometheus_exporter() {
            Ok((exporter, registry)) => {
                let provider = SdkMeterProvider::builder()
                    .with_reader(exporter)
                    .build();
                let meter: Meter = provider.meter("xola");
                let m = Arc::new(Metrics::new(&meter));
                let router = metrics::metrics_router(registry);
                (Some(provider), Some(m), Some(router))
            }
            Err(e) => {
                eprintln!("Warning: failed to init metrics, continuing without: {e}");
                (None, None, None)
            }
        }
    } else {
        (None, None, None)
    };

    // --- Build layered subscriber ---
    // We need to handle the optional OTel layer via dynamic dispatch because
    // the layer types differ depending on whether tracing is enabled.
    if let Some(ref tp) = tracer_provider {
        let otel_layer = tracing_init::otel_layer(tp);
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otel_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }

    Ok(ObservabilityInit {
        guard: ObservabilityGuard {
            _tracer_provider: tracer_provider,
            _meter_provider: meter_provider,
        },
        metrics: prom_metrics,
        metrics_router,
    })
}

/// Errors during observability initialization.
#[derive(Debug, thiserror::Error)]
pub enum ObsInitError {
    #[error("tracing init failed: {0}")]
    Tracing(#[from] tracing_init::TraceInitError),

    #[error("metrics init failed: {0}")]
    Metrics(#[from] metrics::MetricsInitError),
}
