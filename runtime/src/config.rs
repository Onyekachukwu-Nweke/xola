//! Runtime configuration parsed from TOML files.
//!
//! The canonical configuration file lives at `config/local.toml` for
//! development. In production, values are overridden via environment
//! variables or injected TOML files.

use serde::Deserialize;

/// Top-level runtime configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

/// Observability settings controlling tracing and metrics export.
#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable OpenTelemetry distributed tracing.
    #[serde(default = "default_true")]
    pub enable_tracing: bool,

    /// OTLP gRPC endpoint (e.g. `http://localhost:4317`).
    #[serde(default = "default_otlp_endpoint")]
    pub otlp_endpoint: String,

    /// Enable Prometheus metrics via `/metrics` endpoint.
    #[serde(default)]
    pub enable_metrics: bool,

    /// Port for the Prometheus metrics HTTP server.
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Service name reported in traces and metrics.
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_tracing: true,
            otlp_endpoint: default_otlp_endpoint(),
            enable_metrics: false,
            metrics_port: default_metrics_port(),
            service_name: default_service_name(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_otlp_endpoint() -> String {
    "http://localhost:4317".to_string()
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_service_name() -> String {
    "xola-runtime".to_string()
}

/// Load configuration from a TOML file, falling back to defaults.
///
/// If the file does not exist or is empty, returns default configuration.
pub fn load_config(path: &str) -> Result<RuntimeConfig, ConfigError> {
    let contents = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
        path: path.to_string(),
        source: e,
    })?;

    let config: RuntimeConfig =
        toml::from_str(&contents).map_err(|e| ConfigError::Parse {
            path: path.to_string(),
            source: e,
        })?;

    Ok(config)
}

/// Errors that can occur during configuration loading.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file '{path}': {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },

    #[error("failed to parse config file '{path}': {source}")]
    Parse {
        path: String,
        source: toml::de::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ObservabilityConfig::default();
        assert!(config.enable_tracing);
        assert!(!config.enable_metrics);
        assert_eq!(config.otlp_endpoint, "http://localhost:4317");
        assert_eq!(config.metrics_port, 9090);
        assert_eq!(config.service_name, "xola-runtime");
    }

    #[test]
    fn test_parse_minimal_toml() {
        let toml_str = r#"
[observability]
enable_tracing = false
"#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.observability.enable_tracing);
        // Defaults should fill in
        assert_eq!(config.observability.otlp_endpoint, "http://localhost:4317");
    }

    #[test]
    fn test_parse_full_toml() {
        let toml_str = r#"
[observability]
enable_tracing = true
otlp_endpoint = "http://otel:4317"
enable_metrics = true
metrics_port = 9999
service_name = "test-service"
"#;
        let config: RuntimeConfig = toml::from_str(toml_str).unwrap();
        assert!(config.observability.enable_tracing);
        assert!(config.observability.enable_metrics);
        assert_eq!(config.observability.otlp_endpoint, "http://otel:4317");
        assert_eq!(config.observability.metrics_port, 9999);
        assert_eq!(config.observability.service_name, "test-service");
    }

    #[test]
    fn test_empty_toml_uses_defaults() {
        let config: RuntimeConfig = toml::from_str("").unwrap();
        assert!(config.observability.enable_tracing);
        assert!(!config.observability.enable_metrics);
    }
}
