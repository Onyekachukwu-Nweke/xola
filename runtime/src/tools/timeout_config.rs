//! Timeout configuration for tool execution.
//!
//! Provides configurable per-tool timeout durations with a default fallback.
//! Timeouts prevent tools from blocking indefinitely when external services hang
//! or tools enter infinite loops.

use std::collections::HashMap;
use std::time::Duration;

/// Configuration for tool execution timeouts.
///
/// Provides a default timeout for all tools with optional per-tool overrides.
/// Timeout values are in milliseconds for easier configuration via environment variables.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::ToolTimeoutConfig;
///
/// let mut config = ToolTimeoutConfig::default();
/// config.set_timeout("web_search", 15_000);  // 15 seconds
/// config.set_timeout("code_exec", 300_000);  // 5 minutes
///
/// let timeout = config.get_timeout("web_search");
/// assert_eq!(timeout.as_millis(), 15_000);
/// ```
#[derive(Clone, Debug)]
pub struct ToolTimeoutConfig {
    /// Default timeout for all tools in milliseconds (default: 60000 = 60s)
    pub default_timeout_ms: u64,

    /// Per-tool timeout overrides in milliseconds
    pub per_tool_timeout_ms: HashMap<String, u64>,
}

impl ToolTimeoutConfig {
    /// Creates a new timeout config from environment variables.
    ///
    /// Reads:
    /// - `XOLA_TOOL_TIMEOUT_MS` for the default timeout
    /// - Per-tool overrides can be added programmatically via `set_timeout()`
    ///
    /// # Example
    ///
    /// ```bash
    /// export XOLA_TOOL_TIMEOUT_MS=60000  # 60 second default
    /// ```
    ///
    /// ```rust,ignore
    /// let config = ToolTimeoutConfig::from_env();
    /// println!("Default timeout: {}ms", config.default_timeout_ms);
    /// ```
    #[allow(dead_code)] // Will be used when loading config at startup
    pub fn from_env() -> Self {
        let default = std::env::var("XOLA_TOOL_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_000);

        Self {
            default_timeout_ms: default,
            per_tool_timeout_ms: HashMap::new(),
        }
    }

    /// Sets a per-tool timeout override.
    ///
    /// This allows different tools to have different timeout durations based on
    /// their expected execution time. For example, network I/O tools might have
    /// shorter timeouts than code execution tools.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut config = ToolTimeoutConfig::default();
    /// config.set_timeout("web_search", 15_000);  // 15 seconds
    /// config.set_timeout("code_exec", 300_000);  // 5 minutes
    /// ```
    #[allow(dead_code)] // Will be used in L1-05+ when registering real tools
    pub fn set_timeout(&mut self, tool_name: &str, timeout_ms: u64) {
        self.per_tool_timeout_ms
            .insert(tool_name.to_string(), timeout_ms);
    }

    /// Gets the timeout duration for a specific tool.
    ///
    /// Returns the per-tool override if set, otherwise the default timeout.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ToolTimeoutConfig::default();
    /// let timeout = config.get_timeout("mock_echo");
    /// assert_eq!(timeout.as_millis(), 60_000);  // default
    /// ```
    pub fn get_timeout(&self, tool_name: &str) -> Duration {
        let ms = self
            .per_tool_timeout_ms
            .get(tool_name)
            .copied()
            .unwrap_or(self.default_timeout_ms);
        Duration::from_millis(ms)
    }
}

impl Default for ToolTimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 60_000, // 60 seconds
            per_tool_timeout_ms: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_timeout() {
        let config = ToolTimeoutConfig::default();
        assert_eq!(config.default_timeout_ms, 60_000);
        assert_eq!(config.get_timeout("any_tool").as_millis(), 60_000);
    }

    #[test]
    fn test_per_tool_override() {
        let mut config = ToolTimeoutConfig::default();
        config.set_timeout("fast_tool", 5_000);

        assert_eq!(config.get_timeout("fast_tool").as_millis(), 5_000);
        assert_eq!(config.get_timeout("other_tool").as_millis(), 60_000);
    }

    #[test]
    fn test_from_env_with_custom_default() {
        std::env::set_var("XOLA_TOOL_TIMEOUT_MS", "30000");
        let config = ToolTimeoutConfig::from_env();

        assert_eq!(config.default_timeout_ms, 30_000);
        std::env::remove_var("XOLA_TOOL_TIMEOUT_MS");
    }

    #[test]
    fn test_from_env_fallback_to_default() {
        std::env::remove_var("XOLA_TOOL_TIMEOUT_MS");
        let config = ToolTimeoutConfig::from_env();

        assert_eq!(config.default_timeout_ms, 60_000);
    }

    #[test]
    fn test_multiple_per_tool_timeouts() {
        let mut config = ToolTimeoutConfig::default();
        config.set_timeout("web_search", 15_000);
        config.set_timeout("code_exec", 300_000);

        assert_eq!(config.get_timeout("web_search").as_millis(), 15_000);
        assert_eq!(config.get_timeout("code_exec").as_millis(), 300_000);
        assert_eq!(config.get_timeout("url_fetch").as_millis(), 60_000);
    }

    #[test]
    fn test_clone() {
        let mut config = ToolTimeoutConfig::default();
        config.set_timeout("test_tool", 10_000);

        let cloned = config.clone();
        assert_eq!(cloned.default_timeout_ms, 60_000);
        assert_eq!(cloned.get_timeout("test_tool").as_millis(), 10_000);
    }

    #[test]
    fn test_debug_format() {
        let config = ToolTimeoutConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ToolTimeoutConfig"));
        assert!(debug_str.contains("60000"));
    }
}
