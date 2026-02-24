//! Loop detection for agent execution.
//!
//! Detects when the agent is stuck calling the same tool with the same (or similar)
//! inputs repeatedly. This prevents infinite loops that waste tokens and resources.
//!
//! # How It Works
//!
//! - Maintains a sliding window of the last K tool calls (default: K=10)
//! - Hashes each call as `hash(tool_name + canonical(input))`
//! - If a hash appears more than M times in the window (default: M=3), triggers abort
//!
//! Related task: L4-05 in AGENTS.md

use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// Configuration for loop detection.
#[derive(Debug, Clone)]
pub struct LoopDetectorConfig {
    /// Size of the sliding window (number of recent calls to track).
    pub window_size: usize,
    /// Number of times a hash can appear before triggering loop detection.
    pub repeat_threshold: usize,
}

impl Default for LoopDetectorConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            repeat_threshold: 3,
        }
    }
}

/// Loop detector that tracks recent tool calls and identifies infinite loops.
///
/// The detector maintains a sliding window of recent tool call hashes. If the same
/// tool+input combination appears too many times within the window, it signals a loop.
///
/// # Thread Safety
///
/// LoopDetector is not thread-safe and should be owned by a single executor instance.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::reliability::LoopDetector;
/// use serde_json::json;
///
/// let mut detector = LoopDetector::new(10, 3);
///
/// // Record some calls
/// detector.record("web_search", &json!({"query": "rust"}));
/// detector.record("web_search", &json!({"query": "python"}));
/// detector.record("web_search", &json!({"query": "rust"}));
/// detector.record("web_search", &json!({"query": "rust"}));
///
/// // Check for loops
/// if let Some(msg) = detector.is_looping() {
///     println!("Loop detected: {}", msg);
/// }
/// ```
pub struct LoopDetector {
    /// Sliding window of recent call hashes
    window: VecDeque<u64>,
    /// Maximum window size
    window_size: usize,
    /// How many times a hash can appear before it's a loop
    repeat_threshold: usize,
}

impl LoopDetector {
    /// Creates a new loop detector with the given window size and repeat threshold.
    ///
    /// # Parameters
    ///
    /// * `window_size` - Number of recent calls to track
    /// * `repeat_threshold` - Max number of times a call can repeat before triggering
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Track last 10 calls, trigger on 3 repeats
    /// let detector = LoopDetector::new(10, 3);
    /// ```
    pub fn new(window_size: usize, repeat_threshold: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            repeat_threshold,
        }
    }

    /// Creates a loop detector from configuration.
    pub fn from_config(config: LoopDetectorConfig) -> Self {
        Self::new(config.window_size, config.repeat_threshold)
    }

    /// Records a tool call in the sliding window.
    ///
    /// If the window is full, the oldest call is evicted before adding the new one.
    ///
    /// # Parameters
    ///
    /// * `tool_name` - Name of the tool being called
    /// * `input` - Input JSON passed to the tool
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// detector.record("web_search", &json!({"query": "rust"}));
    /// ```
    pub fn record(&mut self, tool_name: &str, input: &Value) {
        let hash = Self::hash_call(tool_name, input);

        // Evict oldest if window is full
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }

        self.window.push_back(hash);
    }

    /// Checks if the agent is currently looping.
    ///
    /// Returns `Some(message)` if a loop is detected, with a descriptive error message.
    /// Returns `None` if no loop is detected.
    ///
    /// # Returns
    ///
    /// * `Some(String)` - Error message describing the loop
    /// * `None` - No loop detected
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(error) = detector.is_looping() {
    ///     return Err(PlanError::Loop(error));
    /// }
    /// ```
    pub fn is_looping(&self) -> Option<String> {
        // Count occurrences of each hash in the window
        let mut counts: HashMap<u64, usize> = HashMap::new();

        for &hash in &self.window {
            *counts.entry(hash).or_insert(0) += 1;
        }

        // Check if any hash exceeds the threshold
        for (hash, count) in counts {
            if count >= self.repeat_threshold {
                return Some(format!(
                    "Detected loop: tool called {} times with identical input (hash: {})",
                    count, hash
                ));
            }
        }

        None
    }

    /// Computes a hash for a tool call.
    ///
    /// The hash is based on the tool name and a canonical representation of the input JSON.
    ///
    /// # Parameters
    ///
    /// * `tool_name` - Name of the tool
    /// * `input` - Input JSON (will be canonicalized)
    ///
    /// # Returns
    ///
    /// A 64-bit hash uniquely identifying this tool+input combination
    fn hash_call(tool_name: &str, input: &Value) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the tool name
        tool_name.hash(&mut hasher);

        // Canonicalize JSON by converting to string
        // serde_json serialization is deterministic for the same data structure
        let canonical = serde_json::to_string(input).unwrap_or_default();
        canonical.hash(&mut hasher);

        hasher.finish()
    }

    /// Returns the current window size for testing/observability.
    #[cfg(test)]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Clears the detection window.
    ///
    /// Useful when starting a new task or after a successful replan.
    pub fn clear(&mut self) {
        self.window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_new_detector_is_empty() {
        let detector = LoopDetector::new(10, 3);
        assert_eq!(detector.window_len(), 0);
        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_record_adds_to_window() {
        let mut detector = LoopDetector::new(10, 3);

        detector.record("web_search", &json!({"query": "rust"}));
        assert_eq!(detector.window_len(), 1);

        detector.record("url_fetch", &json!({"url": "https://example.com"}));
        assert_eq!(detector.window_len(), 2);
    }

    #[test]
    fn test_window_evicts_oldest_when_full() {
        let mut detector = LoopDetector::new(3, 2);

        detector.record("tool1", &json!({"a": 1}));
        detector.record("tool2", &json!({"b": 2}));
        detector.record("tool3", &json!({"c": 3}));
        assert_eq!(detector.window_len(), 3);

        // This should evict the oldest (tool1)
        detector.record("tool4", &json!({"d": 4}));
        assert_eq!(detector.window_len(), 3);
    }

    #[test]
    fn test_no_loop_with_different_calls() {
        let mut detector = LoopDetector::new(10, 3);

        detector.record("web_search", &json!({"query": "rust"}));
        detector.record("web_search", &json!({"query": "python"}));
        detector.record("url_fetch", &json!({"url": "https://example.com"}));

        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_detects_loop_with_repeated_calls() {
        let mut detector = LoopDetector::new(10, 3);

        let input = json!({"query": "rust"});

        // Record the same call 3 times
        detector.record("web_search", &input);
        detector.record("web_search", &input);
        assert!(detector.is_looping().is_none()); // Not looping yet

        detector.record("web_search", &input);
        let result = detector.is_looping();
        assert!(result.is_some());
        assert!(result.unwrap().contains("3 times"));
    }

    #[test]
    fn test_different_inputs_not_a_loop() {
        let mut detector = LoopDetector::new(10, 3);

        // Same tool, different inputs
        detector.record("web_search", &json!({"query": "rust"}));
        detector.record("web_search", &json!({"query": "python"}));
        detector.record("web_search", &json!({"query": "java"}));

        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_same_input_different_tools_not_a_loop() {
        let mut detector = LoopDetector::new(10, 3);

        let input = json!({"data": "test"});

        // Different tools, same input
        detector.record("tool1", &input);
        detector.record("tool2", &input);
        detector.record("tool3", &input);

        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_loop_detection_across_window_boundary() {
        let mut detector = LoopDetector::new(5, 3);

        let input = json!({"query": "test"});

        // Fill window with repeating pattern
        detector.record("web_search", &input);
        detector.record("other_tool", &json!({"x": 1}));
        detector.record("web_search", &input);
        detector.record("other_tool", &json!({"x": 1}));
        detector.record("web_search", &input);

        // Should detect loop (web_search called 3 times)
        assert!(detector.is_looping().is_some());
    }

    #[test]
    fn test_window_eviction_prevents_false_positive() {
        let mut detector = LoopDetector::new(3, 3);

        let input = json!({"query": "test"});

        // Record the same call 2 times
        detector.record("web_search", &input);
        detector.record("web_search", &input);

        // Fill window with different calls (evicts the first two)
        detector.record("tool1", &json!({"a": 1}));
        detector.record("tool2", &json!({"b": 2}));

        // Now record the same call again - should not trigger (only 1 in window)
        detector.record("web_search", &input);

        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_clear_resets_window() {
        let mut detector = LoopDetector::new(10, 3);

        detector.record("web_search", &json!({"query": "rust"}));
        detector.record("web_search", &json!({"query": "rust"}));
        detector.record("web_search", &json!({"query": "rust"}));

        assert!(detector.is_looping().is_some());

        detector.clear();
        assert_eq!(detector.window_len(), 0);
        assert!(detector.is_looping().is_none());
    }

    #[test]
    fn test_hash_call_same_for_same_input() {
        let hash1 = LoopDetector::hash_call("web_search", &json!({"query": "rust"}));
        let hash2 = LoopDetector::hash_call("web_search", &json!({"query": "rust"}));

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_call_different_for_different_input() {
        let hash1 = LoopDetector::hash_call("web_search", &json!({"query": "rust"}));
        let hash2 = LoopDetector::hash_call("web_search", &json!({"query": "python"}));

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_call_different_for_different_tool() {
        let hash1 = LoopDetector::hash_call("web_search", &json!({"query": "rust"}));
        let hash2 = LoopDetector::hash_call("url_fetch", &json!({"query": "rust"}));

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_from_config() {
        let config = LoopDetectorConfig {
            window_size: 5,
            repeat_threshold: 2,
        };

        let detector = LoopDetector::from_config(config);
        assert_eq!(detector.window_size, 5);
        assert_eq!(detector.repeat_threshold, 2);
    }

    #[test]
    fn test_default_config() {
        let config = LoopDetectorConfig::default();
        assert_eq!(config.window_size, 10);
        assert_eq!(config.repeat_threshold, 3);
    }
}
