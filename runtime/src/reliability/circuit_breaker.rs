//! Circuit breaker implementation to prevent cascading failures.
//!
//! A circuit breaker monitors tool execution failures and temporarily stops
//! calling a tool that has failed repeatedly. This prevents resource waste
//! and gives failing services time to recover.
//!
//! # State Machine
//!
//! ```text
//!            success
//!     ┌───────────────────┐
//!     │                   ▼
//! ┌───────┐  failure   ┌───────┐  timeout   ┌───────────┐
//! │CLOSED │──────────▶ │ OPEN  │───────────▶│ HALF-OPEN │
//! │       │ (N times)  │       │ (wait T)   │           │
//! └───────┘            └───────┘            └─────┬─────┘
//!     ▲                    ▲                      │
//!     │                    │    failure            │
//!     │                    └──────────────────────┘
//!     │                         success
//!     └─────────────────────────────────┘
//! ```
//!
//! Related tasks: L4-01, L4-02 in AGENTS.md

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: usize,
    /// Time to wait in open state before attempting a half-open probe.
    pub recovery_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
        }
    }
}

/// Circuit breaker state values for atomic storage.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Normal operation. Failures are counted.
    Closed = 0,
    /// All calls rejected immediately. Entered after N consecutive failures.
    Open = 1,
    /// One probe call allowed after cooldown. Success → Closed. Failure → Open.
    HalfOpen = 2,
}

impl From<u8> for State {
    fn from(value: u8) -> Self {
        match value {
            0 => State::Closed,
            1 => State::Open,
            2 => State::HalfOpen,
            _ => State::Closed, // Defensive default
        }
    }
}

/// Circuit breaker for protecting against cascading failures.
///
/// Each tool in the registry gets its own circuit breaker instance.
/// The circuit breaker uses atomic operations for thread-safe state management.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::reliability::{CircuitBreaker, CircuitBreakerConfig};
///
/// let config = CircuitBreakerConfig::default();
/// let breaker = CircuitBreaker::new(config);
///
/// // Before calling tool
/// if !breaker.can_execute() {
///     return Err("Circuit breaker is open");
/// }
///
/// // After tool call
/// match result {
///     Ok(_) => breaker.record_success(),
///     Err(_) => breaker.record_failure(),
/// }
/// ```
pub struct CircuitBreaker {
    /// Current state: Closed=0, Open=1, HalfOpen=2
    state: AtomicU8,
    /// Count of consecutive failures in Closed state
    failure_count: AtomicUsize,
    /// Timestamp (seconds since UNIX_EPOCH) of last failure
    last_failure: AtomicU64,
    /// Configuration
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(State::Closed as u8),
            failure_count: AtomicUsize::new(0),
            last_failure: AtomicU64::new(0),
            config,
        }
    }

    /// Checks if a tool execution should be allowed.
    ///
    /// Returns `true` if the circuit is Closed or HalfOpen.
    /// Returns `false` if the circuit is Open and the recovery timeout hasn't elapsed.
    ///
    /// If the circuit is Open and the recovery timeout has elapsed, transitions
    /// to HalfOpen and returns `true` to allow a probe call.
    pub fn can_execute(&self) -> bool {
        let state = State::from(self.state.load(Ordering::Acquire));

        match state {
            State::Closed => true,
            State::HalfOpen => true,
            State::Open => {
                // Check if we should attempt reset to half-open
                if self.should_attempt_reset() {
                    // Transition to half-open for probe
                    self.state.store(State::HalfOpen as u8, Ordering::Release);
                    tracing::info!(
                        circuit_breaker.from = "Open",
                        circuit_breaker.to = "HalfOpen",
                        "circuit breaker: transitioning to half-open for probe"
                    );
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Records a successful tool execution.
    ///
    /// - In Closed state: Resets failure count to 0
    /// - In HalfOpen state: Transitions to Closed and resets failure count
    /// - In Open state: No-op (shouldn't reach here)
    pub fn record_success(&self) {
        let state = State::from(self.state.load(Ordering::Acquire));

        match state {
            State::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Release);
            }
            State::HalfOpen => {
                // Probe succeeded - transition back to closed
                self.state.store(State::Closed as u8, Ordering::Release);
                self.failure_count.store(0, Ordering::Release);
                tracing::info!(
                    circuit_breaker.from = "HalfOpen",
                    circuit_breaker.to = "Closed",
                    "circuit breaker: probe succeeded, closing circuit"
                );
            }
            State::Open => {
                // Shouldn't happen, but be defensive
            }
        }
    }

    /// Records a failed tool execution.
    ///
    /// - In Closed state: Increments failure count. If threshold reached, opens circuit.
    /// - In HalfOpen state: Probe failed - transition back to Open
    /// - In Open state: No-op (shouldn't reach here)
    pub fn record_failure(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_failure.store(now, Ordering::Release);

        let state = State::from(self.state.load(Ordering::Acquire));

        match state {
            State::Closed => {
                let new_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;

                if new_count >= self.config.failure_threshold {
                    // Threshold reached - open the circuit
                    self.state.store(State::Open as u8, Ordering::Release);
                    tracing::warn!(
                        circuit_breaker.from = "Closed",
                        circuit_breaker.to = "Open",
                        circuit_breaker.failures = new_count,
                        "circuit breaker: failure threshold reached, opening circuit"
                    );
                }
            }
            State::HalfOpen => {
                // Probe failed - go back to open
                self.state.store(State::Open as u8, Ordering::Release);
                tracing::warn!(
                    circuit_breaker.from = "HalfOpen",
                    circuit_breaker.to = "Open",
                    "circuit breaker: probe failed, reopening circuit"
                );
            }
            State::Open => {
                // Already open, no-op
            }
        }
    }

    /// Checks if enough time has elapsed to attempt a reset from Open to HalfOpen.
    fn should_attempt_reset(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let last_failure = self.last_failure.load(Ordering::Acquire);

        let elapsed = now.saturating_sub(last_failure);
        elapsed >= self.config.recovery_timeout.as_millis() as u64
    }

    /// Returns the current state for testing/observability.
    ///
    /// Note: This is a snapshot and may change immediately after being read.
    #[cfg(test)]
    pub fn current_state(&self) -> String {
        let state = State::from(self.state.load(Ordering::Acquire));
        match state {
            State::Closed => "Closed".to_string(),
            State::Open => "Open".to_string(),
            State::HalfOpen => "HalfOpen".to_string(),
        }
    }

    /// Returns the current failure count for testing/observability.
    #[cfg(test)]
    pub fn failure_count(&self) -> usize {
        self.failure_count.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_breaker_is_closed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(breaker.current_state(), "Closed");
        assert_eq!(breaker.failure_count(), 0);
        assert!(breaker.can_execute());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_secs(30),
        };
        let breaker = CircuitBreaker::new(config);

        // Record some failures
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.failure_count(), 2);

        // Success should reset count
        breaker.record_success();
        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.current_state(), "Closed");
    }

    #[test]
    fn test_opens_after_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_secs(30),
        };
        let breaker = CircuitBreaker::new(config);

        // Record failures up to threshold
        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Closed");

        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Closed");

        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Open");
    }

    #[test]
    fn test_open_circuit_rejects_calls() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_secs(30),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Open");

        // Should reject calls
        assert!(!breaker.can_execute());
    }

    #[test]
    fn test_half_open_probe_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Open");
        assert!(!breaker.can_execute());

        // Wait for recovery timeout
        thread::sleep(Duration::from_millis(150));

        // Should transition to half-open and allow probe
        assert!(breaker.can_execute());
        assert_eq!(breaker.current_state(), "HalfOpen");
    }

    #[test]
    fn test_half_open_success_closes_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();

        // Wait and transition to half-open
        thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());
        assert_eq!(breaker.current_state(), "HalfOpen");

        // Successful probe should close circuit
        breaker.record_success();
        assert_eq!(breaker.current_state(), "Closed");
        assert!(breaker.can_execute());
    }

    #[test]
    fn test_half_open_failure_reopens_circuit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();

        // Wait and transition to half-open
        thread::sleep(Duration::from_millis(150));
        assert!(breaker.can_execute());
        assert_eq!(breaker.current_state(), "HalfOpen");

        // Failed probe should reopen circuit
        breaker.record_failure();
        assert_eq!(breaker.current_state(), "Open");
        assert!(!breaker.can_execute());
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let config = CircuitBreakerConfig {
            failure_threshold: 100,
            recovery_timeout: Duration::from_secs(1),
        };
        let breaker = Arc::new(CircuitBreaker::new(config));

        let mut handles = vec![];

        // Spawn threads that record successes and failures concurrently
        for i in 0..10 {
            let breaker_clone = Arc::clone(&breaker);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    if i % 2 == 0 {
                        breaker_clone.record_success();
                    } else {
                        breaker_clone.record_failure();
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should still be in a valid state
        assert!(breaker.can_execute());
    }

    #[test]
    fn test_default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.recovery_timeout, Duration::from_secs(30));
    }
}
