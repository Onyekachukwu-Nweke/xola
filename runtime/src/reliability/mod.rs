//! Reliability and fault tolerance mechanisms for the Xola runtime.
//!
//! This module provides production-grade fault tolerance including:
//! - Circuit breakers to prevent cascading failures (L4-01, L4-02)
//! - Loop detection to catch infinite agent loops (L4-05)
//! - Hierarchical timeout management (L4-06)
//! - Structured failure taxonomy (L4-07)
//!
//! Related tasks: L4-01 through L4-08 in AGENTS.md

pub mod circuit_breaker;
pub mod loop_detector;
pub mod timeout;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use loop_detector::{LoopDetector, LoopDetectorConfig};
pub use timeout::{TimeoutConfig, TimeoutContext};
