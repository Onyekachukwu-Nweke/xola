//! Reliability and fault tolerance mechanisms for the Xola runtime.
//!
//! This module provides production-grade fault tolerance including:
//! - Circuit breakers to prevent cascading failures
//! - Loop detection to catch infinite agent loops
//! - Hierarchical timeout management
//! - Structured failure taxonomy
//!
//! Related tasks: L4-01 through L4-08 in AGENTS.md

pub mod circuit_breaker;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
