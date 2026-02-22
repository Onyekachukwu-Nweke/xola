//! Xola Agent Runtime Library
//!
//! A two-process AI agent runtime built in Rust and Python.
//! This library provides the core runtime functionality: tool dispatch,
//! memory management, planning, fault tolerance, and observability.
//!
//! See CLAUDE.md and docs/ for architecture and design rationale.

pub mod memory;
pub mod tools;

#[cfg(test)]
pub mod test_support;
