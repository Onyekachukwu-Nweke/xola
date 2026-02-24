//! Xola Agent Runtime Library
//!
//! A two-process AI agent runtime built in Rust and Python.
//! This library provides the core runtime functionality: tool dispatch,
//! memory management, planning, fault tolerance, and observability.
//!
//! See CLAUDE.md and docs/ for architecture and design rationale.

pub mod ipc;
pub mod memory;
pub mod planning;
pub mod reliability;
pub mod tools;

#[cfg(test)]
pub mod test_support;
