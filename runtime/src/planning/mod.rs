//! Task planning and orchestration layer.
//!
//! Implements the ReAct loop: the `PlanExecutor` calls `/reason`, dispatches
//! tools, observes results, and replans when things go wrong. This is the
//! layer that turns a passive LLM into an autonomous agent.
//!
//! See `docs/layer-3-planning.md` for design rationale.

pub mod config;
pub mod error;
pub mod executor;

pub use config::{PlanConfig, ReplanConfig};
pub use error::PlanError;
pub use executor::{PlanExecutor, TaskResult};
