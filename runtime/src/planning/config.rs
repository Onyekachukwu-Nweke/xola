//! Configuration for the plan executor.

/// Top-level configuration for the plan execution loop.
#[derive(Debug, Clone)]
pub struct PlanConfig {
    /// Maximum number of ReAct loop iterations before aborting.
    pub max_iterations: usize,
    /// Maximum token budget for the short-term memory buffer.
    pub stm_token_budget: usize,
    /// Replan limits.
    pub replan: ReplanConfig,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            stm_token_budget: 8000,
            replan: ReplanConfig::default(),
        }
    }
}

/// Limits on replanning attempts.
///
/// When a tool call fails, the executor re-calls `/reason` with error
/// context to get an alternative action. These limits prevent infinite
/// retry loops.
#[derive(Debug, Clone)]
pub struct ReplanConfig {
    /// Maximum replan attempts for a single step before escalating to error.
    pub max_replans_per_step: usize,
    /// Maximum total replan attempts across the entire task.
    pub max_replans_per_task: usize,
}

impl Default for ReplanConfig {
    fn default() -> Self {
        Self {
            max_replans_per_step: 3,
            max_replans_per_task: 10,
        }
    }
}
