//! Plan executor — the ReAct loop that drives autonomous agent behaviour.
//!
//! The executor calls `/reason` on the Python LLM surface, dispatches tools
//! via the `ToolRegistry`, observes results, and repeats until the task is
//! complete or limits are hit.

use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use tracing::{debug, error, info, info_span, warn, Instrument};

use crate::ipc::{IpcMessage, IpcTransport, ReasonRequest, SummarizeRequest};
use crate::memory::{
    EpisodeInput, EpisodicLog, LongTermMemory, Message, Outcome, Role, ShortTermMemory, TaskStep,
};
use crate::observe::metrics::Metrics;
use crate::reliability::{LoopDetector, LoopDetectorConfig, TimeoutConfig, TimeoutContext};
use crate::tools::{ToolRegistry, ToolTimeoutConfig};

use super::config::PlanConfig;
use super::error::PlanError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The result of a completed task execution.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Whether the task succeeded.
    pub outcome: Outcome,
    /// The final answer, if the task succeeded.
    pub final_answer: Option<String>,
    /// All steps taken during execution.
    pub steps: Vec<TaskStep>,
    /// Total wall-clock duration in milliseconds.
    pub duration_ms: i64,
    /// Number of ReAct loop iterations completed.
    pub iterations: usize,
    /// Number of replans triggered across the entire task.
    pub total_replans: usize,
}

/// Outcome of a single tool dispatch (may include replanning).
enum StepOutcome {
    /// Tool returned a result to feed back as an observation.
    Observation(Value),
    /// The LLM declared the task complete during replanning.
    FinalAnswer {
        thought: String,
        answer: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// The main plan executor. Runs the ReAct loop.
///
/// Generic over `T: IpcTransport` so tests can inject a mock IPC layer.
pub struct PlanExecutor<T: IpcTransport> {
    ipc: Arc<T>,
    registry: Arc<ToolRegistry>,
    timeout_config: ToolTimeoutConfig,
    long_term_memory: Option<LongTermMemory>,
    episodic_log: Option<EpisodicLog>,
    config: PlanConfig,
    loop_detector_config: LoopDetectorConfig,
    timeout_ctx_config: TimeoutConfig,
    metrics: Option<Arc<Metrics>>,
}

impl<T: IpcTransport> PlanExecutor<T> {
    /// Create a new executor.
    ///
    /// `long_term_memory` and `episodic_log` are optional so unit tests
    /// can run without Postgres.
    pub fn new(
        ipc: Arc<T>,
        registry: Arc<ToolRegistry>,
        timeout_config: ToolTimeoutConfig,
        config: PlanConfig,
    ) -> Self {
        Self {
            ipc,
            registry,
            timeout_config,
            long_term_memory: None,
            episodic_log: None,
            config,
            loop_detector_config: LoopDetectorConfig::default(),
            timeout_ctx_config: TimeoutConfig::default(),
            metrics: None,
        }
    }

    /// Attach OTel metrics to this executor.
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set custom loop detector configuration.
    pub fn with_loop_detector_config(mut self, config: LoopDetectorConfig) -> Self {
        self.loop_detector_config = config;
        self
    }

    /// Set custom timeout context configuration.
    pub fn with_timeout_context_config(mut self, config: TimeoutConfig) -> Self {
        self.timeout_ctx_config = config;
        self
    }

    /// Attach long-term memory for semantic context retrieval.
    pub fn with_long_term_memory(mut self, ltm: LongTermMemory) -> Self {
        self.long_term_memory = Some(ltm);
        self
    }

    /// Attach episodic log for recording completed tasks.
    pub fn with_episodic_log(mut self, log: EpisodicLog) -> Self {
        self.episodic_log = Some(log);
        self
    }

    /// Execute a task to completion using the ReAct loop.
    ///
    /// Returns a `TaskResult` on success or a `PlanError` if limits are
    /// reached or an unrecoverable error occurs.
    pub async fn execute(&self, goal: &str) -> Result<TaskResult, PlanError> {
        let span = info_span!(
            "agent_run",
            task.goal = %goal,
            task.status = tracing::field::Empty,
            task.iterations = tracing::field::Empty,
            task.replans = tracing::field::Empty,
            task.duration_ms = tracing::field::Empty,
        );

        self.execute_instrumented(goal).instrument(span).await
    }

    /// Instrumented execution wrapper that records span fields on completion.
    async fn execute_instrumented(&self, goal: &str) -> Result<TaskResult, PlanError> {
        let start = Instant::now();

        // Create per-task reliability structures (L4-05, L4-06).
        let mut loop_detector = LoopDetector::from_config(self.loop_detector_config.clone());
        let timeout_ctx = TimeoutContext::new_task(self.timeout_ctx_config.clone());

        // Race execution against task-level timeout.
        let task_duration = timeout_ctx.task_timeout();
        let task_deadline = tokio::time::sleep(task_duration);

        let result = tokio::select! {
            result = self.execute_inner(goal, &mut loop_detector, &timeout_ctx, start) => result,
            _ = task_deadline => {
                timeout_ctx.cancel_task();
                Err(PlanError::Timeout {
                    level: "task".to_string(),
                    duration_ms: task_duration.as_millis() as u64,
                })
            }
        };

        // Record final span attributes.
        let elapsed = start.elapsed();
        let span = tracing::Span::current();
        span.record("task.duration_ms", elapsed.as_millis() as u64);
        match &result {
            Ok(r) => {
                span.record("task.status", "ok");
                span.record("task.iterations", r.iterations as u64);
                span.record("task.replans", r.total_replans as u64);
            }
            Err(_) => {
                span.record("task.status", "error");
            }
        }

        // Record OTel metrics (L5-03).
        if let Some(ref m) = self.metrics {
            m.task_duration_seconds.record(elapsed.as_secs_f64(), &[]);
        }

        result
    }

    /// Inner execution logic, wrapped by timeout in `execute()`.
    async fn execute_inner(
        &self,
        goal: &str,
        loop_detector: &mut LoopDetector,
        _timeout_ctx: &TimeoutContext,
        start: Instant,
    ) -> Result<TaskResult, PlanError> {
        let mut stm = ShortTermMemory::new(self.config.stm_token_budget);
        let mut steps: Vec<TaskStep> = Vec::new();
        let mut total_replans: usize = 0;

        // Push the goal as the initial user message.
        stm.push(Message::new(Role::User, goal, estimate_tokens(goal)));

        // Query long-term memory for relevant context.
        let memory_context = self.query_memory_context(goal).await;

        for iteration in 0..self.config.max_iterations {
            let step_span = info_span!(
                "plan_step",
                step.index = iteration,
                step.action = tracing::field::Empty,
            );
            let _step_guard = step_span.enter();

            // Record OTel metric (L5-03).
            if let Some(ref m) = self.metrics {
                m.plan_steps_total.add(1, &[]);
            }

            debug!(iteration, "ReAct loop iteration");

            // Compress STM if nearing the token budget.
            if stm.needs_summarization() {
                self.maybe_summarize(&mut stm).await?;
            }

            // 1. Call /reason.
            let reason_request = build_reason_request(&stm, &self.registry, goal, &memory_context);
            let reason_response = self.ipc.reason(&reason_request).await?;

            // Record LLM cost metrics (L5-04).
            if let Some(ref m) = self.metrics {
                if let Some(ref usage) = reason_response.usage {
                    let attrs = &[opentelemetry::KeyValue::new("model", usage.model.clone())];
                    m.llm_tokens_total.add(usage.tokens_in + usage.tokens_out, attrs);
                    m.llm_cost_usd_total.add(usage.cost_usd, attrs);
                }
            }

            // Push the thought as an assistant message.
            if !reason_response.thought.is_empty() {
                stm.push(Message::new(
                    Role::Assistant,
                    &reason_response.thought,
                    estimate_tokens(&reason_response.thought),
                ));
            }

            // 2. Check if the task is complete.
            if reason_response.is_final {
                info!(
                    iterations = iteration + 1,
                    total_replans, "Task completed with final answer"
                );
                let result = TaskResult {
                    outcome: Outcome::Success,
                    final_answer: reason_response.final_answer,
                    steps,
                    duration_ms: start.elapsed().as_millis() as i64,
                    iterations: iteration + 1,
                    total_replans,
                };
                self.record_episode(goal, &result).await;
                return Ok(result);
            }

            // 3. Dispatch tool (with replan on failure).
            let action = reason_response.action.unwrap_or_default();
            tracing::Span::current().record("step.action", &action.as_str());
            if action.is_empty() {
                error!("LLM returned is_final=false with no action");
                return Err(PlanError::Ipc(crate::ipc::IpcError::ServerError {
                    status: 0,
                    body: "action is empty but is_final is false".to_string(),
                }));
            }
            let action_input = reason_response.action_input;

            let step_outcome = self
                .execute_tool_with_replan(
                    &action,
                    &action_input,
                    &mut stm,
                    &mut steps,
                    &mut total_replans,
                    goal,
                    &memory_context,
                    loop_detector,
                )
                .await?;

            // 4. Handle the step outcome.
            match step_outcome {
                StepOutcome::Observation(result) => {
                    let observation = format_observation(&action, &result);
                    stm.push(Message::new(
                        Role::User,
                        &observation,
                        estimate_tokens(&observation),
                    ));
                }
                StepOutcome::FinalAnswer { thought, answer } => {
                    info!(
                        iterations = iteration + 1,
                        total_replans, "Task completed via replan final answer"
                    );
                    if !thought.is_empty() {
                        stm.push(Message::new(
                            Role::Assistant,
                            &thought,
                            estimate_tokens(&thought),
                        ));
                    }
                    let result = TaskResult {
                        outcome: Outcome::Success,
                        final_answer: answer,
                        steps,
                        duration_ms: start.elapsed().as_millis() as i64,
                        iterations: iteration + 1,
                        total_replans,
                    };
                    self.record_episode(goal, &result).await;
                    return Ok(result);
                }
            }
        }

        // Max iterations reached without completing.
        warn!(
            max_iterations = self.config.max_iterations,
            "ReAct loop exhausted"
        );
        let result = TaskResult {
            outcome: Outcome::Failure,
            final_answer: None,
            steps,
            duration_ms: start.elapsed().as_millis() as i64,
            iterations: self.config.max_iterations,
            total_replans,
        };
        self.record_episode(goal, &result).await;
        Err(PlanError::MaxIterationsReached(self.config.max_iterations))
    }

    // -----------------------------------------------------------------------
    // Parallel execution (L3-04 skeleton)
    // -----------------------------------------------------------------------

    /// Execute independent subtasks in parallel using a `JoinSet`.
    ///
    /// Each subtask gets its own `ShortTermMemory`. Results are collected
    /// after all branches complete. Uses errgroup semantics: if any branch
    /// fails, the error is propagated.
    ///
    /// **Not yet wired into the main ReAct loop.** The LLM would signal
    /// parallel work via `action: "parallel_plan"` with
    /// `action_input: {"subtasks": [...]}`, but this convention is not
    /// enforced yet. This method is provided for future integration.
    #[allow(dead_code)]
    pub async fn execute_parallel(
        &self,
        subtasks: Vec<String>,
    ) -> Result<Vec<TaskResult>, PlanError>
    where
        T: 'static,
    {
        let mut join_set = tokio::task::JoinSet::new();

        for subtask in subtasks {
            let ipc = self.ipc.clone();
            let registry = self.registry.clone();
            let timeout_config = self.timeout_config.clone();
            let config = self.config.clone();

            join_set.spawn(async move {
                let branch_executor = PlanExecutor::new(ipc, registry, timeout_config, config);
                branch_executor.execute(&subtask).await
            });
        }

        let mut results = Vec::new();
        while let Some(join_result) = join_set.join_next().await {
            let task_result = join_result??;
            results.push(task_result);
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Replanning
    // -----------------------------------------------------------------------

    /// Execute a tool, retrying via replan on failure.
    ///
    /// On tool error: record the failure, append error context to STM,
    /// re-call `/reason` to get an alternative action, and retry — up to
    /// the configured replan limits.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_with_replan(
        &self,
        initial_action: &str,
        initial_input: &Value,
        stm: &mut ShortTermMemory,
        steps: &mut Vec<TaskStep>,
        total_replans: &mut usize,
        goal: &str,
        memory_context: &[String],
        loop_detector: &mut LoopDetector,
    ) -> Result<StepOutcome, PlanError> {
        let mut action = initial_action.to_string();
        let mut input = initial_input.clone();
        let mut step_replans: usize = 0;

        loop {
            // Check for loops before executing (L4-05).
            if let Some(loop_msg) = loop_detector.is_looping() {
                warn!("Loop detected: {}", loop_msg);
                return Err(PlanError::Loop(loop_msg));
            }

            let timeout = self.timeout_config.get_timeout(&action);

            // Record this tool call in the loop detector (L4-05).
            loop_detector.record(&action, &input);

            match self
                .registry
                .execute_with_timeout(&action, input.clone(), timeout)
                .await
            {
                Ok(result) => {
                    steps.push(TaskStep::success(&action, input));
                    return Ok(StepOutcome::Observation(result));
                }
                Err(tool_error) => {
                    let error_msg = tool_error.to_string();
                    steps.push(TaskStep::failure(&action, input.clone(), &error_msg));

                    step_replans += 1;
                    *total_replans += 1;

                    // Record OTel metric (L5-03).
                    if let Some(ref m) = self.metrics {
                        m.plan_replans_total.add(1, &[]);
                    }

                    // Check per-step limit.
                    if step_replans >= self.config.replan.max_replans_per_step {
                        return Err(PlanError::MaxReplansPerStep(step_replans, action));
                    }
                    // Check per-task limit.
                    if *total_replans >= self.config.replan.max_replans_per_task {
                        return Err(PlanError::MaxReplansPerTask(*total_replans));
                    }

                    // Append error context to STM.
                    let error_context = format!(
                        "Tool '{}' failed with error: {}. Please try an alternative approach.",
                        action, error_msg
                    );
                    stm.push(Message::new(
                        Role::User,
                        &error_context,
                        estimate_tokens(&error_context),
                    ));

                    warn!(
                        action = %action,
                        error = %error_msg,
                        step_replans,
                        total_replans = *total_replans,
                        "Replanning after tool failure"
                    );

                    // Re-call /reason with the updated context.
                    let reason_request =
                        build_reason_request(stm, &self.registry, goal, memory_context);
                    let response = self.ipc.reason(&reason_request).await?;

                    // Push the new thought.
                    if !response.thought.is_empty() {
                        stm.push(Message::new(
                            Role::Assistant,
                            &response.thought,
                            estimate_tokens(&response.thought),
                        ));
                    }

                    // If LLM declares done after seeing the error, propagate.
                    if response.is_final {
                        return Ok(StepOutcome::FinalAnswer {
                            thought: response.thought,
                            answer: response.final_answer,
                        });
                    }

                    // Extract the new action for the next attempt.
                    action = response.action.unwrap_or_default();
                    if action.is_empty() {
                        return Err(PlanError::Ipc(crate::ipc::IpcError::ServerError {
                            status: 0,
                            body: "replan returned no action and is_final=false".to_string(),
                        }));
                    }
                    input = response.action_input;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Memory helpers
    // -----------------------------------------------------------------------

    /// Query long-term memory for context relevant to the goal.
    ///
    /// Returns an empty vec if long-term memory is not attached (unit tests)
    /// or if the embedding/query fails (non-fatal).
    async fn query_memory_context(&self, _goal: &str) -> Vec<String> {
        // TODO: call self.ipc.embed(goal) → self.long_term_memory.query(embedding)
        // For now, return empty — wiring this up requires the full runtime
        // integration and is not blocking the core ReAct loop.
        Vec::new()
    }

    /// Compress STM via the `/summarize` endpoint when nearing the token budget.
    async fn maybe_summarize(&self, stm: &mut ShortTermMemory) -> Result<(), PlanError> {
        let messages: Vec<Value> = stm
            .messages()
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role.to_string(),
                    "content": m.content,
                    "token_count": m.token_count
                })
            })
            .collect();

        let request = SummarizeRequest {
            messages,
            model: None,
            max_summary_tokens: None,
        };

        let response = self.ipc.summarize(&request).await?;
        stm.replace_with_summary(Message::new(
            Role::System,
            &response.summary,
            response.token_count as usize,
        ));

        debug!(
            token_count = response.token_count,
            "STM compressed via summarisation"
        );
        Ok(())
    }

    /// Record a completed task in the episodic log (non-fatal on error).
    async fn record_episode(&self, goal: &str, result: &TaskResult) {
        let Some(ref log) = self.episodic_log else {
            return;
        };

        let error_msg;
        let error = if result.outcome == Outcome::Failure {
            error_msg = "task did not complete successfully".to_string();
            Some(error_msg.as_str())
        } else {
            None
        };

        let episode = EpisodeInput {
            task_goal: goal,
            steps: &result.steps,
            outcome: result.outcome.clone(),
            error,
            duration_ms: result.duration_ms,
        };
        if let Err(e) = log.insert(episode).await {
            error!(error = %e, "Failed to record episode — non-fatal, continuing");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough token estimate for Rust-side messages.
///
/// Accurate counting happens in Python via tiktoken. This heuristic
/// (≈ 4 chars per token for English text) is used only for STM budget
/// tracking within a single task.
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Format a tool observation for inclusion in short-term memory.
fn format_observation(action: &str, result: &Value) -> String {
    let result_str = serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());

    // Truncate very long observations to prevent blowing the context window.
    let truncated = if result_str.len() > 4000 {
        format!(
            "{}... [truncated, {} chars total]",
            &result_str[..4000],
            result_str.len()
        )
    } else {
        result_str
    };
    format!("[Observation from {action}]:\n{truncated}")
}

/// Build a `ReasonRequest` from the current short-term memory state.
fn build_reason_request(
    stm: &ShortTermMemory,
    registry: &ToolRegistry,
    goal: &str,
    memory_context: &[String],
) -> ReasonRequest {
    let messages: Vec<IpcMessage> = stm
        .messages()
        .iter()
        .map(|m| IpcMessage {
            role: m.role.to_string(),
            content: m.content.clone(),
        })
        .collect();

    ReasonRequest {
        messages,
        tool_schemas: registry.list_schemas(),
        memory_context: memory_context.to_vec(),
        task_goal: goal.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{EmbedRequest, EmbedResponse, IpcError, SummarizeResponse};
    use crate::tools::mock::MockTool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Mock IPC transport that returns pre-scripted responses.
    struct MockIpc {
        responses: Mutex<Vec<crate::ipc::ReasonResponse>>,
        reason_call_count: AtomicUsize,
    }

    impl MockIpc {
        fn new(responses: Vec<crate::ipc::ReasonResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                reason_call_count: AtomicUsize::new(0),
            }
        }

        fn reason_calls(&self) -> usize {
            self.reason_call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl IpcTransport for MockIpc {
        async fn reason(
            &self,
            _req: &ReasonRequest,
        ) -> Result<crate::ipc::ReasonResponse, IpcError> {
            self.reason_call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(crate::ipc::ReasonResponse {
                    thought: "Default done.".to_string(),
                    action: None,
                    action_input: serde_json::json!({}),
                    is_final: true,
                    final_answer: Some("Default final answer".to_string()),
                    usage: None,
                })
            } else {
                Ok(responses.remove(0))
            }
        }

        async fn embed(&self, _req: &EmbedRequest) -> Result<EmbedResponse, IpcError> {
            Ok(EmbedResponse {
                vector: vec![0.0; 1536],
                token_count: 5,
                cost_usd: None,
            })
        }

        async fn summarize(&self, _req: &SummarizeRequest) -> Result<SummarizeResponse, IpcError> {
            Ok(SummarizeResponse {
                summary: "Summary of conversation.".to_string(),
                token_count: 10,
            })
        }
    }

    fn make_reason_response(
        thought: &str,
        action: Option<&str>,
        action_input: Value,
        is_final: bool,
        final_answer: Option<&str>,
    ) -> crate::ipc::ReasonResponse {
        crate::ipc::ReasonResponse {
            thought: thought.to_string(),
            action: action.map(|s| s.to_string()),
            action_input,
            is_final,
            final_answer: final_answer.map(|s| s.to_string()),
            usage: None,
        }
    }

    fn make_executor(ipc: Arc<MockIpc>) -> PlanExecutor<MockIpc> {
        let mut registry = ToolRegistry::default();
        registry
            .register(Arc::new(MockTool))
            .expect("register mock tool");
        PlanExecutor::new(
            ipc,
            Arc::new(registry),
            ToolTimeoutConfig::default(),
            PlanConfig {
                max_iterations: 10,
                stm_token_budget: 8000,
                ..Default::default()
            },
        )
    }

    // -----------------------------------------------------------------------
    // Core loop tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_executor_immediate_final_answer() {
        let ipc = Arc::new(MockIpc::new(vec![make_reason_response(
            "I know the answer.",
            None,
            serde_json::json!({}),
            true,
            Some("42"),
        )]));

        let executor = make_executor(ipc.clone());
        let result = executor.execute("What is the answer?").await.unwrap();

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.final_answer.as_deref(), Some("42"));
        assert_eq!(result.iterations, 1);
        assert!(result.steps.is_empty());
        assert_eq!(ipc.reason_calls(), 1);
    }

    #[tokio::test]
    async fn test_executor_two_step_task() {
        let ipc = Arc::new(MockIpc::new(vec![
            // Step 1: call mock_echo
            make_reason_response(
                "I should echo a message.",
                Some("mock_echo"),
                serde_json::json!({"message": "hello"}),
                false,
                None,
            ),
            // Step 2: final answer
            make_reason_response(
                "Got the echo. Done.",
                None,
                serde_json::json!({}),
                true,
                Some("echoed: hello"),
            ),
        ]));

        let executor = make_executor(ipc.clone());
        let result = executor.execute("Echo hello").await.unwrap();

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.final_answer.as_deref(), Some("echoed: hello"));
        assert_eq!(result.iterations, 2);
        assert_eq!(result.steps.len(), 1);
        assert!(result.steps[0].success);
        assert_eq!(result.steps[0].action, "mock_echo");
        assert_eq!(ipc.reason_calls(), 2);
    }

    #[tokio::test]
    async fn test_executor_max_iterations() {
        // LLM always wants to call a tool — never returns is_final.
        // Use different messages each time to avoid loop detection.
        let responses: Vec<_> = (0..10)
            .map(|i| {
                make_reason_response(
                    "Still working...",
                    Some("mock_echo"),
                    serde_json::json!({"message": format!("iteration {}", i)}),
                    false,
                    None,
                )
            })
            .collect();

        let ipc = Arc::new(MockIpc::new(responses));
        let executor = make_executor(ipc);
        let err = executor.execute("Never-ending task").await.unwrap_err();

        match err {
            PlanError::MaxIterationsReached(n) => assert_eq!(n, 10),
            other => panic!("Expected MaxIterationsReached, got: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // Replan tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_executor_replan_on_tool_error() {
        let ipc = Arc::new(MockIpc::new(vec![
            // Step 1: call a tool that doesn't exist → ToolError::NotFound.
            make_reason_response(
                "Try nonexistent tool.",
                Some("does_not_exist"),
                serde_json::json!({}),
                false,
                None,
            ),
            // Replan: LLM tries mock_echo instead.
            make_reason_response(
                "Let me try mock_echo instead.",
                Some("mock_echo"),
                serde_json::json!({"message": "fallback"}),
                false,
                None,
            ),
            // Final answer.
            make_reason_response(
                "Done.",
                None,
                serde_json::json!({}),
                true,
                Some("Recovered successfully"),
            ),
        ]));

        let executor = make_executor(ipc.clone());
        let result = executor.execute("Test replan").await.unwrap();

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.total_replans, 1);
        // Steps: 1 failure (does_not_exist) + 1 success (mock_echo)
        assert_eq!(result.steps.len(), 2);
        assert!(!result.steps[0].success);
        assert!(result.steps[1].success);
    }

    #[tokio::test]
    async fn test_executor_max_replans_per_step() {
        // Always try a nonexistent tool — replans keep failing.
        let responses: Vec<_> = (0..5)
            .map(|_| {
                make_reason_response(
                    "Try again.",
                    Some("does_not_exist"),
                    serde_json::json!({}),
                    false,
                    None,
                )
            })
            .collect();

        let ipc = Arc::new(MockIpc::new(responses));
        let mut executor = make_executor(ipc);
        executor.config.replan.max_replans_per_step = 2;

        let err = executor.execute("Failing task").await.unwrap_err();

        match err {
            PlanError::MaxReplansPerStep(n, action) => {
                assert_eq!(n, 2);
                assert_eq!(action, "does_not_exist");
            }
            other => panic!("Expected MaxReplansPerStep, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_executor_replan_final_answer_during_replan() {
        let ipc = Arc::new(MockIpc::new(vec![
            // Step 1: call a tool that doesn't exist.
            make_reason_response(
                "Try nonexistent tool.",
                Some("does_not_exist"),
                serde_json::json!({}),
                false,
                None,
            ),
            // Replan: LLM decides it's done despite the error.
            make_reason_response(
                "Actually, I can answer without tools.",
                None,
                serde_json::json!({}),
                true,
                Some("Direct answer"),
            ),
        ]));

        let executor = make_executor(ipc);
        let result = executor.execute("Test replan final").await.unwrap();

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.final_answer.as_deref(), Some("Direct answer"));
        assert_eq!(result.total_replans, 1);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_tokens_nonempty() {
        assert_eq!(estimate_tokens("hello world!"), 3); // 12 / 4
    }

    #[test]
    fn test_estimate_tokens_short() {
        assert_eq!(estimate_tokens("hi"), 1); // < 4 chars → max(0,1) = 1
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 1); // 0 / 4 → max(0,1) = 1
    }

    #[test]
    fn test_format_observation_truncates_long_output() {
        let long_value = serde_json::json!({"data": "x".repeat(5000)});
        let obs = format_observation("test_tool", &long_value);
        assert!(obs.contains("[truncated"));
        assert!(obs.len() < 5000);
    }

    #[test]
    fn test_format_observation_short_output() {
        let value = serde_json::json!({"result": "ok"});
        let obs = format_observation("test_tool", &value);
        assert!(obs.contains("[Observation from test_tool]"));
        assert!(obs.contains("ok"));
        assert!(!obs.contains("[truncated"));
    }

    #[test]
    fn test_build_reason_request_includes_goal() {
        let stm = ShortTermMemory::new(1000);
        let registry = ToolRegistry::default();
        let req = build_reason_request(&stm, &registry, "Find papers", &[]);
        assert_eq!(req.task_goal, "Find papers");
    }

    #[test]
    fn test_build_reason_request_includes_memory_context() {
        let stm = ShortTermMemory::new(1000);
        let registry = ToolRegistry::default();
        let ctx = vec!["memory 1".to_string(), "memory 2".to_string()];
        let req = build_reason_request(&stm, &registry, "goal", &ctx);
        assert_eq!(req.memory_context, ctx);
    }

    // -----------------------------------------------------------------------
    // Loop detection tests (L4-05)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_executor_detects_loop() {
        // LLM tries to call the same tool with the same input 5 times.
        let input = serde_json::json!({"message": "repeated"});
        let responses: Vec<_> = (0..5)
            .map(|_| {
                make_reason_response("Try again.", Some("mock_echo"), input.clone(), false, None)
            })
            .collect();

        let ipc = Arc::new(MockIpc::new(responses));
        let mut executor = make_executor(ipc);

        // Set loop detector to trigger after 3 repeats (default)
        executor.loop_detector_config = LoopDetectorConfig {
            window_size: 10,
            repeat_threshold: 3,
        };

        let err = executor.execute("Loop task").await.unwrap_err();

        match err {
            PlanError::Loop(msg) => {
                assert!(msg.contains("3 times"), "Expected '3 times' in: {}", msg);
            }
            other => panic!("Expected Loop error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_executor_no_loop_with_varied_inputs() {
        // Same tool, different inputs each time — should not trigger loop detection.
        let responses: Vec<_> = (0..5)
            .map(|i| {
                make_reason_response(
                    "Next step.",
                    Some("mock_echo"),
                    serde_json::json!({"message": format!("step {}", i)}),
                    false,
                    None,
                )
            })
            .collect();

        // Add final answer to terminate.
        let mut all_responses = responses;
        all_responses.push(make_reason_response(
            "Done.",
            None,
            serde_json::json!({}),
            true,
            Some("Completed"),
        ));

        let ipc = Arc::new(MockIpc::new(all_responses));
        let executor = make_executor(ipc);
        let result = executor.execute("Varied task").await.unwrap();

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.steps.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Timeout tests (L4-06)
    // -----------------------------------------------------------------------

    /// Mock IPC with artificial delay to test timeouts.
    struct SlowMockIpc {
        responses: Mutex<Vec<crate::ipc::ReasonResponse>>,
        delay: std::time::Duration,
    }

    impl SlowMockIpc {
        fn new(responses: Vec<crate::ipc::ReasonResponse>, delay: std::time::Duration) -> Self {
            Self {
                responses: Mutex::new(responses),
                delay,
            }
        }
    }

    #[async_trait::async_trait]
    impl IpcTransport for SlowMockIpc {
        async fn reason(
            &self,
            _req: &ReasonRequest,
        ) -> Result<crate::ipc::ReasonResponse, IpcError> {
            // Simulate slow LLM call.
            tokio::time::sleep(self.delay).await;

            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(crate::ipc::ReasonResponse {
                    thought: "Default done.".to_string(),
                    action: None,
                    action_input: serde_json::json!({}),
                    is_final: true,
                    final_answer: Some("Default final answer".to_string()),
                    usage: None,
                })
            } else {
                Ok(responses.remove(0))
            }
        }

        async fn embed(&self, _req: &EmbedRequest) -> Result<EmbedResponse, IpcError> {
            Ok(EmbedResponse {
                vector: vec![0.0; 1536],
                token_count: 5,
                cost_usd: None,
            })
        }

        async fn summarize(&self, _req: &SummarizeRequest) -> Result<SummarizeResponse, IpcError> {
            Ok(SummarizeResponse {
                summary: "Summary of conversation.".to_string(),
                token_count: 10,
            })
        }
    }

    #[tokio::test]
    async fn test_executor_task_timeout() {
        use std::time::Duration;

        // LLM tries to run multiple steps, but each step takes 50ms.
        let responses: Vec<_> = (0..10)
            .map(|i| {
                make_reason_response(
                    "Working...",
                    Some("mock_echo"),
                    serde_json::json!({"message": format!("step {}", i)}),
                    false,
                    None,
                )
            })
            .collect();

        // Each reason call takes 50ms → 10 iterations = 500ms minimum.
        let ipc = Arc::new(SlowMockIpc::new(responses, Duration::from_millis(50)));

        let mut registry = ToolRegistry::default();
        registry
            .register(Arc::new(MockTool))
            .expect("register mock tool");

        let mut executor = PlanExecutor::new(
            ipc,
            Arc::new(registry),
            ToolTimeoutConfig::default(),
            PlanConfig {
                max_iterations: 20,
                stm_token_budget: 8000,
                ..Default::default()
            },
        );

        // Task timeout is 100ms → should trigger after 2 iterations (2 × 50ms).
        executor.timeout_ctx_config = TimeoutConfig {
            tool_timeout: Duration::from_secs(30),
            step_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_millis(100),
        };

        let err = executor.execute("Long task").await.unwrap_err();

        match err {
            PlanError::Timeout { level, .. } => {
                assert_eq!(level, "task");
            }
            other => panic!("Expected Timeout error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_executor_completes_within_task_timeout() {
        use std::time::Duration;

        // Quick task that completes before timeout.
        let ipc = Arc::new(MockIpc::new(vec![make_reason_response(
            "Done immediately.",
            None,
            serde_json::json!({}),
            true,
            Some("Fast result"),
        )]));

        let mut executor = make_executor(ipc);
        executor.timeout_ctx_config = TimeoutConfig {
            tool_timeout: Duration::from_secs(30),
            step_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_secs(5),
        };

        let result = executor.execute("Fast task").await.unwrap();
        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.final_answer.as_deref(), Some("Fast result"));
    }
}
