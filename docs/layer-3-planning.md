# Layer 3 — Task Planning & Orchestration

A single LLM call is not an agent. An agent **breaks a goal into steps, executes them, observes results, and replans when reality doesn't match expectations.** This layer is where passive LLM output becomes autonomous action.

**Related tasks:** L3-01 through L3-07 in [AGENTS.md](../AGENTS.md)

---

## Core Pattern: ReAct

Xola uses the **ReAct** (Reason + Act) pattern as its execution primitive. Each cycle:

```
┌──────────────────────────────────────────────────────────────┐
│                        ReAct Loop                            │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌───────────┐             │
│   │  Thought │───▶│  Action  │───▶│  Observe  │──┐          │
│   │          │    │          │    │           │  │          │
│   │  "I need │    │  Call    │    │  Record   │  │          │
│   │  to..."  │    │  tool    │    │  result   │  │          │
│   └──────────┘    └──────────┘    └───────────┘  │          │
│        ▲                                          │          │
│        └──────────────────────────────────────────┘          │
│                                                              │
│   Loop until: is_final = true OR max iterations reached      │
└──────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Thought** — The LLM reasons about what to do next given the goal, current context, and past observations
2. **Action** — The LLM selects a tool and provides input parameters
3. **Observation** — The runtime executes the tool and appends the result to context
4. **Repeat** — The cycle continues until the LLM signals a final answer or the iteration limit is hit

### The `/reason` Endpoint

The Python LLM surface handles prompt construction and ReAct parsing:

```
Rust sends:
  - messages (conversation history)
  - tool_schemas (available tools)
  - memory_context (relevant long-term memories)
  - task_goal (what the agent is trying to accomplish)

Python returns:
  - thought (LLM's reasoning)
  - action (tool name, or null if done)
  - action_input (tool parameters)
  - is_final (true when task is complete)
  - final_answer (the result, when is_final = true)
```

See [IPC Contract](ipc-contract.md) for the full schema.

---

## Plan Execution

### Sequential Execution

The `PlanExecutor` runs actions in sequence, feeding each observation back into the next reasoning call:

```rust
pub struct PlanExecutor {
    registry: Arc<ToolRegistry>,
    memory: Arc<ShortTermMemory>,
    ipc_client: Arc<IpcClient>,
    max_iterations: usize,
}

impl PlanExecutor {
    pub async fn execute(&self, goal: &str) -> Result<TaskResult> {
        for iteration in 0..self.max_iterations {
            // 1. Call /reason with current context
            let action = self.ipc_client.reason(&self.build_context(goal)).await?;

            // 2. Check if done
            if action.is_final {
                return Ok(TaskResult::success(action.final_answer));
            }

            // 3. Dispatch tool
            let observation = self.registry.dispatch(
                &action.action,
                action.action_input,
                self.tool_timeout,
            ).await;

            // 4. Append observation to memory
            self.memory.push(Message::observation(observation));
        }

        Err(PlanError::MaxIterationsReached)
    }
}
```

### Parallel Execution

When a task has independent subtasks, the runtime fans them out across `tokio` tasks and joins results:

```rust
// Fan-out independent subtasks
let mut join_set = JoinSet::new();

for subtask in plan.parallel_branches() {
    let registry = self.registry.clone();
    let client = self.ipc_client.clone();

    join_set.spawn(async move {
        execute_subtask(subtask, registry, client).await
    });
}

// Fan-in results
let mut results = Vec::new();
while let Some(result) = join_set.join_next().await {
    results.push(result??);
}
```

**Key constraints:**
- Only truly independent subtasks run in parallel
- Each parallel branch has its own short-term memory fork
- Results are merged back into the parent context after all branches complete
- `errgroup` semantics: if any branch fails, cancellation propagates

---

## Replanning

Real tasks don't go according to plan. Tools fail, return unexpected data, or produce results that invalidate the original approach. The runtime handles this through **replanning triggers**.

### When Replanning Fires

| Trigger | Example |
|---------|---------|
| Tool execution error | `web_search` returns HTTP 429 (rate limited) |
| Unexpected output | Search returns no results when results were expected |
| Subtask failure | One branch of a parallel plan fails |
| Context change | New information invalidates a previous assumption |

### How It Works

1. Tool call fails or returns unexpected result
2. Error context is appended to the conversation: `"Tool 'web_search' failed with: rate limit exceeded"`
3. Runtime calls `/reason` again with the failure context included
4. LLM generates a revised plan (e.g., try a different search query, use an alternative approach)
5. Execution continues with the revised action

### Replan Limits

Replanning is bounded:

```rust
pub struct ReplanConfig {
    pub max_replans_per_step: usize,  // Default: 3
    pub max_replans_per_task: usize,  // Default: 10
}
```

After exceeding the limit, the runtime escalates to an error rather than looping forever. This interacts with the [Loop Detector](layer-4-reliability.md) in Layer 4.

---

## End-to-End Example

Goal: *"Find the three most cited papers on RAG published in 2024 and summarize their key contributions."*

```
Turn 1:
  Thought: I need to search for highly cited RAG papers from 2024.
  Action: web_search
  Input:  { "query": "most cited RAG papers 2024", "num_results": 10 }

Turn 2:
  Observation: [10 search results with titles, URLs, snippets]
  Thought: I found several candidates. Let me fetch the top 3 for details.
  Action: url_fetch
  Input:  { "url": "https://arxiv.org/abs/..." }

Turn 3:
  Observation: [Full paper abstract and metadata]
  Thought: Got the first paper. Fetching the second.
  Action: url_fetch
  Input:  { "url": "https://arxiv.org/abs/..." }

Turn 4:
  Observation: [Paper content]
  Thought: Two down, one to go.
  Action: url_fetch
  Input:  { "url": "https://arxiv.org/abs/..." }

Turn 5:
  Observation: [Error: timeout after 30s]
  → REPLAN triggered
  Thought: The third URL timed out. Let me try a cached version.
  Action: web_search
  Input:  { "query": "site:semanticscholar.org <paper title>" }

Turn 6:
  Observation: [Alternative URL found]
  Action: url_fetch
  Input:  { "url": "https://semanticscholar.org/..." }

Turn 7:
  Observation: [Paper content retrieved]
  Thought: I now have all three papers. Composing the summary.
  is_final: true
  final_answer: "The three most cited RAG papers from 2024 are..."
```

---

## What You'll Learn

Building this layer teaches:

- **The ReAct pattern** — the standard architecture for LLM-driven agents
- **DAG-based task execution** — representing plans as graphs with dependencies
- **Fan-out/fan-in** — parallel execution with `JoinSet` and structured error handling
- **Stateful execution loops** — maintaining and evolving state across iterations
- **Graceful degradation** — replanning when the original approach fails

---

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Unit tests | ReAct parser extracts thought/action/observation correctly | `llm_surface/` |
| Unit tests | PlanExecutor iteration logic, max iteration enforcement | `runtime/src/planning/` |
| Integration tests | Full multi-step research task end-to-end | `tests/` |
| Integration tests | Replan trigger on injected tool failure | `tests/` |
