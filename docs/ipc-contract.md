# IPC Contract Reference

This document is the **canonical specification** for the Rust ↔ Python inter-process communication boundary. Both sides of the boundary must conform to these schemas exactly.

> **⚠️ Immutable without approval.** Changing any endpoint path, request field, or response field requires a human-approved `PROPOSAL.md`. See [Contributing](contributing.md) for the process.

---

## Transport

| Property | Value |
|----------|-------|
| **Protocol** | HTTP/1.1 over Unix socket (default) or gRPC |
| **Default socket path** | `/tmp/agent.sock` |
| **Content-Type** | `application/json` |
| **Direction** | Rust → Python only. Python never calls into Rust. |
| **Server** | FastAPI + Uvicorn (Python) |
| **Client** | `reqwest` with Unix socket transport (Rust) |

---

## Endpoints

### `POST /reason`

The core reasoning endpoint. Called on every planning iteration of the ReAct loop.

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "tool_schemas": [
    {
      "name": "string",
      "description": "string",
      "parameters": {}
    }
  ],
  "memory_context": ["string"],
  "task_goal": "string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `array[Message]` | Conversation history (short-term memory window) |
| `messages[].role` | `"user" \| "assistant"` | Message role |
| `messages[].content` | `string` | Message content |
| `tool_schemas` | `array[ToolSchema]` | Available tools with JSON Schema descriptors |
| `memory_context` | `array[string]` | Relevant long-term memories for this turn |
| `task_goal` | `string` | The high-level goal the agent is working toward |

**Response:**

```json
{
  "thought": "string",
  "action": "string | null",
  "action_input": {},
  "is_final": false,
  "final_answer": "string | null"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `thought` | `string` | The LLM's reasoning about what to do next |
| `action` | `string \| null` | Tool name to call. Null when `is_final` is true. |
| `action_input` | `object` | Parameters for the tool call |
| `is_final` | `boolean` | `true` when the agent has completed the task |
| `final_answer` | `string \| null` | The final result. Present only when `is_final` is true. |

---

### `POST /embed`

Generates an embedding vector for a text string. Used by long-term memory for writes and similarity queries.

**Request:**

```json
{
  "text": "string",
  "model": "text-embedding-3-small"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Text to embed |
| `model` | `string` | Embedding model name. Default: `text-embedding-3-small` |

**Response:**

```json
{
  "vector": [0.0, 0.1, -0.05, ...],
  "token_count": 42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `vector` | `array[float]` | Embedding vector (1536 dimensions for `text-embedding-3-small`) |
| `token_count` | `integer` | Number of tokens in the input text |

---

### `POST /parse`

Validates raw LLM output against a target schema. Used for structured output enforcement (Layer 4).

**Request:**

```json
{
  "raw": "string",
  "schema": {},
  "attempt": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `raw` | `string` | Raw LLM output to validate |
| `schema` | `object` | Target JSON Schema or Pydantic model schema |
| `attempt` | `integer` | Current attempt number (1-indexed). Used for corrective prompting. |

**Response:**

```json
{
  "parsed": {},
  "success": true,
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `parsed` | `object \| null` | The validated, parsed output. Null on failure. |
| `success` | `boolean` | Whether parsing succeeded |
| `error` | `string \| null` | Validation error message. Null on success. |

---

## Error Handling

All endpoints return standard HTTP status codes:

| Status | Meaning | Rust action |
|--------|---------|-------------|
| `200` | Success | Process response normally |
| `400` | Bad request (malformed input from Rust) | Bug in Rust client — log and fix |
| `422` | Validation error (Pydantic rejected input) | Check request schema conformance |
| `500` | Internal server error (Python crash) | Retry with backoff, then escalate |
| `503` | Service unavailable (overloaded) | Retry with backoff |

---

## Contract Rules

1. All request and response bodies use JSON with `Content-Type: application/json`
2. Python server is **stateless** between requests — all state lives in Rust / Postgres
3. Data flows **one way**: Rust → Python. Python never initiates calls to Rust.
4. All fields shown above are **required** unless explicitly marked optional
5. Adding new fields to request/response bodies requires a `PROPOSAL.md`
6. Adding new endpoints requires a `PROPOSAL.md`
7. Removing or renaming any field is a **breaking change** requiring human approval
