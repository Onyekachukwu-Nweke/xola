"""System prompt builder and tool schema converter for the /reason endpoint.

The system prompt instructs the LLM to act as an autonomous agent using the
ReAct pattern: reason about the current state, select a tool, observe the
result, and repeat until the task is complete.

A synthetic ``final_answer`` tool is injected so the LLM can signal task
completion through the same function-calling mechanism as regular tool use.

Prompt templates are currently hardcoded.
TODO: load from ``config/prompts.toml`` so they can be tuned without code
changes (see AGENTS.md escalation rules for structural prompt changes).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Chat model used for /reason calls. Separate from embedding/summarisation models.
REASON_MODEL: str = "gpt-4o"

# ---------------------------------------------------------------------------
# Synthetic tool injected into every /reason call
# ---------------------------------------------------------------------------

FINAL_ANSWER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": (
            "Call this tool when the task is complete and you have the final "
            "result. Pass the result as the 'answer' argument."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the task.",
                },
            },
            "required": ["answer"],
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE: str = """\
You are an autonomous agent. Your goal is:

{task_goal}

You have access to tools. Use them to accomplish your goal.
When you have gathered enough information to answer, call the `final_answer` tool.

## Relevant context from memory

{memory_block}

## Instructions

- Think step by step about what action to take next.
- Use exactly one tool per turn.
- When you have the final answer, use the `final_answer` tool.
- If a tool call fails, analyse the error and try an alternative approach.
- Do not repeat the same failing action more than twice."""


def build_system_prompt(
    task_goal: str,
    memory_context: list[str],
) -> str:
    """Build the system prompt for a single ReAct reasoning turn.

    Args:
        task_goal: The high-level goal the agent is working toward.
        memory_context: Relevant long-term memories retrieved for this turn.

    Returns:
        The fully rendered system prompt string.
    """
    if memory_context:
        memory_block = "\n".join(f"- {m}" for m in memory_context)
    else:
        memory_block = "No relevant memories."

    return _SYSTEM_PROMPT_TEMPLATE.format(
        task_goal=task_goal,
        memory_block=memory_block,
    )


def build_tool_schemas_for_openai(
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Xola tool schemas to the OpenAI ``tools`` parameter format.

    Xola format (from ``ToolRegistry::list_schemas()``)::

        [{"name": "...", "description": "...", "parameters": {...}}, ...]

    OpenAI format::

        [{"type": "function", "function": {"name": "...", ...}}, ...]

    The synthetic ``final_answer`` tool is always appended.

    Args:
        tool_schemas: Tool descriptors in Xola format.

    Returns:
        Tool descriptors in OpenAI function-calling format, including
        the ``final_answer`` tool.
    """
    openai_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            },
        }
        for schema in tool_schemas
    ]
    openai_tools.append(FINAL_ANSWER_TOOL)
    return openai_tools
