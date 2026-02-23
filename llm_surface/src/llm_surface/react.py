"""ReAct reasoning step — call the LLM and parse its response.

This module handles the actual OpenAI API call for the ``/reason`` endpoint
and converts the response into the IPC contract format defined in AGENTS.md::

    {thought, action, action_input, is_final, final_answer}

The primary path uses OpenAI function/tool calling for structured output.
A fallback handles cases where the model returns plain text instead of a
tool call (e.g. refusal, safety filter, or when it has a direct answer).
"""

import json
import logging
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from llm_surface.prompts import (
    REASON_MODEL,
    build_system_prompt,
    build_tool_schemas_for_openai,
)

logger = logging.getLogger(__name__)


async def reason_step(
    messages: list[dict[str, str]],
    tool_schemas: list[dict[str, Any]],
    memory_context: list[str],
    task_goal: str,
    *,
    client: AsyncOpenAI,
    model: str = REASON_MODEL,
) -> dict[str, Any]:
    """Execute one ReAct reasoning step via OpenAI function calling.

    Constructs the chat completion request from the conversation history,
    available tools, memory context, and task goal. Parses the response
    into the IPC contract format.

    Args:
        messages:       Conversation history from short-term memory.
        tool_schemas:   Available tools in Xola format (from ToolRegistry).
        memory_context: Relevant long-term memories for this turn.
        task_goal:      The high-level task goal.
        client:         Injected ``AsyncOpenAI`` client.
        model:          Chat model to use for reasoning.

    Returns:
        Dict matching the ``ReasonResponse`` schema::

            {"thought": str, "action": str | None, "action_input": dict,
             "is_final": bool, "final_answer": str | None}

    Raises:
        openai.APIError: On LLM API failure.
        ValueError:      On an unparseable response (empty content and no
                         tool call).
    """
    system_prompt = build_system_prompt(task_goal, memory_context)
    openai_tools = build_tool_schemas_for_openai(tool_schemas)

    # Build the message list: system prompt + conversation history.
    chat_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
    ]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # OpenAI accepts "system", "user", "assistant".
        chat_messages.append({"role": role, "content": content})  # type: ignore[arg-type,misc]

    # Call the LLM with function calling enabled.
    typed_tools = cast(list[ChatCompletionToolParam], openai_tools)
    response = await client.chat.completions.create(
        model=model,
        messages=chat_messages,
        tools=typed_tools,
        tool_choice="auto",
    )

    choice = response.choices[0]
    message = choice.message
    thought = (message.content or "").strip()

    # Log the raw response for debugging (AGENTS.md: never swallow failures silently).
    logger.debug(
        "Raw /reason response: content=%r, tool_calls=%r",
        message.content,
        message.tool_calls,
    )

    # --- Path 1: Model made a tool call ---
    if message.tool_calls:
        tool_call = message.tool_calls[0]

        # We only use standard function tool calls. The OpenAI SDK's type
        # union includes CustomToolCall which lacks .function; guard here.
        if not hasattr(tool_call, "function"):
            raise ValueError(f"Unexpected tool call type: {type(tool_call)}")
        func = tool_call.function

        try:
            arguments: dict[str, Any] = (
                json.loads(func.arguments) if func.arguments else {}
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "Malformed tool_call arguments from LLM: name=%s args=%r",
                func.name,
                func.arguments,
            )
            raise ValueError(
                f"Malformed tool call arguments for '{func.name}': {exc}"
            ) from exc

        # Synthetic final_answer tool → mark task as complete.
        if func.name == "final_answer":
            return {
                "thought": thought,
                "action": None,
                "action_input": {},
                "is_final": True,
                "final_answer": arguments.get("answer", ""),
            }

        # Regular tool call.
        return {
            "thought": thought,
            "action": func.name,
            "action_input": arguments,
            "is_final": False,
            "final_answer": None,
        }

    # --- Path 2: No tool call (plain text response) ---
    if thought:
        # The model chose not to use any tool. Treat as a final answer.
        logger.warning(
            "LLM returned plain text without a tool call — treating as final answer. "
            "content=%r",
            thought,
        )
        return {
            "thought": thought,
            "action": None,
            "action_input": {},
            "is_final": True,
            "final_answer": thought,
        }

    # --- Path 3: Empty response (safety filter or malformed) ---
    logger.error("LLM returned empty content and no tool call.")
    raise ValueError(
        "LLM returned an empty response with no tool call. "
        "This may indicate a safety filter or malformed request."
    )
