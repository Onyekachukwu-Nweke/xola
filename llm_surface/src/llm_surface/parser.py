"""LLM output parsing and validation with corrective re-prompting.

This module provides structured output enforcement for LLM responses. When the
LLM returns malformed JSON or output that doesn't match the expected schema,
the parser validates against a Pydantic schema and optionally retries with
corrective prompts.

Related tasks: L4-03, L4-04 in AGENTS.md
"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, create_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models (L4-03)
# ---------------------------------------------------------------------------


class ParseRequest(BaseModel):
    """Request body for POST /parse endpoint.

    Matches the IPC contract in AGENTS.md:
        {"raw": "string", "schema": {}, "attempt": 1}
    """

    model_config = {"protected_namespaces": ()}  # Allow 'schema' field name

    raw: str = Field(..., description="Raw LLM output to parse.")
    schema: dict[str, Any] = Field(..., description="JSON Schema to validate against.")
    attempt: int = Field(default=1, ge=1, description="Attempt number (for logging).")


class ParseResponse(BaseModel):
    """Response from POST /parse endpoint.

    Matches the IPC contract in AGENTS.md:
        {"parsed": {}, "success": true, "error": null}
    """

    parsed: dict[str, Any] | None = Field(
        default=None, description="Successfully parsed JSON object."
    )
    success: bool = Field(..., description="Whether parsing succeeded.")
    error: str | None = Field(default=None, description="Error message if parsing failed.")


# ---------------------------------------------------------------------------
# Schema validation (L4-03)
# ---------------------------------------------------------------------------


def create_model_from_schema(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> type[BaseModel]:
    """Create a Pydantic model dynamically from a JSON Schema.

    This is a simplified converter that handles common JSON Schema patterns.
    For production use, consider using a library like `pydantic-json-schema`
    or implementing a full JSON Schema → Pydantic converter.

    Parameters
    ----------
    schema : dict
        JSON Schema object (must have "type": "object" and "properties").
    model_name : str
        Name for the generated Pydantic model.

    Returns
    -------
    type[BaseModel]
        A Pydantic model class that validates against the schema.

    Raises
    ------
    ValueError
        If the schema is not a valid object schema.
    """
    if schema.get("type") != "object":
        raise ValueError("Schema must be an object type")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Build Pydantic field definitions
    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_schema_type_to_python(field_schema)
        is_required = field_name in required

        if is_required:
            fields[field_name] = (
                field_type,
                Field(..., description=field_schema.get("description", "")),
            )
        else:
            default_value = field_schema.get("default")
            # For optional fields, use the type directly (not a union with None)
            # Pydantic will handle the optional nature via the default
            fields[field_name] = (
                field_type,
                Field(default=default_value, description=field_schema.get("description", "")),
            )

    # Create model with proper typing
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def _json_schema_type_to_python(field_schema: dict[str, Any]) -> type:
    """Convert JSON Schema type to Python type for Pydantic.

    Parameters
    ----------
    field_schema : dict
        JSON Schema field definition.

    Returns
    -------
    type
        Python type annotation.
    """
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    json_type = field_schema.get("type", "string")
    return type_mapping.get(json_type, str)


def validate_against_schema(raw: str, schema: dict[str, Any]) -> ParseResponse:
    """Validate raw LLM output against a JSON Schema.

    Parameters
    ----------
    raw : str
        Raw LLM output (should be JSON).
    schema : dict
        JSON Schema to validate against.

    Returns
    -------
    ParseResponse
        Result with success flag, parsed data, or error message.
    """
    try:
        # Step 1: Parse JSON
        data = json.loads(raw)

        # Step 2: Create dynamic Pydantic model
        model_class = create_model_from_schema(schema)

        # Step 3: Validate
        validated = model_class(**data)

        return ParseResponse(parsed=validated.model_dump(), success=True, error=None)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return ParseResponse(parsed=None, success=False, error=f"Invalid JSON: {e}")

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return ParseResponse(parsed=None, success=False, error=f"Schema validation failed: {e}")

    except ValueError as e:
        logger.warning(f"Schema error: {e}")
        return ParseResponse(parsed=None, success=False, error=f"Invalid schema: {e}")

    except Exception as e:
        logger.error(f"Unexpected error during parsing: {e}")
        return ParseResponse(parsed=None, success=False, error=f"Parsing error: {e}")


# ---------------------------------------------------------------------------
# Corrective re-prompting (L4-04)
# ---------------------------------------------------------------------------


def build_corrective_prompt(original: str, error: str, schema: dict[str, Any]) -> str:
    """Build a corrective prompt to help the LLM fix malformed output.

    Parameters
    ----------
    original : str
        The original malformed output.
    error : str
        The validation error message.
    schema : dict
        The expected JSON Schema.

    Returns
    -------
    str
        A corrective prompt instructing the LLM to fix the output.
    """
    return f"""Your previous output was malformed and could not be parsed.

Original output:
{original}

Validation error:
{error}

Expected schema:
{json.dumps(schema, indent=2)}

Please provide a corrected output that matches the schema exactly.
Return ONLY valid JSON, with no markdown formatting or extra text."""


async def parse_with_retry(
    client: AsyncOpenAI,
    raw: str,
    schema: dict[str, Any],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> ParseResponse:
    """Parse LLM output with corrective re-prompting on failure.

    If the initial parse fails, this function will call the LLM again with a
    corrective prompt that includes the validation error. This repeats up to
    max_retries times.

    Parameters
    ----------
    client : AsyncOpenAI
        OpenAI client for making LLM calls.
    raw : str
        Raw LLM output to parse.
    schema : dict
        JSON Schema to validate against.
    model : str
        OpenAI model to use for corrective prompts.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    ParseResponse
        Final parse result (success or failure after all retries).
    """
    current_output = raw

    for attempt in range(1, max_retries + 1):
        result = validate_against_schema(current_output, schema)

        if result.success:
            logger.info(f"Parse succeeded on attempt {attempt}")
            return result

        if attempt < max_retries:
            # Build corrective prompt
            corrective_prompt = build_corrective_prompt(
                current_output, result.error or "Unknown error", schema
            )

            logger.warning(
                f"Parse failed (attempt {attempt}/{max_retries}), retrying with corrective prompt"
            )
            logger.debug(f"Error: {result.error}")

            # Call LLM with corrective prompt
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": corrective_prompt}],
                    temperature=0.0,  # Use deterministic output for correction
                )

                current_output = response.choices[0].message.content or ""
                logger.debug(f"Corrective response: {current_output[:200]}...")

            except Exception as e:
                logger.error(f"LLM call failed during retry: {e}")
                return ParseResponse(parsed=None, success=False, error=f"Retry failed: {e}")

    # All retries exhausted
    logger.error(f"Parse failed after {max_retries} attempts")
    return result  # Return the last failure
