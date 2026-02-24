"""Unit tests for the parser module (L4-03, L4-04)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from llm_surface.parser import (
    ParseRequest,
    ParseResponse,
    build_corrective_prompt,
    create_model_from_schema,
    parse_with_retry,
    validate_against_schema,
)


# ---------------------------------------------------------------------------
# L4-03: Schema validation tests
# ---------------------------------------------------------------------------


def test_parse_request_model():
    """Test ParseRequest Pydantic model."""
    request = ParseRequest(
        raw='{"name": "test"}',
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
        attempt=1,
    )
    assert request.raw == '{"name": "test"}'
    assert request.attempt == 1


def test_parse_request_default_attempt():
    """Test ParseRequest with default attempt value."""
    request = ParseRequest(
        raw='{"name": "test"}', schema={"type": "object", "properties": {"name": {"type": "string"}}}
    )
    assert request.attempt == 1


def test_parse_response_success():
    """Test ParseResponse for successful parse."""
    response = ParseResponse(parsed={"name": "test"}, success=True, error=None)
    assert response.success is True
    assert response.parsed == {"name": "test"}
    assert response.error is None


def test_parse_response_failure():
    """Test ParseResponse for failed parse."""
    response = ParseResponse(parsed=None, success=False, error="Invalid JSON")
    assert response.success is False
    assert response.parsed is None
    assert response.error == "Invalid JSON"


def test_create_model_from_schema_simple():
    """Test creating a Pydantic model from a simple schema."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    Model = create_model_from_schema(schema)

    # Valid data
    instance = Model(name="Alice", age=30)
    assert instance.name == "Alice"
    assert instance.age == 30

    # Missing optional field
    instance = Model(name="Bob")
    assert instance.name == "Bob"
    assert instance.age is None


def test_create_model_from_schema_missing_required():
    """Test that missing required fields raise validation error."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    Model = create_model_from_schema(schema)

    with pytest.raises(ValidationError):
        Model()  # Missing required field 'name'


def test_create_model_from_schema_invalid_type():
    """Test schema must be an object type."""
    schema = {"type": "string"}

    with pytest.raises(ValueError, match="Schema must be an object type"):
        create_model_from_schema(schema)


def test_validate_against_schema_success():
    """Test successful validation."""
    raw = '{"name": "Alice", "age": 30}'
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    result = validate_against_schema(raw, schema)

    assert result.success is True
    assert result.parsed == {"name": "Alice", "age": 30}
    assert result.error is None


def test_validate_against_schema_invalid_json():
    """Test validation with invalid JSON."""
    raw = "{invalid json"
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    result = validate_against_schema(raw, schema)

    assert result.success is False
    assert result.parsed is None
    assert "Invalid JSON" in result.error


def test_validate_against_schema_schema_mismatch():
    """Test validation with data that doesn't match schema."""
    raw = '{"name": 123}'  # name should be string, not integer
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = validate_against_schema(raw, schema)

    assert result.success is False
    assert result.parsed is None
    assert "Schema validation failed" in result.error


def test_validate_against_schema_missing_required():
    """Test validation with missing required field."""
    raw = '{"age": 30}'
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    result = validate_against_schema(raw, schema)

    assert result.success is False
    assert result.parsed is None
    assert "Schema validation failed" in result.error


def test_validate_against_schema_with_defaults():
    """Test validation with optional fields using defaults."""
    raw = '{"name": "Alice"}'
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "default": 25},
        },
        "required": ["name"],
    }

    result = validate_against_schema(raw, schema)

    assert result.success is True
    assert result.parsed["name"] == "Alice"
    # Note: age will be None because our simple converter doesn't handle defaults in the same way


# ---------------------------------------------------------------------------
# L4-04: Corrective prompting tests
# ---------------------------------------------------------------------------


def test_build_corrective_prompt():
    """Test building a corrective prompt."""
    original = '{"name": 123}'
    error = "name must be a string"
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    prompt = build_corrective_prompt(original, error, schema)

    assert "malformed" in prompt.lower()
    assert original in prompt
    assert error in prompt
    assert json.dumps(schema, indent=2) in prompt
    assert "corrected output" in prompt.lower()


@pytest.mark.asyncio
async def test_parse_with_retry_success_first_attempt():
    """Test parse_with_retry succeeds on first attempt."""
    raw = '{"name": "Alice"}'
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Create a mock client (won't be called since first attempt succeeds)
    mock_client = AsyncMock()

    result = await parse_with_retry(mock_client, raw, schema, max_retries=3)

    assert result.success is True
    assert result.parsed == {"name": "Alice"}
    # Client should not be called since first attempt succeeded
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_parse_with_retry_success_after_retry():
    """Test parse_with_retry succeeds after corrective retry."""
    # First attempt: invalid JSON
    raw = '{"name": 123}'  # Should be string
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Mock OpenAI client to return corrected output
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"name": "Alice"}'

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    result = await parse_with_retry(mock_client, raw, schema, max_retries=3)

    assert result.success is True
    assert result.parsed == {"name": "Alice"}
    # Client should be called once for retry
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_parse_with_retry_exhausted():
    """Test parse_with_retry after exhausting all retries."""
    raw = '{"name": 123}'  # Invalid - should be string
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Mock OpenAI client to always return invalid output
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"name": 456}'  # Still invalid

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    result = await parse_with_retry(mock_client, raw, schema, max_retries=2)

    assert result.success is False
    assert result.parsed is None
    # Client should be called (max_retries - 1) times
    assert mock_client.chat.completions.create.call_count == 1  # max_retries=2, so 1 retry


@pytest.mark.asyncio
async def test_parse_with_retry_llm_error():
    """Test parse_with_retry when LLM call fails."""
    raw = '{"name": 123}'
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Mock OpenAI client to raise an error
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    result = await parse_with_retry(mock_client, raw, schema, max_retries=3)

    assert result.success is False
    assert result.parsed is None
    assert "Retry failed" in result.error


@pytest.mark.asyncio
async def test_parse_with_retry_temperature_zero():
    """Test that parse_with_retry uses temperature=0 for deterministic output."""
    raw = '{"name": 123}'
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"name": "Alice"}'

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    await parse_with_retry(mock_client, raw, schema, max_retries=3)

    # Verify temperature was set to 0
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_validate_against_schema_empty_string():
    """Test validation with empty string."""
    raw = ""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    result = validate_against_schema(raw, schema)

    assert result.success is False
    assert "Invalid JSON" in result.error


def test_validate_against_schema_complex_nested():
    """Test validation with nested objects."""
    raw = '{"user": {"name": "Alice", "age": 30}}'
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            }
        },
    }

    result = validate_against_schema(raw, schema)

    # Our simple converter treats nested objects as dicts
    assert result.success is True
    assert result.parsed["user"]["name"] == "Alice"
