//! Input validation for tool execution.
//!
//! Validates tool inputs against JSON Schemas before execution, catching invalid
//! inputs at the boundary and providing clear error messages the LLM can use to
//! self-correct.

use super::ToolError;
use serde_json::Value;

/// Validates tool inputs against JSON Schemas.
///
/// Uses the jsonschema crate to enforce that tool inputs match their declared schemas
/// before execution. Validation failures return ToolError::InvalidInput with clear
/// error messages the LLM can use to self-correct.
///
/// # Example
///
/// ```rust,ignore
/// use xola_runtime::tools::{InputValidator, ToolError};
/// use serde_json::json;
///
/// let schema = json!({
///     "type": "object",
///     "properties": {
///         "query": { "type": "string" }
///     },
///     "required": ["query"]
/// });
///
/// let input = json!({ "query": "hello" });
/// assert!(InputValidator::validate(&schema, &input).is_ok());
///
/// let invalid = json!({});
/// assert!(InputValidator::validate(&schema, &invalid).is_err());
/// ```
pub struct InputValidator;

impl InputValidator {
    /// Validates input against a JSON Schema.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::InvalidInput` if:
    /// - The schema itself is invalid
    /// - The input doesn't match the schema (missing fields, wrong types, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "query": { "type": "string" }
    ///     },
    ///     "required": ["query"]
    /// });
    ///
    /// let input = json!({ "query": "hello" });
    /// InputValidator::validate(&schema, &input).unwrap();
    /// ```
    pub fn validate(schema: &Value, input: &Value) -> Result<(), ToolError> {
        // Compile schema
        let json_schema = jsonschema::validator_for(schema)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid schema definition: {}", e)))?;

        // Validate input - collect all validation errors
        let errors: Vec<String> = json_schema
            .iter_errors(input)
            .map(|e| e.to_string())
            .collect();

        if !errors.is_empty() {
            return Err(ToolError::InvalidInput(format!(
                "Input validation failed: {}",
                errors.join(", ")
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_input_passes() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        });

        let input = json!({ "query": "hello" });
        assert!(InputValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_missing_required_field() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        });

        let input = json!({});
        let result = InputValidator::validate(&schema, &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("validation failed"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_wrong_type_rejected() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        });

        let input = json!({ "query": 42 });
        let result = InputValidator::validate(&schema, &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_additional_properties_allowed() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        });

        let input = json!({ "query": "hello", "extra": "field" });
        assert!(InputValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_invalid_schema_rejected() {
        let schema = json!({
            "type": "invalid_type"
        });

        let input = json!({});
        let result = InputValidator::validate(&schema, &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("Invalid schema"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_nested_object_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"]
                }
            },
            "required": ["user"]
        });

        let valid_input = json!({ "user": { "name": "Alice" } });
        assert!(InputValidator::validate(&schema, &valid_input).is_ok());

        let invalid_input = json!({ "user": {} });
        assert!(InputValidator::validate(&schema, &invalid_input).is_err());
    }

    #[test]
    fn test_array_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["items"]
        });

        let valid_input = json!({ "items": ["a", "b", "c"] });
        assert!(InputValidator::validate(&schema, &valid_input).is_ok());

        let invalid_input = json!({ "items": [1, 2, 3] });
        assert!(InputValidator::validate(&schema, &invalid_input).is_err());
    }
}
