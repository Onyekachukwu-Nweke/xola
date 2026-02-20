//! Tool registry for managing available agent capabilities.
//!
//! The registry provides centralized storage and lookup for all tools
//! available to the agent. Tools are registered during startup and then
//! accessed by name during execution.

use super::{Tool, ToolError};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during tool registration.
#[derive(Error, Debug)]
pub enum RegistryError {
    /// Attempted to register a tool with a name that's already taken.
    #[error("Tool '{0}' is already registered")]
    DuplicateName(String),
}

/// Central registry for all tools available to the agent.
///
/// Stores tools in a HashMap keyed by name. Each tool is wrapped in `Arc<dyn Tool>`
/// for thread-safe sharing across async tasks.
///
/// # Thread Safety
///
/// The registry itself does not implement interior mutability. Registration happens
/// during startup, then the registry is wrapped in `Arc<ToolRegistry>` and shared
/// read-only across the runtime.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use xola_runtime::tools::{ToolRegistry, mock::MockTool};
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(MockTool)).unwrap();
///
/// let tool = registry.get("mock_echo").unwrap();
/// assert_eq!(tool.name(), "mock_echo");
///
/// let schemas = registry.list_schemas();
/// assert_eq!(schemas.len(), 1);
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool in the registry.
    ///
    /// # Errors
    ///
    /// Returns `RegistryError::DuplicateName` if a tool with the same name
    /// is already registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use xola_runtime::tools::{ToolRegistry, mock::MockTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Arc::new(MockTool)).unwrap();
    /// ```
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), RegistryError> {
        let name = tool.name().to_string();

        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }

        self.tools.insert(name, tool);
        Ok(())
    }

    /// Looks up a tool by name.
    ///
    /// Returns `None` if no tool with that name is registered.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = registry.get("mock_echo").unwrap();
    /// println!("Found tool: {}", tool.name());
    /// ```
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Returns all registered tool names.
    ///
    /// Useful for debugging and diagnostics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let names = registry.list_names();
    /// println!("Available tools: {:?}", names);
    /// ```
    pub fn list_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Generates tool schemas for the LLM prompt.
    ///
    /// Returns a JSON array where each element has:
    /// - `name`: Tool name
    /// - `description`: Human-readable description
    /// - `parameters`: JSON Schema for input validation
    ///
    /// This format matches the IPC contract for the `/reason` endpoint.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let schemas = registry.list_schemas();
    /// for schema in &schemas {
    ///     println!("{}: {}", schema["name"], schema["description"]);
    /// }
    /// ```
    pub fn list_schemas(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.input_schema()
                })
            })
            .collect()
    }

    /// Validates input against a tool's schema.
    ///
    /// This is a convenience method that looks up the tool, retrieves its schema,
    /// and validates the input. Used by the dispatcher before tool execution.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::NotFound` if the tool doesn't exist.
    /// Returns `ToolError::InvalidInput` if validation fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = json!({ "message": "hello" });
    /// registry.validate_input("mock_echo", &input).unwrap();
    /// ```
    pub fn validate_input(&self, tool_name: &str, input: &Value) -> Result<(), ToolError> {
        let tool = self
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;

        let schema = tool.input_schema();
        crate::tools::validator::InputValidator::validate(&schema, input)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::mock::MockTool;

    #[test]
    fn test_new_registry_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.get("anything").is_none());
        assert!(registry.list_schemas().is_empty());
        assert!(registry.list_names().is_empty());
    }

    #[test]
    fn test_default_is_empty() {
        let registry = ToolRegistry::default();
        assert!(registry.get("anything").is_none());
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::new();
        let tool = Arc::new(MockTool);

        registry.register(tool.clone()).unwrap();

        let retrieved = registry.get("mock_echo").unwrap();
        assert_eq!(retrieved.name(), "mock_echo");
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let mut registry = ToolRegistry::new();
        let tool1 = Arc::new(MockTool);
        let tool2 = Arc::new(MockTool);

        registry.register(tool1).unwrap();
        let err = registry.register(tool2).unwrap_err();

        match err {
            RegistryError::DuplicateName(name) => {
                assert_eq!(name, "mock_echo");
            }
        }
    }

    #[test]
    fn test_list_schemas_format() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 1);

        let schema = &schemas[0];
        assert_eq!(schema["name"], "mock_echo");
        assert_eq!(
            schema["description"],
            "A mock tool that echoes back the input message"
        );
        assert!(schema["parameters"].is_object());
        assert_eq!(schema["parameters"]["type"], "object");
    }

    #[test]
    fn test_list_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let names = registry.list_names();
        assert_eq!(names, vec!["mock_echo"]);
    }

    #[test]
    fn test_get_nonexistent_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let names = registry.list_names();
        assert_eq!(names.len(), 1);

        let schemas = registry.list_schemas();
        assert_eq!(schemas.len(), 1);
    }

    #[test]
    fn test_validate_input_success() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": "test" });
        assert!(registry.validate_input("mock_echo", &input).is_ok());
    }

    #[test]
    fn test_validate_input_missing_field() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({});
        let result = registry.validate_input("mock_echo", &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::InvalidInput(_) => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_validate_input_wrong_type() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool)).unwrap();

        let input = json!({ "message": 123 });
        assert!(registry.validate_input("mock_echo", &input).is_err());
    }

    #[test]
    fn test_validate_input_tool_not_found() {
        let registry = ToolRegistry::new();

        let input = json!({ "message": "test" });
        let result = registry.validate_input("nonexistent", &input);

        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::NotFound(name) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected NotFound error"),
        }
    }
}
