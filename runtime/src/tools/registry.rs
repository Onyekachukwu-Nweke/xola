//! Tool registry for managing available agent capabilities.
//!
//! The registry provides centralized storage and lookup for all tools
//! available to the agent. Tools are registered during startup and then
//! accessed by name during execution.

use super::Tool;
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
}
