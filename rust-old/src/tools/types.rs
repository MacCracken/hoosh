//! Unified tool-use types — provider-neutral representations for tool definitions,
//! tool calls, and tool results across OpenAI, Anthropic, Gemini, and Ollama.

use serde::{Deserialize, Serialize};

/// A tool definition passed in an inference request (what tools the model may call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match `^[a-zA-Z0-9_-]+$`).
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool's input parameters.
    pub parameters: serde_json::Value,
}

/// How the model should choose tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Model decides whether to call tools.
    #[default]
    Auto,
    /// Model must not call any tools.
    None,
    /// Model must call at least one tool.
    Required,
    /// Force a specific tool by name.
    #[serde(untagged)]
    Tool(String),
}

/// A tool call emitted by the model in its response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call (used to match results in multi-turn).
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments to pass to the tool (parsed JSON).
    pub arguments: serde_json::Value,
}

/// A tool result for multi-turn conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the tool call this result responds to.
    pub tool_call_id: String,
    /// Result content (text).
    pub content: String,
    /// Whether this result represents an error.
    #[serde(default)]
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_serde() {
        let def = ToolDefinition {
            name: "get_weather".into(),
            description: "Get current weather for a location".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        };
        let json = serde_json::to_string(&def).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "get_weather");
        assert!(back.parameters["properties"]["location"].is_object());
    }

    #[test]
    fn tool_call_serde() {
        let call = ToolCall {
            id: "call_123".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"location": "London"}),
        };
        let json = serde_json::to_string(&call).unwrap();
        let back: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "call_123");
        assert_eq!(back.arguments["location"], "London");
    }

    #[test]
    fn tool_result_serde() {
        let result = ToolResult {
            tool_call_id: "call_123".into(),
            content: "22°C, partly cloudy".into(),
            is_error: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool_call_id, "call_123");
        assert!(!back.is_error);
    }

    #[test]
    fn tool_result_error() {
        let result = ToolResult {
            tool_call_id: "call_456".into(),
            content: "Location not found".into(),
            is_error: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ToolResult = serde_json::from_str(&json).unwrap();
        assert!(back.is_error);
    }

    #[test]
    fn tool_choice_variants() {
        let auto: ToolChoice = serde_json::from_str("\"auto\"").unwrap();
        assert!(matches!(auto, ToolChoice::Auto));

        let none: ToolChoice = serde_json::from_str("\"none\"").unwrap();
        assert!(matches!(none, ToolChoice::None));

        let required: ToolChoice = serde_json::from_str("\"required\"").unwrap();
        assert!(matches!(required, ToolChoice::Required));
    }

    #[test]
    fn tool_choice_default() {
        let choice = ToolChoice::default();
        assert!(matches!(choice, ToolChoice::Auto));
    }

    #[test]
    fn tool_definition_empty_params() {
        let def = ToolDefinition {
            name: "ping".into(),
            description: "No-op ping".into(),
            parameters: serde_json::json!({}),
        };
        let json = serde_json::to_string(&def).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "ping");
    }
}
