//! Provider format conversion — translate hoosh's unified tool types to/from
//! OpenAI, Anthropic, and Ollama native JSON formats.

use super::types::{ToolCall, ToolDefinition};

// ---------------------------------------------------------------------------
// Tool definitions → provider-native format
// ---------------------------------------------------------------------------

/// Convert tool definitions to OpenAI/Ollama format.
/// OpenAI wraps each tool in `{"type": "function", "function": {...}}`.
pub fn to_openai_tools(defs: &[ToolDefinition]) -> Vec<serde_json::Value> {
    defs.iter()
        .map(|d| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters,
                }
            })
        })
        .collect()
}

/// Convert tool definitions to Anthropic format.
/// Anthropic uses `{"name": ..., "description": ..., "input_schema": {...}}`.
pub fn to_anthropic_tools(defs: &[ToolDefinition]) -> Vec<serde_json::Value> {
    defs.iter()
        .map(|d| {
            serde_json::json!({
                "name": d.name,
                "description": d.description,
                "input_schema": d.parameters,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Provider response → unified ToolCall
// ---------------------------------------------------------------------------

/// Parse tool calls from an OpenAI/Ollama chat completion response.
/// Expects `choices[0].message.tool_calls` array.
pub fn parse_openai_tool_calls(response: &serde_json::Value) -> Vec<ToolCall> {
    let tool_calls = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array());

    let Some(calls) = tool_calls else {
        return Vec::new();
    };

    calls
        .iter()
        .filter_map(|tc| {
            let id = tc["id"].as_str()?;
            let name = tc["function"]["name"].as_str()?;
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
            let arguments = serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
            Some(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })
        })
        .collect()
}

/// Parse tool calls from Anthropic's response content blocks.
/// Anthropic returns `content: [{type: "tool_use", id, name, input}, ...]`.
pub fn parse_anthropic_tool_calls(content: &[serde_json::Value]) -> Vec<ToolCall> {
    content
        .iter()
        .filter(|block| block["type"].as_str() == Some("tool_use"))
        .filter_map(|block| {
            let id = block["id"].as_str()?;
            let name = block["name"].as_str()?;
            let arguments = block["input"].clone();
            Some(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })
        })
        .collect()
}

/// Extract text content from Anthropic response content blocks.
/// Filters to `type: "text"` blocks and joins them.
pub fn extract_anthropic_text(content: &[serde_json::Value]) -> String {
    content
        .iter()
        .filter(|block| block["type"].as_str() == Some("text"))
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_tools_format() {
        let defs = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"loc": {"type": "string"}}}),
        }];
        let result = to_openai_tools(&defs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "function");
        assert_eq!(result[0]["function"]["name"], "get_weather");
        assert_eq!(result[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn anthropic_tools_format() {
        let defs = vec![ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            parameters: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }];
        let result = to_anthropic_tools(&defs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["name"], "search");
        assert_eq!(result[0]["input_schema"]["type"], "object");
        // No "function" wrapper — Anthropic uses flat structure
        assert!(result[0].get("function").is_none());
    }

    #[test]
    fn parse_openai_tool_calls_basic() {
        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"London\"}"
                        }
                    }]
                }
            }]
        });
        let calls = parse_openai_tool_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "London");
    }

    #[test]
    fn parse_openai_no_tool_calls() {
        let response = serde_json::json!({
            "choices": [{"message": {"content": "Hello!"}}]
        });
        let calls = parse_openai_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_openai_multiple_tool_calls() {
        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "c2", "function": {"name": "b", "arguments": "{\"x\":1}"}},
                    ]
                }
            }]
        });
        let calls = parse_openai_tool_calls(&response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments["x"], 1);
    }

    #[test]
    fn parse_anthropic_tool_calls_basic() {
        let content = vec![
            serde_json::json!({"type": "text", "text": "Let me check that."}),
            serde_json::json!({
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "London"}
            }),
        ];
        let calls = parse_anthropic_tool_calls(&content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_123");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "London");
    }

    #[test]
    fn parse_anthropic_no_tool_calls() {
        let content = vec![serde_json::json!({"type": "text", "text": "Hello!"})];
        let calls = parse_anthropic_tool_calls(&content);
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_anthropic_text_mixed() {
        let content = vec![
            serde_json::json!({"type": "text", "text": "I'll check "}),
            serde_json::json!({"type": "tool_use", "id": "t1", "name": "x", "input": {}}),
            serde_json::json!({"type": "text", "text": "the weather."}),
        ];
        let text = extract_anthropic_text(&content);
        assert_eq!(text, "I'll check the weather.");
    }

    #[test]
    fn extract_anthropic_text_only() {
        let content = vec![serde_json::json!({"type": "text", "text": "Hello!"})];
        assert_eq!(extract_anthropic_text(&content), "Hello!");
    }

    #[test]
    fn empty_defs_produce_empty_arrays() {
        assert!(to_openai_tools(&[]).is_empty());
        assert!(to_anthropic_tools(&[]).is_empty());
    }
}
