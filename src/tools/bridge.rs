//! MCP bridge — wraps bote's Dispatcher + szal's built-in tools for direct
//! tool invocation via /v1/tools/list and /v1/tools/call endpoints.

use bote::{Dispatcher, JsonRpcRequest, JsonRpcResponse};

/// MCP tool bridge backed by bote protocol dispatch and szal built-in tools.
pub struct McpBridge {
    dispatcher: Dispatcher,
}

impl McpBridge {
    /// Create a new bridge with szal's 47 built-in tools registered.
    pub fn new() -> Self {
        let dispatcher = szal::mcp::register_tools();
        Self { dispatcher }
    }

    /// List all registered tools as a JSON value.
    pub fn list_tools(&self) -> serde_json::Value {
        let request = JsonRpcRequest::new(1, "tools/list");
        match self.dispatcher.dispatch(&request) {
            Some(resp) if resp.error.is_none() => {
                resp.result.unwrap_or(serde_json::json!({"tools": []}))
            }
            Some(resp) => serde_json::json!({
                "error": resp.error.map(|e| e.message).unwrap_or_default()
            }),
            None => serde_json::json!({"tools": []}),
        }
    }

    /// Call a tool by name with the given arguments.
    pub fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let request = JsonRpcRequest::new(1, "tools/call")
            .with_params(serde_json::json!({
                "name": name,
                "arguments": arguments,
            }));

        match self.dispatcher.dispatch(&request) {
            Some(resp) => response_to_result(resp),
            None => (
                serde_json::json!({"error": "no response from dispatcher"}),
                true,
            ),
        }
    }

    /// Number of registered tools.
    pub fn tool_count(&self) -> usize {
        // List tools and count entries
        let list = self.list_tools();
        list["tools"].as_array().map(|a| a.len()).unwrap_or(0)
    }
}

impl Default for McpBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a JSON-RPC response to (result_value, is_error).
fn response_to_result(resp: JsonRpcResponse) -> (serde_json::Value, bool) {
    if let Some(err) = resp.error {
        (serde_json::json!({"error": err.message}), true)
    } else {
        (
            resp.result.unwrap_or(serde_json::Value::Null),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_creation() {
        let bridge = McpBridge::new();
        assert!(bridge.tool_count() > 0);
    }

    #[test]
    fn list_tools_returns_array() {
        let bridge = McpBridge::new();
        let list = bridge.list_tools();
        assert!(list["tools"].is_array());
        assert!(!list["tools"].as_array().unwrap().is_empty());
    }

    // szal tools use Handle::current().block_on() internally,
    // so these must run within a multi-thread tokio runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_uuid() {
        let bridge = McpBridge::new();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_uuid", serde_json::json!({}))
        })
        .await
        .unwrap();
        assert!(!is_error, "szal_uuid should succeed: {result}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_timestamp() {
        let bridge = McpBridge::new();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_timestamp", serde_json::json!({}))
        })
        .await
        .unwrap();
        assert!(!is_error, "szal_timestamp should succeed: {result}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_unknown_tool() {
        let bridge = McpBridge::new();
        let (_, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("nonexistent_tool", serde_json::json!({}))
        })
        .await
        .unwrap();
        assert!(is_error);
    }

    #[test]
    fn default_impl() {
        let bridge = McpBridge::default();
        assert!(bridge.tool_count() > 0);
    }
}
