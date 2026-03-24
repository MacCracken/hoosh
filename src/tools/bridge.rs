//! MCP bridge — wraps bote's Dispatcher + szal's built-in tools for direct
//! tool invocation via /v1/tools/list and /v1/tools/call endpoints.

use bote::{Dispatcher, JsonRpcRequest, JsonRpcResponse};

/// MCP tool bridge backed by bote protocol dispatch and szal built-in tools,
/// plus hoosh's own workflow-as-tool.
pub struct McpBridge {
    dispatcher: Dispatcher,
}

impl McpBridge {
    /// Create a new bridge with szal's built-in tools + workflow-as-tool.
    pub fn new() -> Self {
        let dispatcher = szal::mcp::register_tools();
        Self { dispatcher }
    }

    /// List all registered tools as a JSON value.
    pub fn list_tools(&self) -> serde_json::Value {
        let request = JsonRpcRequest::new(1, "tools/list");
        let mut result = match self.dispatcher.dispatch(&request) {
            Some(resp) if resp.error.is_none() => {
                resp.result.unwrap_or(serde_json::json!({"tools": []}))
            }
            Some(resp) => serde_json::json!({
                "error": resp.error.map(|e| e.message).unwrap_or_default()
            }),
            None => serde_json::json!({"tools": []}),
        };

        // Append hoosh-registered tools
        if let Some(tools) = result["tools"].as_array_mut() {
            tools.push(serde_json::json!({
                "name": "szal_workflow_run",
                "description": "Execute a szal workflow (sequential/parallel/DAG steps with retry and rollback)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "flow": {"type": "object", "description": "Flow definition with name, mode, and steps"},
                        "max_concurrency": {"type": "integer", "description": "Max parallel steps (default: 16)", "default": 16}
                    },
                    "required": ["flow"]
                }
            }));
        }

        result
    }

    /// Call a tool by name with the given arguments.
    pub fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        // Handle hoosh-registered tools first
        if name == "szal_workflow_run" {
            return run_workflow(arguments);
        }

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

/// Execute a szal workflow — called directly for the `szal_workflow_run` tool.
fn run_workflow(args: serde_json::Value) -> (serde_json::Value, bool) {
    let flow_json = &args["flow"];
    let flow_def: szal::flow::FlowDef = match serde_json::from_value(flow_json.clone()) {
        Ok(f) => f,
        Err(e) => return (serde_json::json!({"error": format!("invalid flow: {e}")}), true),
    };

    if let Err(e) = flow_def.validate() {
        return (
            serde_json::json!({"error": format!("flow validation failed: {e}")}),
            true,
        );
    }

    let max_concurrency = args["max_concurrency"].as_u64().unwrap_or(16) as usize;
    let config = szal::engine::EngineConfig {
        max_concurrency,
        ..Default::default()
    };

    let handler = szal::engine::handler_fn(|step| async move {
        Ok(serde_json::json!({"step": step.name, "status": "completed"}))
    });

    let engine = szal::engine::Engine::new(config, handler);

    let rt = tokio::runtime::Handle::current();
    match rt.block_on(engine.run(&flow_def)) {
        Ok(result) => (
            serde_json::json!({
                "success": result.success,
                "flow": result.flow_name,
                "completed": result.completed_count(),
                "failed": result.failed_count(),
                "skipped": result.skipped_count(),
                "duration_ms": result.total_duration_ms,
                "rolled_back": result.rolled_back,
            }),
            false,
        ),
        Err(e) => (serde_json::json!({"error": format!("workflow failed: {e}")}), true),
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_workflow_run() {
        let bridge = McpBridge::new();
        let mut flow_def = szal::flow::FlowDef::new("test-flow", szal::flow::FlowMode::Sequential);
        flow_def.steps.push(szal::step::StepDef::new("step-1"));
        flow_def.steps.push(szal::step::StepDef::new("step-2"));
        let flow = serde_json::to_value(&flow_def).unwrap();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_workflow_run", serde_json::json!({"flow": flow}))
        })
        .await
        .unwrap();
        assert!(!is_error, "workflow should succeed: {result}");
        assert_eq!(result["success"], true);
        assert_eq!(result["completed"], 2);
        assert_eq!(result["failed"], 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_workflow_invalid_flow() {
        let bridge = McpBridge::new();
        let (result, _) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_workflow_run", serde_json::json!({"flow": "not an object"}))
        })
        .await
        .unwrap();
        assert!(result["error"].as_str().unwrap().contains("invalid flow"));
    }
}
