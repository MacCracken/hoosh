//! MCP bridge — wraps bote's Dispatcher + szal's built-in tools for direct
//! tool invocation via /v1/tools/list and /v1/tools/call endpoints.
//!
//! Extended integrations (feature-gated):
//! - `tools-audit`: Tamper-proof tool call logging via libro audit chain
//! - `tools-events`: Tool lifecycle events via majra pub/sub
//! - `tools-discovery`: Cross-node tool announcement and discovery
//! - `tools-sandbox`: Kavach-sandboxed tool execution

use std::sync::Arc;

use bote::{Dispatcher, JsonRpcRequest, JsonRpcResponse};

/// MCP tool bridge backed by bote protocol dispatch and szal built-in tools,
/// plus hoosh's own workflow-as-tool.
pub struct McpBridge {
    dispatcher: Dispatcher,
    /// Tracks active workflow cancellation tokens by execution ID.
    cancellation_tokens:
        std::sync::RwLock<std::collections::HashMap<String, tokio_util::sync::CancellationToken>>,
    /// Execution store for workflow persistence.
    execution_store: Arc<szal::storage::InMemoryExecutionStore>,
    /// Workflow storage for sub-flow composition.
    workflow_storage: Arc<szal::storage::InMemoryStorage>,
    /// Discovery service for multi-node tool sharing.
    #[cfg(feature = "tools-discovery")]
    discovery: Option<bote::discovery::DiscoveryService>,
}

impl McpBridge {
    /// Create a new bridge with szal's built-in tools + workflow-as-tool.
    ///
    /// Wires up audit, events, discovery, and sandbox integrations
    /// based on compile-time feature flags.
    pub fn new() -> Self {
        // Build audit sink (feature: tools-audit)
        #[cfg(feature = "tools-audit")]
        let audit: Option<Arc<dyn bote::AuditSink>> = Some(Arc::new(
            bote::audit::LibroAudit::new()
                .with_source("hoosh")
                .with_agent_id("hoosh-mcp"),
        ));
        #[cfg(not(feature = "tools-audit"))]
        let audit: Option<Arc<dyn bote::AuditSink>> = None;

        // Build event sink (feature: tools-events)
        #[cfg(feature = "tools-events")]
        let events: Option<Arc<dyn bote::EventSink>> =
            Some(Arc::new(bote::events::MajraEvents::new()));
        #[cfg(not(feature = "tools-events"))]
        let events: Option<Arc<dyn bote::EventSink>> = None;

        // Register szal tools with audit + events sinks
        let dispatcher = szal::mcp::register_tools_with(audit, events.clone());

        // Build discovery service (feature: tools-discovery)
        #[cfg(feature = "tools-discovery")]
        let discovery = events.clone().map(|ev| {
            let node_id = format!("hoosh-{}", uuid::Uuid::new_v4());
            tracing::info!(node_id = %node_id, "tool discovery service initialized");
            bote::discovery::DiscoveryService::new(node_id, ev)
        });

        Self {
            dispatcher,
            cancellation_tokens: std::sync::RwLock::new(std::collections::HashMap::new()),
            execution_store: Arc::new(szal::storage::InMemoryExecutionStore::new()),
            workflow_storage: Arc::new(szal::storage::InMemoryStorage::new()),
            #[cfg(feature = "tools-discovery")]
            discovery,
        }
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
                "description": "Execute a szal workflow (sequential/parallel/DAG/hierarchical steps with retry, rollback, conditions, and cancellation)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "flow": {"type": "object", "description": "Flow definition with name, mode (Sequential/Parallel/Dag/Hierarchical), and steps"},
                        "max_concurrency": {"type": "integer", "description": "Max parallel steps (default: 16)", "default": 16}
                    },
                    "required": ["flow"]
                }
            }));
            tools.push(serde_json::json!({
                "name": "szal_workflow_cancel",
                "description": "Cancel a running workflow by execution ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Execution ID to cancel"}
                    },
                    "required": ["execution_id"]
                }
            }));
            tools.push(serde_json::json!({
                "name": "szal_workflow_status",
                "description": "Get workflow execution status and history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string", "description": "Specific execution ID (optional)"},
                        "flow_name": {"type": "string", "description": "Filter by flow name (optional)"}
                    }
                }
            }));
            tools.push(serde_json::json!({
                "name": "szal_workflow_register",
                "description": "Register a reusable workflow template for sub-flow composition",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "flow": {"type": "object", "description": "Flow definition to register as a template"}
                    },
                    "required": ["flow"]
                }
            }));
        }

        result
    }

    /// Call a tool by name with the given arguments.
    pub fn call_tool(&self, name: &str, arguments: serde_json::Value) -> (serde_json::Value, bool) {
        // Handle hoosh-registered tools first
        match name {
            "szal_workflow_run" => return self.run_workflow(arguments),
            "szal_workflow_cancel" => return self.cancel_workflow(arguments),
            "szal_workflow_status" => return self.workflow_status(arguments),
            "szal_workflow_register" => return self.register_workflow(arguments),
            _ => {}
        }

        // Sandbox wrapping for external tools (feature: tools-sandbox)
        #[cfg(feature = "tools-sandbox")]
        if let Some(real_name) = name.strip_prefix("sandbox:") {
            return self.call_sandboxed(real_name, arguments);
        }

        let request = JsonRpcRequest::new(1, "tools/call").with_params(serde_json::json!({
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
        let list = self.list_tools();
        list["tools"].as_array().map(|a| a.len()).unwrap_or(0)
    }

    /// Dynamically register a tool at runtime.
    pub fn register_tool(
        &self,
        name: &str,
        description: &str,
        handler: bote::dispatch::ToolHandler,
    ) -> bool {
        tracing::info!(tool = name, "dynamically registering tool");
        self.dispatcher
            .register_tool(
                bote::ToolDef::new(
                    name,
                    description,
                    bote::ToolSchema::new("object", std::collections::HashMap::new(), vec![]),
                ),
                handler,
            )
            .is_ok()
    }

    /// Dynamically deregister a tool at runtime.
    pub fn deregister_tool(&self, name: &str) -> bool {
        tracing::info!(tool = name, "deregistering tool");
        self.dispatcher.deregister_tool(name).is_ok()
    }

    /// Announce this node's tools to the discovery mesh.
    #[cfg(feature = "tools-discovery")]
    pub fn announce_tools(&self) {
        if let Some(discovery) = &self.discovery {
            let list = self.list_tools();
            let tools: Vec<bote::ToolDef> = list["tools"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| serde_json::from_value(t.clone()).ok())
                        .collect()
                })
                .unwrap_or_default();
            discovery.announce(&tools);
        }
    }

    // ─── Workflow execution ─────────────────────────────────────────────

    /// Execute a szal workflow with full feature integration:
    /// - DAG/Parallel/Sequential/Hierarchical modes
    /// - Condition DSL on steps
    /// - Cancellation token tracking
    /// - Execution persistence
    /// - Step-type metrics to Prometheus
    /// - Sub-flow composition via workflow storage
    fn run_workflow(&self, args: serde_json::Value) -> (serde_json::Value, bool) {
        let flow_json = &args["flow"];
        let flow_def: szal::flow::FlowDef = match serde_json::from_value(flow_json.clone()) {
            Ok(f) => f,
            Err(e) => {
                return (
                    serde_json::json!({"error": format!("invalid flow: {e}")}),
                    true,
                );
            }
        };

        if let Err(e) = flow_def.validate() {
            return (
                serde_json::json!({"error": format!("flow validation failed: {e}")}),
                true,
            );
        }

        let max_concurrency = args["max_concurrency"].as_u64().unwrap_or(16) as usize;

        // Build engine config with execution store and step-type metrics
        let config = szal::engine::EngineConfig {
            max_concurrency,
            execution_store: Some(self.execution_store.clone()),
            step_type_metrics: Some(Arc::new(|step_type, status, duration_ms| {
                crate::metrics::record_workflow_step(step_type, status, duration_ms);
            })),
            ..Default::default()
        };

        // Sub-flow handler wrapping the base handler
        let base_handler = szal::engine::handler_fn(|step| async move {
            Ok(serde_json::json!({"step": step.name, "status": "completed"}))
        });
        let handler = szal::engine::sub_flow_handler(self.workflow_storage.clone(), base_handler);

        let engine = szal::engine::Engine::new(config, handler);

        // Create cancellation token and track it
        let token = szal::engine::CancellationToken::new();
        let execution_id = flow_def.id.to_string();
        {
            let mut tokens = self
                .cancellation_tokens
                .write()
                .unwrap_or_else(|e| e.into_inner());
            tokens.insert(execution_id.clone(), token.clone());
        }

        let rt = tokio::runtime::Handle::current();
        let result = rt.block_on(engine.run_with_cancellation(&flow_def, token));

        // Clean up cancellation token
        {
            let mut tokens = self
                .cancellation_tokens
                .write()
                .unwrap_or_else(|e| e.into_inner());
            tokens.remove(&execution_id);
        }

        match result {
            Ok(result) => (
                serde_json::json!({
                    "execution_id": execution_id,
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
            Err(e) => (
                serde_json::json!({
                    "execution_id": execution_id,
                    "error": format!("workflow failed: {e}"),
                }),
                true,
            ),
        }
    }

    /// Cancel a running workflow by execution ID.
    fn cancel_workflow(&self, args: serde_json::Value) -> (serde_json::Value, bool) {
        let execution_id = match args["execution_id"].as_str() {
            Some(id) => id,
            None => return (serde_json::json!({"error": "missing execution_id"}), true),
        };

        let tokens = self
            .cancellation_tokens
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(token) = tokens.get(execution_id) {
            token.cancel();
            tracing::info!(execution_id, "workflow cancellation requested");
            (
                serde_json::json!({"cancelled": true, "execution_id": execution_id}),
                false,
            )
        } else {
            (
                serde_json::json!({"error": "execution not found or already completed"}),
                true,
            )
        }
    }

    /// Get workflow execution status from the execution store.
    fn workflow_status(&self, args: serde_json::Value) -> (serde_json::Value, bool) {
        use szal::storage::ExecutionStore;

        if let Some(execution_id) = args["execution_id"].as_str() {
            match self.execution_store.get(execution_id) {
                Some(record) => (serde_json::to_value(&record).unwrap_or_default(), false),
                None => (serde_json::json!({"error": "execution not found"}), true),
            }
        } else {
            let flow_name = args["flow_name"].as_str();
            let ids = self.execution_store.list(flow_name);
            let records: Vec<serde_json::Value> = ids
                .iter()
                .filter_map(|id| self.execution_store.get(id))
                .map(|r| serde_json::to_value(&r).unwrap_or_default())
                .collect();
            (serde_json::json!({"executions": records}), false)
        }
    }

    /// Register a reusable workflow template.
    fn register_workflow(&self, args: serde_json::Value) -> (serde_json::Value, bool) {
        let flow_json = &args["flow"];
        let flow_def: szal::flow::FlowDef = match serde_json::from_value(flow_json.clone()) {
            Ok(f) => f,
            Err(e) => {
                return (
                    serde_json::json!({"error": format!("invalid flow: {e}")}),
                    true,
                );
            }
        };

        if let Err(e) = flow_def.validate() {
            return (
                serde_json::json!({"error": format!("flow validation failed: {e}")}),
                true,
            );
        }

        let name = flow_def.name.clone();
        self.workflow_storage.insert(flow_def);
        tracing::info!(flow = %name, "workflow template registered");
        (
            serde_json::json!({"registered": true, "flow_name": name}),
            false,
        )
    }

    /// Execute a tool in a sandbox (feature: tools-sandbox).
    #[cfg(feature = "tools-sandbox")]
    fn call_sandboxed(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> (serde_json::Value, bool) {
        let config = bote::sandbox::ToolSandboxConfig::basic();
        let handler = bote::sandbox::wrap_command(name, config);
        let result = handler(arguments);
        let is_error = result.get("error").is_some();
        (result, is_error)
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
        (resp.result.unwrap_or(serde_json::Value::Null), false)
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

    #[test]
    fn list_tools_includes_workflow_tools() {
        let bridge = McpBridge::new();
        let list = bridge.list_tools();
        let tools = list["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"szal_workflow_run"));
        assert!(names.contains(&"szal_workflow_cancel"));
        assert!(names.contains(&"szal_workflow_status"));
        assert!(names.contains(&"szal_workflow_register"));
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
        assert!(result["execution_id"].as_str().is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_workflow_dag_mode() {
        let bridge = McpBridge::new();
        let mut flow = szal::flow::FlowDef::new("dag-flow", szal::flow::FlowMode::Dag);
        flow.steps.push(szal::step::StepDef::new("a"));
        flow.steps.push(szal::step::StepDef::new("b"));
        let flow_json = serde_json::to_value(&flow).unwrap();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_workflow_run", serde_json::json!({"flow": flow_json}))
        })
        .await
        .unwrap();
        assert!(!is_error, "DAG workflow should succeed: {result}");
        assert_eq!(result["success"], true);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn call_workflow_invalid_flow() {
        let bridge = McpBridge::new();
        let (result, _) = tokio::task::spawn_blocking(move || {
            bridge.call_tool(
                "szal_workflow_run",
                serde_json::json!({"flow": "not an object"}),
            )
        })
        .await
        .unwrap();
        assert!(result["error"].as_str().unwrap().contains("invalid flow"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_cancel_nonexistent() {
        let bridge = McpBridge::new();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool(
                "szal_workflow_cancel",
                serde_json::json!({"execution_id": "nonexistent"}),
            )
        })
        .await
        .unwrap();
        assert!(is_error);
        assert!(result["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_status_empty() {
        let bridge = McpBridge::new();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool("szal_workflow_status", serde_json::json!({}))
        })
        .await
        .unwrap();
        assert!(!is_error);
        assert!(result["executions"].is_array());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workflow_register_and_status() {
        let bridge = McpBridge::new();
        let mut flow = szal::flow::FlowDef::new("template-1", szal::flow::FlowMode::Sequential);
        flow.steps.push(szal::step::StepDef::new("s1"));
        let flow_json = serde_json::to_value(&flow).unwrap();
        let (result, is_error) = tokio::task::spawn_blocking(move || {
            bridge.call_tool(
                "szal_workflow_register",
                serde_json::json!({"flow": flow_json}),
            )
        })
        .await
        .unwrap();
        assert!(!is_error);
        assert_eq!(result["registered"], true);
        assert_eq!(result["flow_name"], "template-1");
    }

    #[test]
    fn dynamic_tool_registration() {
        let bridge = McpBridge::new();
        let initial = bridge.tool_count();
        bridge.register_tool(
            "hoosh_test_tool",
            "test tool",
            Arc::new(|_| serde_json::json!({"ok": true})),
        );
        assert_eq!(bridge.tool_count(), initial + 1);

        assert!(bridge.deregister_tool("hoosh_test_tool"));
        assert_eq!(bridge.tool_count(), initial);
    }
}
