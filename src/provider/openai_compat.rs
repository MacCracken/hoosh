//! OpenAI-compatible base provider for llama.cpp, LM Studio, LocalAI, Synapse.

use std::time::Instant;

use serde::Deserialize;
use tokio::sync::mpsc;

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo, Role, TokenUsage};
use crate::provider::{LlmProvider, ProviderType, TlsConfig, build_provider_client};

/// Generic provider that speaks the OpenAI-compatible `/v1/` API.
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    provider_type: ProviderType,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        provider_type: ProviderType,
        tls_config: Option<&TlsConfig>,
    ) -> Self {
        Self {
            client: build_provider_client(tls_config),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            provider_type,
        }
    }

    /// The base URL this provider points at.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Build the JSON body for a chat completions request.
fn build_chat_body(req: &InferenceRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = if req.messages.is_empty() {
        // Synthesize from prompt + optional system
        let mut msgs = Vec::new();
        if let Some(sys) = &req.system {
            msgs.push(serde_json::json!({"role": "system", "content": sys}));
        }
        msgs.push(serde_json::json!({"role": "user", "content": req.prompt}));
        msgs
    } else {
        req.messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                serde_json::json!({"role": role, "content": m.content})
            })
            .collect()
    };

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
    });

    if let Some(max) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(tp) = req.top_p {
        body["top_p"] = serde_json::json!(tp);
    }
    if !req.tools.is_empty() {
        body["tools"] = serde_json::json!(crate::tools::to_openai_tools(&req.tools));
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = serde_json::json!(choice);
    }

    body
}

// ---- Response deserialization types ----

#[derive(Deserialize)]
struct OaiChatResponse {
    model: Option<String>,
    choices: Vec<OaiChoice>,
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiMessage,
}

#[derive(Deserialize)]
struct OaiMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaiToolCall>,
}

#[derive(Deserialize)]
struct OaiToolCall {
    id: String,
    function: OaiToolCallFunction,
}

#[derive(Deserialize)]
struct OaiToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OaiUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct OaiStreamChunk {
    choices: Vec<OaiStreamChoice>,
}

#[derive(Deserialize)]
struct OaiStreamChoice {
    delta: OaiDelta,
    #[serde(default)]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaiDelta {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiStreamToolCall>>,
}

#[derive(Deserialize)]
struct OaiStreamToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaiStreamFunction>,
}

#[derive(Deserialize)]
struct OaiStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulates incremental tool call deltas into complete ToolCalls.
struct ToolCallAccumulator {
    calls: Vec<(String, String, String)>, // (id, name, arguments_json)
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self { calls: Vec::new() }
    }

    fn process_delta(&mut self, tool_calls: &[OaiStreamToolCall]) {
        for tc in tool_calls {
            // Grow the calls vec as needed
            while self.calls.len() <= tc.index {
                self.calls
                    .push((String::new(), String::new(), String::new()));
            }
            let entry = &mut self.calls[tc.index];
            if let Some(id) = &tc.id {
                entry.0 = id.clone();
            }
            if let Some(f) = &tc.function {
                if let Some(name) = &f.name {
                    entry.1 = name.clone();
                }
                if let Some(args) = &f.arguments {
                    entry.2.push_str(args);
                }
            }
        }
    }

    fn finish(self) -> Vec<crate::tools::ToolCall> {
        self.calls
            .into_iter()
            .filter(|(id, name, _)| !id.is_empty() && !name.is_empty())
            .map(|(id, name, args)| crate::tools::ToolCall {
                id,
                name,
                arguments: serde_json::from_str(&args).unwrap_or(serde_json::json!({})),
            })
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

#[derive(Deserialize)]
struct OaiModelsResponse {
    data: Vec<OaiModelObject>,
}

#[derive(Deserialize)]
struct OaiModelObject {
    id: String,
    owned_by: Option<String>,
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = build_chat_body(&InferenceRequest {
            stream: false,
            ..request.clone()
        });

        let start = Instant::now();
        let mut rb = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await?.error_for_status()?;
        let oai: OaiChatResponse = resp.json().await?;
        let latency = start.elapsed().as_millis() as u64;

        let first_choice = oai.choices.first();
        let text = first_choice
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let tool_calls = first_choice
            .map(|c| {
                c.message
                    .tool_calls
                    .iter()
                    .map(|tc| crate::tools::ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({})),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let usage = oai.usage.as_ref();
        Ok(InferenceResponse {
            text,
            model: oai.model.unwrap_or_else(|| request.model.clone()),
            usage: TokenUsage {
                prompt_tokens: usage.and_then(|u| u.prompt_tokens).unwrap_or(0),
                completion_tokens: usage.and_then(|u| u.completion_tokens).unwrap_or(0),
                total_tokens: usage.and_then(|u| u.total_tokens).unwrap_or(0),
            },
            provider: self.provider_type.to_string(),
            latency_ms: latency,
            tool_calls,
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<mpsc::Receiver<anyhow::Result<String>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = build_chat_body(&InferenceRequest {
            stream: true,
            ..request.clone()
        });

        let mut rb = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await?.error_for_status()?;

        // Validate response is SSE, not an HTML error page
        if let Some(ct) = resp.headers().get("content-type") {
            let ct_str = ct.to_str().unwrap_or("");
            if !ct_str.contains("text/event-stream") && !ct_str.contains("application/json") {
                return Err(anyhow::anyhow!(
                    "expected SSE stream, got Content-Type: {ct_str}"
                ));
            }
        }

        let (tx, rx) = mpsc::channel(256);

        tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut tool_acc = ToolCallAccumulator::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(e.into())).await;
                        return;
                    }
                };
                // Guard against unbounded buffer growth
                if buf.len() + chunk.len() > 1024 * 1024 {
                    let _ = tx
                        .send(Err(anyhow::anyhow!("SSE line exceeded 1MB limit")))
                        .await;
                    return;
                }
                buf.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE lines
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf = buf[pos + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }
                    let data = if let Some(d) = line.strip_prefix("data: ") {
                        d.trim()
                    } else {
                        continue;
                    };
                    if data == "[DONE]" {
                        // Emit accumulated tool calls as a special marker
                        if !tool_acc.is_empty() {
                            let calls = tool_acc.finish();
                            if let Ok(json) = serde_json::to_string(&calls) {
                                let _ = tx.send(Ok(format!("\x00TOOL_CALLS:{json}"))).await;
                            }
                        }
                        return;
                    }
                    match serde_json::from_str::<OaiStreamChunk>(data) {
                        Err(e) => {
                            tracing::warn!("malformed SSE chunk from provider: {e}");
                        }
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                // Accumulate tool call deltas
                                if let Some(tool_calls) = &choice.delta.tool_calls {
                                    tool_acc.process_delta(tool_calls);
                                }
                                // Stream text content
                                if let Some(content) = &choice.delta.content
                                    && !content.is_empty()
                                    && tx.send(Ok(content.clone())).await.is_err()
                                {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let mut rb = self.client.get(&url);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await?.error_for_status()?;
        let models: OaiModelsResponse = resp.json().await?;
        let provider = self.provider_type.to_string();

        Ok(models
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
                provider: m.owned_by.unwrap_or_else(|| provider.clone()),
                parameters: None,
                context_length: None,
                available: true,
            })
            .collect())
    }

    async fn health_check(&self) -> anyhow::Result<bool> {
        let url = format!("{}/v1/models", self.base_url);
        let mut rb = self.client.get(&url);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await?;
        Ok(resp.status().is_success())
    }

    async fn embeddings(
        &self,
        request: &crate::inference::EmbeddingsRequest,
    ) -> anyhow::Result<crate::inference::EmbeddingsResponse> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let mut rb = self.client.post(&url).json(request);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let resp = rb.send().await?.error_for_status()?;
        let result: crate::inference::EmbeddingsResponse = resp.json().await?;
        Ok(result)
    }

    fn provider_type(&self) -> ProviderType {
        self.provider_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Message;

    #[test]
    fn build_body_from_prompt() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hello".into(),
            ..Default::default()
        };
        let body = build_chat_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
        assert_eq!(body["model"], "llama3");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn build_body_from_prompt_with_system() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hello".into(),
            system: Some("You are helpful.".into()),
            ..Default::default()
        };
        let body = build_chat_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hello");
    }

    #[test]
    fn build_body_from_messages() {
        let req = InferenceRequest {
            model: "gpt-4".into(),
            messages: vec![
                Message::new(Role::System, "Be concise."),
                Message::new(Role::User, "Hi"),
                Message::new(Role::Assistant, "Hello!"),
            ],
            ..Default::default()
        };
        let body = build_chat_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "Hello!");
    }

    #[test]
    fn build_body_optional_params() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hi".into(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: true,
            ..Default::default()
        };
        let body = build_chat_body(&req);
        assert_eq!(body["max_tokens"], 100);
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < f64::EPSILON);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < f64::EPSILON);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn build_body_no_optional_params() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let body = build_chat_body(&req);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn provider_creation() {
        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080",
            None,
            ProviderType::LlamaCpp,
            None,
        );
        assert_eq!(p.base_url(), "http://localhost:8080");
        assert_eq!(p.provider_type(), ProviderType::LlamaCpp);
    }

    #[test]
    fn provider_strips_trailing_slash() {
        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080/",
            Some("sk-test".into()),
            ProviderType::OpenAi,
            None,
        );
        assert_eq!(p.base_url(), "http://localhost:8080");
    }

    #[test]
    fn provider_preserves_api_key() {
        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080",
            Some("sk-secret".into()),
            ProviderType::OpenAi,
            None,
        );
        assert!(p.api_key.is_some());
        assert_eq!(p.api_key.as_deref(), Some("sk-secret"));
    }

    #[test]
    fn provider_no_api_key() {
        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080",
            None,
            ProviderType::LlamaCpp,
            None,
        );
        assert!(p.api_key.is_none());
    }

    #[test]
    fn provider_with_tls_config() {
        let tls = TlsConfig {
            pinned_certs: vec!["/nonexistent/cert.pem".into()],
            ..Default::default()
        };
        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080",
            None,
            ProviderType::LlamaCpp,
            Some(&tls),
        );
        assert_eq!(p.base_url(), "http://localhost:8080");
    }

    #[test]
    fn build_body_with_tools() {
        use crate::tools::{ToolChoice, ToolDefinition};
        let req = InferenceRequest {
            model: "gpt-4o".into(),
            prompt: "What's the weather?".into(),
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get weather".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"loc": {"type": "string"}}}),
            }],
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };
        let body = build_chat_body(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn build_body_no_tools() {
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let body = build_chat_body(&req);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn tool_call_accumulator_single_call() {
        let mut acc = ToolCallAccumulator::new();
        // First chunk: id + name + partial args
        acc.process_delta(&[OaiStreamToolCall {
            index: 0,
            id: Some("call_abc".into()),
            function: Some(OaiStreamFunction {
                name: Some("get_weather".into()),
                arguments: Some("{\"lo".into()),
            }),
        }]);
        // Second chunk: more args
        acc.process_delta(&[OaiStreamToolCall {
            index: 0,
            id: None,
            function: Some(OaiStreamFunction {
                name: None,
                arguments: Some("cation\":\"London\"}".into()),
            }),
        }]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "London");
    }

    #[test]
    fn tool_call_accumulator_multiple_calls() {
        let mut acc = ToolCallAccumulator::new();
        acc.process_delta(&[
            OaiStreamToolCall {
                index: 0,
                id: Some("c1".into()),
                function: Some(OaiStreamFunction {
                    name: Some("tool_a".into()),
                    arguments: Some("{}".into()),
                }),
            },
            OaiStreamToolCall {
                index: 1,
                id: Some("c2".into()),
                function: Some(OaiStreamFunction {
                    name: Some("tool_b".into()),
                    arguments: Some("{\"x\":1}".into()),
                }),
            },
        ]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
        assert_eq!(calls[1].arguments["x"], 1);
    }

    #[test]
    fn tool_call_accumulator_empty() {
        let acc = ToolCallAccumulator::new();
        assert!(acc.is_empty());
        assert!(acc.finish().is_empty());
    }

    #[test]
    fn tool_call_accumulator_invalid_json_args() {
        let mut acc = ToolCallAccumulator::new();
        acc.process_delta(&[OaiStreamToolCall {
            index: 0,
            id: Some("c1".into()),
            function: Some(OaiStreamFunction {
                name: Some("tool".into()),
                arguments: Some("not valid json".into()),
            }),
        }]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        // Falls back to empty object
        assert!(calls[0].arguments.is_object());
    }
}
