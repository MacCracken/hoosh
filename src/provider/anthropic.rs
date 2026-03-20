//! Anthropic provider — uses the Anthropic Messages API (not OpenAI-compatible).

use std::time::Instant;

use serde::Deserialize;
use tokio::sync::mpsc;

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo, Role, TokenUsage};
use crate::provider::{LlmProvider, ProviderType};

const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic provider using the Messages API.
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    api_version: String,
}

impl AnthropicProvider {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "https://api.anthropic.com".to_string()
        } else {
            url.trim_end_matches('/').to_string()
        };
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url: url,
            api_key,
            api_version: std::env::var("ANTHROPIC_API_VERSION")
                .unwrap_or_else(|_| DEFAULT_ANTHROPIC_VERSION.to_string()),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn build_request(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut rb = rb.header("anthropic-version", &self.api_version);
        if let Some(key) = &self.api_key {
            rb = rb.header("x-api-key", key);
        }
        rb
    }
}

fn build_anthropic_body(req: &InferenceRequest, stream: bool) -> serde_json::Value {
    let mut system_text: Option<String> = None;
    let messages: Vec<serde_json::Value> = if req.messages.is_empty() {
        if let Some(sys) = &req.system {
            system_text = Some(sys.clone());
        }
        vec![serde_json::json!({"role": "user", "content": req.prompt})]
    } else {
        let mut msgs = Vec::new();
        for m in &req.messages {
            match m.role {
                Role::System => {
                    system_text = Some(m.content.clone());
                }
                Role::User => {
                    msgs.push(serde_json::json!({"role": "user", "content": m.content}));
                }
                Role::Assistant => {
                    msgs.push(serde_json::json!({"role": "assistant", "content": m.content}));
                }
            }
        }
        msgs
    };

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens.unwrap_or(1024),
        "stream": stream,
    });

    if let Some(sys) = system_text {
        body["system"] = serde_json::json!(sys);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(tp) = req.top_p {
        body["top_p"] = serde_json::json!(tp);
    }

    body
}

// ---- Response deserialization ----

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    model: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<StreamDelta>,
}

#[derive(Deserialize)]
struct StreamDelta {
    text: Option<String>,
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = build_anthropic_body(request, false);

        let start = Instant::now();
        let rb = self.build_request(self.client.post(&url).json(&body));
        let resp = rb.send().await?.error_for_status()?;
        let parsed: AnthropicResponse = resp.json().await?;
        let latency = start.elapsed().as_millis() as u64;

        let text = parsed
            .content
            .iter()
            .filter_map(|b| b.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("");

        let usage = parsed.usage.as_ref();
        let input = usage.and_then(|u| u.input_tokens).unwrap_or(0);
        let output = usage.and_then(|u| u.output_tokens).unwrap_or(0);

        Ok(InferenceResponse {
            text,
            model: parsed.model.unwrap_or_else(|| request.model.clone()),
            usage: TokenUsage {
                prompt_tokens: input,
                completion_tokens: output,
                total_tokens: input + output,
            },
            provider: "anthropic".into(),
            latency_ms: latency,
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<mpsc::Receiver<anyhow::Result<String>>> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = build_anthropic_body(&request, true);

        let rb = self.build_request(self.client.post(&url).json(&body));
        let resp = rb.send().await?.error_for_status()?;

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

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(e.into())).await;
                        return;
                    }
                };
                if buf.len() + chunk.len() > 1024 * 1024 {
                    let _ = tx
                        .send(Err(anyhow::anyhow!("SSE line exceeded 1MB limit")))
                        .await;
                    return;
                }
                buf.push_str(&String::from_utf8_lossy(&chunk));

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
                        return;
                    }
                    if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                        if event.event_type == "content_block_delta"
                            && let Some(delta) = &event.delta
                            && let Some(text) = &delta.text
                            && !text.is_empty()
                            && tx.send(Ok(text.clone())).await.is_err()
                        {
                            return;
                        }
                        if event.event_type == "message_stop" {
                            return;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        // Anthropic doesn't have a models endpoint — return known models
        Ok(vec![
            ModelInfo {
                id: "claude-opus-4-20250514".into(),
                name: "Claude Opus 4".into(),
                provider: "anthropic".into(),
                parameters: None,
                context_length: Some(200_000),
                available: true,
            },
            ModelInfo {
                id: "claude-sonnet-4-20250514".into(),
                name: "Claude Sonnet 4".into(),
                provider: "anthropic".into(),
                parameters: None,
                context_length: Some(200_000),
                available: true,
            },
            ModelInfo {
                id: "claude-haiku-4-20250514".into(),
                name: "Claude Haiku 4".into(),
                provider: "anthropic".into(),
                parameters: None,
                context_length: Some(200_000),
                available: true,
            },
        ])
    }

    async fn health_check(&self) -> anyhow::Result<bool> {
        // HEAD request to check reachability without consuming tokens
        let url = format!("{}/v1/messages", self.base_url);
        let mut rb = self.client.post(&url).header("content-length", "0");
        if let Some(key) = &self.api_key {
            rb = rb.header("x-api-key", key);
        }
        rb = rb.header("anthropic-version", &self.api_version);
        match rb.send().await {
            Ok(resp) => {
                // Any response (including 400/401/422) means the endpoint is reachable
                let status = resp.status().as_u16();
                Ok(status != 404 && status != 502 && status != 503)
            }
            Err(_) => Ok(false),
        }
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Message;

    #[test]
    fn default_url() {
        let p = AnthropicProvider::new("", Some("sk-ant-test".into()));
        assert_eq!(p.base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn custom_url() {
        let p = AnthropicProvider::new("https://proxy.example.com", None);
        assert_eq!(p.base_url(), "https://proxy.example.com");
    }

    #[test]
    fn provider_type_is_anthropic() {
        let p = AnthropicProvider::new("", None);
        assert_eq!(p.provider_type(), ProviderType::Anthropic);
    }

    #[test]
    fn build_body_from_prompt() {
        let req = InferenceRequest {
            model: "claude-sonnet-4-20250514".into(),
            prompt: "Hello".into(),
            system: Some("Be helpful.".into()),
            ..Default::default()
        };
        let body = build_anthropic_body(&req, false);
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["system"], "Be helpful.");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn build_body_from_messages() {
        let req = InferenceRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: "Be concise.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Hi".into(),
                },
                Message {
                    role: Role::Assistant,
                    content: "Hello!".into(),
                },
                Message {
                    role: Role::User,
                    content: "More".into(),
                },
            ],
            max_tokens: Some(500),
            ..Default::default()
        };
        let body = build_anthropic_body(&req, false);
        assert_eq!(body["system"], "Be concise.");
        let msgs = body["messages"].as_array().unwrap();
        // System message extracted, not in messages array
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(body["max_tokens"], 500);
    }

    #[test]
    fn build_body_default_max_tokens() {
        let req = InferenceRequest {
            model: "claude-sonnet-4-20250514".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let body = build_anthropic_body(&req, false);
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hello world"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello world"));
        assert_eq!(resp.usage.unwrap().input_tokens, Some(10));
    }

    #[test]
    fn stream_event_deserialization() {
        let json =
            r#"{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.delta.unwrap().text.as_deref(), Some("Hello"));
    }

    #[test]
    fn list_models_returns_known() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let p = AnthropicProvider::new("", None);
        let models = rt.block_on(p.list_models()).unwrap();
        assert!(models.len() >= 3);
        assert!(models.iter().any(|m| m.id.contains("opus")));
        assert!(models.iter().any(|m| m.id.contains("sonnet")));
    }
}
