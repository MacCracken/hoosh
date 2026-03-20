//! OpenAI-compatible base provider for llama.cpp, LM Studio, LocalAI, Synapse.

use std::time::Instant;

use serde::Deserialize;
use tokio::sync::mpsc;

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo, Role, TokenUsage};
use crate::provider::{LlmProvider, ProviderType};

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
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
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
}

#[derive(Deserialize)]
struct OaiDelta {
    content: Option<String>,
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

        let text = oai
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
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

        let (tx, rx) = mpsc::channel(64);

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
                        return;
                    }
                    if let Ok(chunk) = serde_json::from_str::<OaiStreamChunk>(data) {
                        for choice in &chunk.choices {
                            if let Some(content) = &choice.delta.content {
                                if !content.is_empty() {
                                    if tx.send(Ok(content.clone())).await.is_err() {
                                        return;
                                    }
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

    fn provider_type(&self) -> ProviderType {
        self.provider_type
    }
}
