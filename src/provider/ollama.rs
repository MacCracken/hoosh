//! Ollama provider — native REST API (not OpenAI-compatible).

use std::time::Instant;

use serde::Deserialize;
use tokio::sync::mpsc;

use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo, Role, TokenUsage};
use crate::provider::{LlmProvider, ProviderType};

/// Ollama provider using its native `/api/` endpoints.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        let url = base_url.into();
        let url = if url.is_empty() {
            "http://localhost:11434".to_string()
        } else {
            url.trim_end_matches('/').to_string()
        };
        Self {
            client: reqwest::Client::new(),
            base_url: url,
        }
    }

    /// Pull a model from the Ollama registry.
    pub async fn pull_model(&self, name: &str) -> anyhow::Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        self.client
            .post(&url)
            .json(&serde_json::json!({"name": name}))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Delete a model from Ollama.
    pub async fn delete_model(&self, name: &str) -> anyhow::Result<()> {
        let url = format!("{}/api/delete", self.base_url);
        self.client
            .delete(&url)
            .json(&serde_json::json!({"name": name}))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

fn build_messages(req: &InferenceRequest) -> Vec<serde_json::Value> {
    if req.messages.is_empty() {
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
    }
}

// ---- Response types ----

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Option<OllamaMessage>,
    eval_count: Option<u32>,
    prompt_eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct OllamaStreamLine {
    message: Option<OllamaMessage>,
    done: bool,
    eval_count: Option<u32>,
    prompt_eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
    size: Option<u64>,
}

#[async_trait::async_trait]
impl LlmProvider for OllamaProvider {
    async fn infer(&self, request: &InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let messages = build_messages(request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "stream": false,
        });
        if let Some(temp) = request.temperature {
            body["options"] = serde_json::json!({"temperature": temp});
        }

        let start = Instant::now();
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let oai: OllamaChatResponse = resp.json().await?;
        let latency = start.elapsed().as_millis() as u64;

        let text = oai
            .message
            .and_then(|m| m.content)
            .unwrap_or_default();

        let completion_tokens = oai.eval_count.unwrap_or(0);
        let prompt_tokens = oai.prompt_eval_count.unwrap_or(0);

        Ok(InferenceResponse {
            text,
            model: request.model.clone(),
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            provider: "ollama".into(),
            latency_ms: latency,
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<mpsc::Receiver<anyhow::Result<String>>> {
        let url = format!("{}/api/chat", self.base_url);
        let messages = build_messages(&request);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "stream": true,
        });
        if let Some(temp) = request.temperature {
            body["options"] = serde_json::json!({"temperature": temp});
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

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

                // NDJSON: each line is a complete JSON object
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf = buf[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<OllamaStreamLine>(&line) {
                        Ok(parsed) => {
                            if let Some(msg) = &parsed.message {
                                if let Some(content) = &msg.content {
                                    if !content.is_empty() {
                                        if tx.send(Ok(content.clone())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                            if parsed.done {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e.into())).await;
                            return;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?.error_for_status()?;
        let tags: OllamaTagsResponse = resp.json().await?;

        Ok(tags
            .models
            .into_iter()
            .map(|m| ModelInfo {
                id: m.name.clone(),
                name: m.name,
                provider: "ollama".into(),
                parameters: m.size,
                context_length: None,
                available: true,
            })
            .collect())
    }

    async fn health_check(&self) -> anyhow::Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;
        Ok(resp.status().is_success())
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Ollama
    }
}
