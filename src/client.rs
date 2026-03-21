//! HTTP client for talking to a hoosh server.
//!
//! This is what downstream crates (tarang, daimon, consumer apps) use to
//! call hoosh over the network. OpenAI-compatible API.

use serde::Deserialize;
use tokio::sync::mpsc;

use crate::error::{HooshError, Result};
use crate::inference::{InferenceRequest, InferenceResponse, ModelInfo, Role, TokenUsage};

/// HTTP client for the hoosh inference gateway.
#[derive(Debug, Clone)]
pub struct HooshClient {
    base_url: String,
    client: reqwest::Client,
}

/// Build an OpenAI-compatible chat body from an InferenceRequest.
fn to_chat_body(request: &InferenceRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = if request.messages.is_empty() {
        let mut msgs = Vec::new();
        if let Some(sys) = &request.system {
            msgs.push(serde_json::json!({"role": "system", "content": sys}));
        }
        msgs.push(serde_json::json!({"role": "user", "content": request.prompt}));
        msgs
    } else {
        request
            .messages
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
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    });
    if let Some(max) = request.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(tp) = request.top_p {
        body["top_p"] = serde_json::json!(tp);
    }
    body
}

#[derive(Deserialize)]
struct ChatCompletionResp {
    model: Option<String>,
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsageResp>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMsg,
}

#[derive(Deserialize)]
struct ChatMsg {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatUsageResp {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResp {
    data: Vec<ModelObj>,
}

#[derive(Deserialize)]
struct ModelObj {
    id: String,
    owned_by: Option<String>,
}

impl HooshClient {
    /// Create a new client pointing at the given hoosh server.
    ///
    /// The client is tuned for low-latency local connections:
    /// - TCP_NODELAY disables Nagle's algorithm (avoids 40ms batching delay)
    /// - Connection pooling keeps TCP connections alive across requests
    /// - HTTP/2 adaptive window for multiplexed requests
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .tcp_nodelay(true)
                .tcp_keepalive(std::time::Duration::from_secs(60))
                .pool_idle_timeout(std::time::Duration::from_secs(600))
                .pool_max_idle_per_host(32)
                .http2_adaptive_window(true)
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Run inference via the hoosh server.
    pub async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = to_chat_body(&InferenceRequest {
            stream: false,
            ..request.clone()
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        let parsed: ChatCompletionResp = resp
            .json()
            .await
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        let text = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = parsed.usage.as_ref();
        Ok(InferenceResponse {
            text,
            model: parsed.model.unwrap_or_else(|| request.model.clone()),
            usage: TokenUsage {
                prompt_tokens: usage.and_then(|u| u.prompt_tokens).unwrap_or(0),
                completion_tokens: usage.and_then(|u| u.completion_tokens).unwrap_or(0),
                total_tokens: usage.and_then(|u| u.total_tokens).unwrap_or(0),
            },
            provider: "hoosh".into(),
            latency_ms: 0,
        })
    }

    /// Stream inference results token by token.
    pub async fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<mpsc::Receiver<std::result::Result<String, HooshError>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = to_chat_body(&InferenceRequest {
            stream: true,
            ..request.clone()
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        if let Some(ct) = resp.headers().get("content-type") {
            let ct_str = ct.to_str().unwrap_or("");
            if !ct_str.contains("text/event-stream") && !ct_str.contains("application/json") {
                return Err(HooshError::Provider(format!(
                    "expected SSE stream, got Content-Type: {ct_str}"
                )));
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
                        let _ = tx.send(Err(HooshError::Provider(e.to_string()))).await;
                        return;
                    }
                };
                if buf.len() + chunk.len() > 1024 * 1024 {
                    let _ = tx
                        .send(Err(HooshError::Provider(
                            "SSE line exceeded 1MB limit".into(),
                        )))
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
                    } else if let Some(d) = line.strip_prefix("data:") {
                        d.trim()
                    } else {
                        continue;
                    };
                    if data == "[DONE]" {
                        return;
                    }
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                        for choice in &chunk.choices {
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
        });

        Ok(rx)
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        let parsed: ModelsResp = resp
            .json()
            .await
            .map_err(|e| HooshError::Provider(e.to_string()))?;

        Ok(parsed
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
                provider: m.owned_by.unwrap_or_else(|| "hoosh".into()),
                parameters: None,
                context_length: None,
                available: true,
            })
            .collect())
    }

    /// Health check.
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/v1/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Base URL of the hoosh server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_creation() {
        let client = HooshClient::new("http://localhost:8088");
        assert_eq!(client.base_url(), "http://localhost:8088");
    }

    #[test]
    fn client_strips_trailing_slash() {
        let client = HooshClient::new("http://localhost:8088/");
        assert_eq!(client.base_url(), "http://localhost:8088");
    }
}
