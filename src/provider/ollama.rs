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

        let text = oai.message.and_then(|m| m.content).unwrap_or_default();

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
                            if let Some(msg) = &parsed.message
                                && let Some(content) = &msg.content
                                && !content.is_empty()
                                && tx.send(Ok(content.clone())).await.is_err()
                            {
                                return;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Message;

    #[test]
    fn default_url() {
        let p = OllamaProvider::new("");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn custom_url() {
        let p = OllamaProvider::new("http://my-ollama:9999");
        assert_eq!(p.base_url, "http://my-ollama:9999");
    }

    #[test]
    fn strips_trailing_slash() {
        let p = OllamaProvider::new("http://localhost:11434/");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn provider_type_is_ollama() {
        let p = OllamaProvider::new("");
        assert_eq!(p.provider_type(), ProviderType::Ollama);
    }

    #[test]
    fn messages_from_prompt() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            ..Default::default()
        };
        let msgs = build_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
    }

    #[test]
    fn messages_from_prompt_with_system() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            system: Some("Be helpful.".into()),
            ..Default::default()
        };
        let msgs = build_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn messages_from_conversation() {
        let req = InferenceRequest {
            messages: vec![
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
                    content: "How are you?".into(),
                },
            ],
            ..Default::default()
        };
        let msgs = build_messages(&req);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], "How are you?");
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "message": {"content": "Hello world"},
            "eval_count": 5,
            "prompt_eval_count": 10
        }"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.unwrap().content.unwrap(), "Hello world");
        assert_eq!(resp.eval_count, Some(5));
        assert_eq!(resp.prompt_eval_count, Some(10));
    }

    #[test]
    fn response_deserialization_minimal() {
        let json = r#"{"message": null}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.message.is_none());
        assert!(resp.eval_count.is_none());
    }

    #[test]
    fn stream_line_deserialization() {
        let json = r#"{"message": {"content": "tok"}, "done": false}"#;
        let line: OllamaStreamLine = serde_json::from_str(json).unwrap();
        assert!(!line.done);
        assert_eq!(line.message.unwrap().content.unwrap(), "tok");
    }

    #[test]
    fn stream_line_done() {
        let json = r#"{"message": {"content": ""}, "done": true, "eval_count": 42, "prompt_eval_count": 10}"#;
        let line: OllamaStreamLine = serde_json::from_str(json).unwrap();
        assert!(line.done);
        assert_eq!(line.eval_count, Some(42));
    }

    #[test]
    fn tags_response_deserialization() {
        let json = r#"{"models": [{"name": "llama3:latest", "size": 4000000000}, {"name": "mistral:7b"}]}"#;
        let resp: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 2);
        assert_eq!(resp.models[0].name, "llama3:latest");
        assert_eq!(resp.models[0].size, Some(4000000000));
        assert!(resp.models[1].size.is_none());
    }
}
