//! Cache warming — pre-populate cache on startup with common prompts.
//!
//! Reads warming prompts from config and fires inference requests in the
//! background to populate the cache before user traffic arrives.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::inference::{InferenceRequest, Message, Role};

/// A prompt to warm in the cache on startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingPrompt {
    /// Model to use for the warming request.
    pub model: String,
    /// Messages to send (typically a single user message).
    pub messages: Vec<WarmingMessage>,
}

/// A single message in a warming prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingMessage {
    pub role: String,
    pub content: String,
}

/// Convert warming prompts into inference requests.
#[must_use]
pub fn to_inference_requests(prompts: &[WarmingPrompt]) -> Vec<InferenceRequest> {
    prompts
        .iter()
        .map(|p| InferenceRequest {
            model: p.model.clone(),
            messages: p
                .messages
                .iter()
                .map(|m| {
                    Message::new(
                        match m.role.as_str() {
                            "system" => Role::System,
                            "assistant" => Role::Assistant,
                            _ => Role::User,
                        },
                        &m.content,
                    )
                })
                .collect(),
            ..Default::default()
        })
        .collect()
}

/// Spawn a background task that warms the cache with pre-configured prompts.
///
/// This function is fire-and-forget — failures are logged but don't block startup.
pub fn spawn_warming_task<F, Fut>(
    prompts: Vec<WarmingPrompt>,
    cache: Arc<crate::cache::ResponseCache>,
    infer_fn: F,
) where
    F: Fn(InferenceRequest) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = anyhow::Result<(String, String)>> + Send + 'static,
{
    if prompts.is_empty() {
        return;
    }

    let count = prompts.len();
    tracing::info!("warming cache with {count} prompts");

    let infer_fn = Arc::new(infer_fn);
    tokio::spawn(async move {
        let requests = to_inference_requests(&prompts);
        let mut warmed = 0usize;

        for req in requests {
            let key = crate::cache::cache_key(&req.model, &req.messages);
            // Skip if already cached
            if cache.get(&key).is_some() {
                continue;
            }

            let f = infer_fn.clone();
            match f(req).await {
                Ok((cache_key, response_text)) => {
                    cache.insert(cache_key, response_text);
                    warmed += 1;
                }
                Err(e) => {
                    tracing::warn!("cache warming failed: {e}");
                }
            }
        }

        tracing::info!("cache warming complete: {warmed}/{count} entries cached");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warming_prompt_to_request() {
        let prompts = vec![WarmingPrompt {
            model: "llama3".into(),
            messages: vec![WarmingMessage {
                role: "user".into(),
                content: "Hello".into(),
            }],
        }];
        let requests = to_inference_requests(&prompts);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "llama3");
        assert_eq!(requests[0].messages.len(), 1);
        assert_eq!(requests[0].messages[0].role, Role::User);
    }

    #[test]
    fn warming_prompt_with_system() {
        let prompts = vec![WarmingPrompt {
            model: "gpt-4o".into(),
            messages: vec![
                WarmingMessage {
                    role: "system".into(),
                    content: "You are a helper.".into(),
                },
                WarmingMessage {
                    role: "user".into(),
                    content: "Hi".into(),
                },
            ],
        }];
        let requests = to_inference_requests(&prompts);
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[0].messages[0].role, Role::System);
    }

    #[test]
    fn empty_prompts_no_requests() {
        let requests = to_inference_requests(&[]);
        assert!(requests.is_empty());
    }
}
