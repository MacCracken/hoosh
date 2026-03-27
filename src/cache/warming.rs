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
                Ok((_returned_key, response_text)) => {
                    cache.insert(key, response_text);
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

    #[test]
    fn warming_prompt_with_assistant_role() {
        let prompts = vec![WarmingPrompt {
            model: "gpt-4o".into(),
            messages: vec![
                WarmingMessage {
                    role: "user".into(),
                    content: "Hello".into(),
                },
                WarmingMessage {
                    role: "assistant".into(),
                    content: "Hi there!".into(),
                },
                WarmingMessage {
                    role: "user".into(),
                    content: "How are you?".into(),
                },
            ],
        }];
        let requests = to_inference_requests(&prompts);
        assert_eq!(requests[0].messages.len(), 3);
        assert_eq!(requests[0].messages[1].role, Role::Assistant);
    }

    #[test]
    fn warming_prompt_unknown_role_defaults_to_user() {
        let prompts = vec![WarmingPrompt {
            model: "test".into(),
            messages: vec![WarmingMessage {
                role: "custom-role".into(),
                content: "test".into(),
            }],
        }];
        let requests = to_inference_requests(&prompts);
        assert_eq!(requests[0].messages[0].role, Role::User);
    }

    #[test]
    fn warming_prompt_multiple() {
        let prompts = vec![
            WarmingPrompt {
                model: "llama3".into(),
                messages: vec![WarmingMessage {
                    role: "user".into(),
                    content: "Hello".into(),
                }],
            },
            WarmingPrompt {
                model: "gpt-4o".into(),
                messages: vec![WarmingMessage {
                    role: "user".into(),
                    content: "World".into(),
                }],
            },
        ];
        let requests = to_inference_requests(&prompts);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].model, "llama3");
        assert_eq!(requests[1].model, "gpt-4o");
    }

    #[tokio::test]
    async fn spawn_warming_task_empty_prompts_returns_early() {
        let cache = Arc::new(crate::cache::ResponseCache::new(
            crate::cache::CacheConfig::default(),
        ));
        // Should return immediately without spawning
        spawn_warming_task(vec![], cache, |_req| async {
            Ok(("key".to_string(), "value".to_string()))
        });
        // No panic means it worked
    }

    #[tokio::test]
    async fn spawn_warming_task_populates_cache() {
        let cache = Arc::new(crate::cache::ResponseCache::new(
            crate::cache::CacheConfig::default(),
        ));
        let prompts = vec![WarmingPrompt {
            model: "test-model".into(),
            messages: vec![WarmingMessage {
                role: "user".into(),
                content: "warm me up".into(),
            }],
        }];
        let cache_clone = cache.clone();
        spawn_warming_task(prompts, cache_clone, |_req| async {
            Ok(("key".to_string(), "warmed response".to_string()))
        });
        // Wait for the background task to complete
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // The cache should have an entry now
        let stats = cache.stats();
        assert!(stats.entries > 0);
    }

    #[tokio::test]
    async fn spawn_warming_task_handles_inference_error() {
        let cache = Arc::new(crate::cache::ResponseCache::new(
            crate::cache::CacheConfig::default(),
        ));
        let prompts = vec![WarmingPrompt {
            model: "fail-model".into(),
            messages: vec![WarmingMessage {
                role: "user".into(),
                content: "fail".into(),
            }],
        }];
        spawn_warming_task(prompts, cache.clone(), |_req| async {
            Err(anyhow::anyhow!("inference failed"))
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Cache should be empty since inference failed
        assert_eq!(cache.stats().entries, 0);
    }
}
