//! Integration tests for hoosh.

use crate::*;

#[test]
fn client_creation() {
    let client = HooshClient::new("http://localhost:8088");
    assert_eq!(client.base_url(), "http://localhost:8088");
}

#[test]
fn inference_request_roundtrip() {
    let req = InferenceRequest {
        model: "llama3".into(),
        prompt: "Explain Rust.".into(),
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: InferenceRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model, "llama3");
    assert_eq!(back.max_tokens, Some(100));
}

#[test]
fn provider_types_exhaustive() {
    // Verify all providers have Display
    let providers = [
        ProviderType::Ollama,
        ProviderType::LlamaCpp,
        ProviderType::Synapse,
        ProviderType::LmStudio,
        ProviderType::LocalAi,
        ProviderType::OpenAi,
        ProviderType::Anthropic,
        ProviderType::DeepSeek,
        ProviderType::Mistral,
        ProviderType::Google,
        ProviderType::Groq,
        ProviderType::Grok,
        ProviderType::OpenRouter,
        ProviderType::Whisper,
    ];
    for p in &providers {
        assert!(!p.to_string().is_empty());
    }
    assert_eq!(providers.len(), 14);
}

#[test]
fn router_selects_provider() {
    use crate::router::*;
    let routes = vec![
        ProviderRoute {
            provider: ProviderType::Ollama,
            priority: 1,
            model_patterns: vec!["llama*".into()],
            enabled: true,
            base_url: "http://localhost:11434".into(),
            api_key: None,
            max_tokens_limit: None,
        },
        ProviderRoute {
            provider: ProviderType::OpenAi,
            priority: 2,
            model_patterns: vec!["gpt-*".into()],
            enabled: true,
            base_url: "https://api.openai.com".into(),
            api_key: None,
            max_tokens_limit: None,
        },
    ];
    let router = Router::new(routes, RoutingStrategy::Priority);
    assert_eq!(
        router.select("llama3").unwrap().provider,
        ProviderType::Ollama
    );
    assert_eq!(
        router.select("gpt-4o").unwrap().provider,
        ProviderType::OpenAi
    );
    assert!(router.select("claude-sonnet-4-20250514").is_none());
}

#[test]
fn token_budget_lifecycle() {
    let mut budget = TokenBudget::new();
    budget.add_pool(TokenPool::new("agent-1", 10000));

    // Reserve
    assert!(budget.reserve("agent-1", 2000));
    assert_eq!(budget.get_pool("agent-1").unwrap().available(), 8000);

    // Complete with actual usage
    budget.report("agent-1", 2000, 1500);
    let pool = budget.get_pool("agent-1").unwrap();
    assert_eq!(pool.used, 1500);
    assert_eq!(pool.available(), 8500);
}

#[test]
fn cache_basic() {
    let cache = ResponseCache::new(crate::cache::CacheConfig::default());
    cache.insert("prompt-hash".into(), "cached response".into());
    assert_eq!(&*cache.get("prompt-hash").unwrap(), "cached response");
    assert!(cache.get("missing").is_none());
}

#[test]
fn error_types() {
    let err = HooshError::ModelNotFound("gpt-5".into());
    assert!(err.to_string().contains("gpt-5"));

    let err = HooshError::BudgetExceeded {
        pool: "default".into(),
        remaining: 42,
    };
    assert!(err.to_string().contains("42"));
}

// ---------------------------------------------------------------------------
// ProviderRegistry
// ---------------------------------------------------------------------------

#[test]
fn provider_registry_empty() {
    let registry = ProviderRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn provider_registry_default() {
    let registry = ProviderRegistry::default();
    assert!(registry.is_empty());
}

#[test]
fn provider_registry_get_missing() {
    let registry = ProviderRegistry::new();
    assert!(
        registry
            .get(ProviderType::Ollama, "http://localhost:11434")
            .is_none()
    );
}

#[cfg(feature = "ollama")]
#[test]
fn provider_registry_register_ollama() {
    use crate::router::ProviderRoute;
    let mut registry = ProviderRegistry::new();
    let route = ProviderRoute {
        provider: ProviderType::Ollama,
        priority: 1,
        model_patterns: vec!["llama*".into()],
        enabled: true,
        base_url: "http://localhost:11434".into(),
        api_key: None,
        max_tokens_limit: None,
    };
    registry.register_from_route(&route);
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());

    let provider = registry
        .get(ProviderType::Ollama, "http://localhost:11434")
        .unwrap();
    assert_eq!(provider.provider_type(), ProviderType::Ollama);
}

#[cfg(feature = "ollama")]
#[test]
fn provider_registry_dedup() {
    use crate::router::ProviderRoute;
    let mut registry = ProviderRegistry::new();
    let route = ProviderRoute {
        provider: ProviderType::Ollama,
        priority: 1,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://localhost:11434".into(),
        api_key: None,
        max_tokens_limit: None,
    };
    registry.register_from_route(&route);
    registry.register_from_route(&route);
    assert_eq!(registry.len(), 1);
}

#[cfg(all(feature = "ollama", feature = "llamacpp"))]
#[test]
fn provider_registry_multiple_providers() {
    use crate::router::ProviderRoute;
    let mut registry = ProviderRegistry::new();
    registry.register_from_route(&ProviderRoute {
        provider: ProviderType::Ollama,
        priority: 1,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://localhost:11434".into(),
        api_key: None,
        max_tokens_limit: None,
    });
    registry.register_from_route(&ProviderRoute {
        provider: ProviderType::LlamaCpp,
        priority: 2,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://localhost:8080".into(),
        api_key: None,
        max_tokens_limit: None,
    });
    assert_eq!(registry.len(), 2);

    // Different base_urls for same type = different entries
    registry.register_from_route(&ProviderRoute {
        provider: ProviderType::Ollama,
        priority: 1,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://other-host:11434".into(),
        api_key: None,
        max_tokens_limit: None,
    });
    assert_eq!(registry.len(), 3);
}

#[test]
fn provider_registry_unrecognized_type_not_registered() {
    use crate::router::ProviderRoute;
    let mut registry = ProviderRegistry::new();
    // Whisper is not handled by register_from_route
    let route = ProviderRoute {
        provider: ProviderType::Whisper,
        priority: 1,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://localhost:9999".into(),
        api_key: None,
        max_tokens_limit: None,
    };
    registry.register_from_route(&route);
    assert!(registry.is_empty());
}

#[test]
fn provider_registry_all_iterator() {
    let registry = ProviderRegistry::new();
    assert_eq!(registry.all().count(), 0);
}

// ---------------------------------------------------------------------------
// Mock HTTP server integration tests
// ---------------------------------------------------------------------------

/// Spin up a fake OpenAI-compatible server and test providers against it.
mod mock_server {
    use axum::{
        Json, Router,
        routing::{get, post},
    };
    use serde_json::json;
    use tokio::net::TcpListener;

    use crate::inference::InferenceRequest;
    use crate::provider::LlmProvider;
    use crate::provider::ProviderType;
    use crate::provider::openai_compat::OpenAiCompatibleProvider;

    /// Start a mock server that responds to OpenAI-compatible endpoints.
    /// Returns the base URL (e.g. "http://127.0.0.1:PORT").
    async fn start_mock_oai_server() -> String {
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat_completions))
            .route("/v1/models", get(mock_models));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        format!("http://127.0.0.1:{}", addr.port())
    }

    async fn mock_chat_completions(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
        let model = body["model"].as_str().unwrap_or("mock-model");
        let stream = body["stream"].as_bool().unwrap_or(false);

        // We only handle non-streaming in this mock
        assert!(!stream, "mock does not handle streaming");

        Json(json!({
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Mock response from server"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
    }

    async fn mock_models() -> Json<serde_json::Value> {
        Json(json!({
            "object": "list",
            "data": [
                {"id": "mock-model-1", "object": "model", "owned_by": "mock"},
                {"id": "mock-model-2", "object": "model", "owned_by": "mock"}
            ]
        }))
    }

    #[tokio::test]
    async fn openai_compat_infer() {
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LlamaCpp);

        let req = InferenceRequest {
            model: "test-model".into(),
            prompt: "Hello".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();

        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.model, "test-model");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
        assert_eq!(resp.provider, "llamacpp");
        assert!(resp.latency_ms < 5000);
    }

    #[tokio::test]
    async fn openai_compat_infer_with_messages() {
        use crate::inference::{Message, Role};

        let base_url = start_mock_oai_server().await;
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LocalAi);

        let req = InferenceRequest {
            model: "chat-model".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: "Be helpful.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Hi".into(),
                },
            ],
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "localai");
    }

    #[tokio::test]
    async fn openai_compat_list_models() {
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LmStudio);

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "mock-model-1");
        assert_eq!(models[0].provider, "mock");
        assert_eq!(models[1].id, "mock-model-2");
        assert!(models[0].available);
    }

    #[tokio::test]
    async fn openai_compat_health_check() {
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LlamaCpp);

        let healthy = provider.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn openai_compat_health_check_unreachable() {
        let provider =
            OpenAiCompatibleProvider::new("http://127.0.0.1:1", None, ProviderType::LlamaCpp);
        // Should return an error (connection refused), not panic
        let result = provider.health_check().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn openai_compat_infer_unreachable() {
        let provider =
            OpenAiCompatibleProvider::new("http://127.0.0.1:1", None, ProviderType::LlamaCpp);
        let req = InferenceRequest {
            model: "test".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let result = provider.infer(&req).await;
        assert!(result.is_err());
    }

    /// Start a mock Ollama server and test the Ollama provider.
    #[cfg(feature = "ollama")]
    mod ollama_mock {
        use axum::{
            Json, Router,
            routing::{get, post},
        };
        use serde_json::json;
        use tokio::net::TcpListener;

        use crate::inference::InferenceRequest;
        use crate::provider::LlmProvider;
        use crate::provider::ollama::OllamaProvider;

        async fn start_mock_ollama() -> String {
            let app = Router::new()
                .route("/api/chat", post(mock_chat))
                .route("/api/tags", get(mock_tags));

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            format!("http://127.0.0.1:{}", addr.port())
        }

        async fn mock_chat(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
            let stream = body["stream"].as_bool().unwrap_or(false);
            assert!(!stream, "mock does not handle streaming");

            Json(json!({
                "message": {"role": "assistant", "content": "Ollama mock reply"},
                "eval_count": 8,
                "prompt_eval_count": 12
            }))
        }

        async fn mock_tags() -> Json<serde_json::Value> {
            Json(json!({
                "models": [
                    {"name": "llama3:latest", "size": 4_000_000_000_i64},
                    {"name": "mistral:7b"}
                ]
            }))
        }

        #[tokio::test]
        async fn ollama_infer() {
            let base_url = start_mock_ollama().await;
            let provider = OllamaProvider::new(&base_url);

            let req = InferenceRequest {
                model: "llama3".into(),
                prompt: "Hello".into(),
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Ollama mock reply");
            assert_eq!(resp.model, "llama3");
            assert_eq!(resp.usage.prompt_tokens, 12);
            assert_eq!(resp.usage.completion_tokens, 8);
            assert_eq!(resp.usage.total_tokens, 20);
            assert_eq!(resp.provider, "ollama");
        }

        #[tokio::test]
        async fn ollama_infer_with_temperature() {
            let base_url = start_mock_ollama().await;
            let provider = OllamaProvider::new(&base_url);

            let req = InferenceRequest {
                model: "llama3".into(),
                prompt: "Hello".into(),
                temperature: Some(0.3),
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Ollama mock reply");
        }

        #[tokio::test]
        async fn ollama_list_models() {
            let base_url = start_mock_ollama().await;
            let provider = OllamaProvider::new(&base_url);

            let models = provider.list_models().await.unwrap();
            assert_eq!(models.len(), 2);
            assert_eq!(models[0].id, "llama3:latest");
            assert_eq!(models[0].parameters, Some(4000000000));
            assert_eq!(models[0].provider, "ollama");
            assert!(models[0].available);
            assert_eq!(models[1].id, "mistral:7b");
            assert!(models[1].parameters.is_none());
        }

        #[tokio::test]
        async fn ollama_health_check() {
            let base_url = start_mock_ollama().await;
            let provider = OllamaProvider::new(&base_url);

            let healthy = provider.health_check().await.unwrap();
            assert!(healthy);
        }

        #[tokio::test]
        async fn ollama_health_check_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1");
            let result = provider.health_check().await;
            assert!(result.is_err());
        }
    }

    /// Mock Anthropic server.
    #[cfg(feature = "anthropic")]
    mod anthropic_mock {
        use axum::{Json, Router, routing::post};
        use serde_json::json;
        use tokio::net::TcpListener;

        use crate::inference::InferenceRequest;
        use crate::provider::LlmProvider;
        use crate::provider::anthropic::AnthropicProvider;

        async fn start_mock_anthropic() -> String {
            async fn mock_messages(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
                let model = body["model"].as_str().unwrap_or("claude-sonnet-4-20250514");
                Json(json!({
                    "id": "msg-mock",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Anthropic mock response"}],
                    "model": model,
                    "usage": {"input_tokens": 15, "output_tokens": 8}
                }))
            }

            let app = Router::new().route("/v1/messages", post(mock_messages));
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://127.0.0.1:{}", addr.port())
        }

        #[tokio::test]
        async fn anthropic_infer() {
            let base_url = start_mock_anthropic().await;
            let provider = AnthropicProvider::new(&base_url, Some("sk-ant-test".into()));

            let req = InferenceRequest {
                model: "claude-sonnet-4-20250514".into(),
                prompt: "Hello".into(),
                system: Some("Be helpful.".into()),
                max_tokens: Some(100),
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Anthropic mock response");
            assert_eq!(resp.provider, "anthropic");
            assert_eq!(resp.usage.prompt_tokens, 15);
            assert_eq!(resp.usage.completion_tokens, 8);
            assert_eq!(resp.usage.total_tokens, 23);
        }

        #[tokio::test]
        async fn anthropic_infer_with_messages() {
            use crate::inference::{Message, Role};
            let base_url = start_mock_anthropic().await;
            let provider = AnthropicProvider::new(&base_url, Some("key".into()));

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
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Anthropic mock response");
        }

        #[tokio::test]
        async fn anthropic_list_models() {
            let provider = AnthropicProvider::new("http://unused", None);
            let models = provider.list_models().await.unwrap();
            assert!(models.len() >= 3);
            assert!(models.iter().any(|m| m.id.contains("opus")));
        }

        #[tokio::test]
        async fn anthropic_health_reachable() {
            let base_url = start_mock_anthropic().await;
            let provider = AnthropicProvider::new(&base_url, Some("key".into()));
            // Mock returns 200 on POST /v1/messages, so health should pass
            let healthy = provider.health_check().await.unwrap();
            assert!(healthy);
        }

        #[tokio::test]
        async fn anthropic_health_no_key() {
            let provider = AnthropicProvider::new("http://unused", None);
            let healthy = provider.health_check().await.unwrap();
            assert!(!healthy);
        }

        #[tokio::test]
        async fn anthropic_health_unreachable() {
            let provider = AnthropicProvider::new("http://127.0.0.1:1", Some("key".into()));
            let healthy = provider.health_check().await.unwrap();
            assert!(!healthy);
        }
    }

    /// Test the thin wrapper providers against the mock OAI server.
    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn llamacpp_provider_infer() {
        use crate::provider::llamacpp::LlamaCppProvider;

        let base_url = start_mock_oai_server().await;
        let provider = LlamaCppProvider::new(&base_url);

        let req = InferenceRequest {
            model: "my-gguf".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "llamacpp");
    }

    #[cfg(feature = "lmstudio")]
    #[tokio::test]
    async fn lmstudio_provider_infer() {
        use crate::provider::lmstudio::LmStudioProvider;

        let base_url = start_mock_oai_server().await;
        let provider = LmStudioProvider::new(&base_url);

        let req = InferenceRequest {
            model: "lm-model".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "lmstudio");
    }

    #[cfg(feature = "localai")]
    #[tokio::test]
    async fn localai_provider_infer() {
        use crate::provider::localai::LocalAiProvider;

        let base_url = start_mock_oai_server().await;
        let provider = LocalAiProvider::new(&base_url);

        let req = InferenceRequest {
            model: "local-model".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "localai");
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_provider_infer() {
        use crate::provider::synapse::SynapseProvider;

        let base_url = start_mock_oai_server().await;
        let provider = SynapseProvider::new(&base_url);

        let req = InferenceRequest {
            model: "synapse-model".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "synapse");
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn openai_remote_provider_infer() {
        use crate::provider::openai_remote::OpenAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "gpt-4".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "openai");
    }

    #[cfg(feature = "deepseek")]
    #[tokio::test]
    async fn deepseek_provider_infer() {
        use crate::provider::deepseek::DeepSeekProvider;
        let base_url = start_mock_oai_server().await;
        let provider = DeepSeekProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "deepseek-chat".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.provider, "deepseek");
    }

    #[cfg(feature = "mistral")]
    #[tokio::test]
    async fn mistral_provider_infer() {
        use crate::provider::mistral::MistralProvider;
        let base_url = start_mock_oai_server().await;
        let provider = MistralProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "mistral-large".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.provider, "mistral");
    }

    #[cfg(feature = "groq")]
    #[tokio::test]
    async fn groq_provider_infer() {
        use crate::provider::groq::GroqProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GroqProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "llama3".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.provider, "groq");
    }

    #[cfg(feature = "openrouter")]
    #[tokio::test]
    async fn openrouter_provider_infer() {
        use crate::provider::openrouter::OpenRouterProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenRouterProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "meta/llama3".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.provider, "openrouter");
    }

    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn grok_provider_infer() {
        use crate::provider::grok::GrokProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GrokProvider::new(&base_url, None);
        let req = InferenceRequest {
            model: "grok-2".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.provider, "grok");
    }

    /// Test list_models and health_check through thin wrappers.
    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn llamacpp_list_models_and_health() {
        use crate::provider::llamacpp::LlamaCppProvider;

        let base_url = start_mock_oai_server().await;
        let provider = LlamaCppProvider::new(&base_url);

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);

        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn openai_list_models_and_health() {
        use crate::provider::openai_remote::OpenAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiProvider::new(&base_url, None);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "groq")]
    #[tokio::test]
    async fn groq_list_models_and_health() {
        use crate::provider::groq::GroqProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GroqProvider::new(&base_url, None);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "deepseek")]
    #[tokio::test]
    async fn deepseek_list_models_and_health() {
        use crate::provider::deepseek::DeepSeekProvider;
        let base_url = start_mock_oai_server().await;
        let provider = DeepSeekProvider::new(&base_url, None);
        assert_eq!(provider.list_models().await.unwrap().len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "mistral")]
    #[tokio::test]
    async fn mistral_health() {
        use crate::provider::mistral::MistralProvider;
        let base_url = start_mock_oai_server().await;
        let provider = MistralProvider::new(&base_url, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "openrouter")]
    #[tokio::test]
    async fn openrouter_health() {
        use crate::provider::openrouter::OpenRouterProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenRouterProvider::new(&base_url, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn grok_health() {
        use crate::provider::grok::GrokProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GrokProvider::new(&base_url, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "lmstudio")]
    #[tokio::test]
    async fn lmstudio_health() {
        use crate::provider::lmstudio::LmStudioProvider;
        let base_url = start_mock_oai_server().await;
        let provider = LmStudioProvider::new(&base_url);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "localai")]
    #[tokio::test]
    async fn localai_health() {
        use crate::provider::localai::LocalAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = LocalAiProvider::new(&base_url);
        assert!(provider.health_check().await.unwrap());
    }
}

// ---------------------------------------------------------------------------
// Server-level integration tests (AppState wiring)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Live provider tests (require running backends — run with `cargo test -- --ignored`)
// ---------------------------------------------------------------------------

mod live {
    use crate::inference::InferenceRequest;
    use crate::provider::LlmProvider;

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore] // Requires `ollama serve` running on localhost:11434
    async fn ollama_live_health() {
        use crate::provider::ollama::OllamaProvider;
        let provider = OllamaProvider::new("http://127.0.0.1:11434");
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn ollama_live_list_models() {
        use crate::provider::ollama::OllamaProvider;
        let provider = OllamaProvider::new("http://127.0.0.1:11434");
        let models = provider.list_models().await.unwrap();
        assert!(
            !models.is_empty(),
            "Ollama should have at least one model pulled"
        );
        for m in &models {
            println!("  model: {} (size: {:?})", m.id, m.parameters);
        }
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn ollama_live_infer() {
        use crate::provider::ollama::OllamaProvider;
        let provider = OllamaProvider::new("http://127.0.0.1:11434");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());

        let req = InferenceRequest {
            model: models[0].id.clone(),
            prompt: "Say hello in one word.".into(),
            max_tokens: Some(10),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        println!("  response: {}", resp.text);
        println!(
            "  tokens: prompt={}, completion={}, total={}",
            resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens
        );
        println!("  latency: {}ms", resp.latency_ms);

        assert!(!resp.text.is_empty());
        assert_eq!(resp.provider, "ollama");
        assert!(resp.usage.total_tokens > 0);
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn ollama_live_stream() {
        use crate::provider::ollama::OllamaProvider;
        let provider = OllamaProvider::new("http://127.0.0.1:11434");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());

        let req = InferenceRequest {
            model: models[0].id.clone(),
            prompt: "Count to 3.".into(),
            max_tokens: Some(20),
            stream: true,
            ..Default::default()
        };
        let mut rx = provider.infer_stream(req).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(result) = rx.recv().await {
            let token = result.unwrap();
            tokens.push(token);
        }
        assert!(!tokens.is_empty(), "Should receive at least one token");
        let full = tokens.join("");
        println!("  streamed: {full}");
        assert!(!full.is_empty());
    }
}

mod server_wiring {
    use std::sync::Arc;

    use crate::budget::TokenBudget;
    use crate::cache::{CacheConfig, ResponseCache};
    use crate::provider::{ProviderRegistry, ProviderType};
    use crate::router::{self as hoosh_router, ProviderRoute, RoutingStrategy};
    use crate::server::AppState;

    fn make_state(routes: Vec<ProviderRoute>) -> Arc<AppState> {
        let mut providers = ProviderRegistry::new();
        for route in &routes {
            if route.enabled {
                providers.register_from_route(route);
            }
        }
        Arc::new(AppState {
            router: hoosh_router::Router::new(routes, RoutingStrategy::Priority),
            cache: ResponseCache::new(CacheConfig::default()),
            budget: std::sync::Mutex::new(TokenBudget::new()),
            providers,
            #[cfg(feature = "whisper")]
            whisper: None,
            #[cfg(feature = "piper")]
            tts: None,
        })
    }

    #[test]
    fn app_state_empty_routes() {
        let state = make_state(vec![]);
        assert_eq!(state.router.routes().len(), 0);
        assert!(state.providers.is_empty());
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn app_state_registers_ollama() {
        let state = make_state(vec![ProviderRoute {
            provider: ProviderType::Ollama,
            priority: 1,
            model_patterns: vec!["llama*".into()],
            enabled: true,
            base_url: "http://localhost:11434".into(),
            api_key: None,
            max_tokens_limit: None,
        }]);
        assert_eq!(state.providers.len(), 1);
        assert!(
            state
                .providers
                .get(ProviderType::Ollama, "http://localhost:11434")
                .is_some()
        );
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn app_state_disabled_route_not_registered() {
        let state = make_state(vec![ProviderRoute {
            provider: ProviderType::Ollama,
            priority: 1,
            model_patterns: vec![],
            enabled: false,
            base_url: "http://localhost:11434".into(),
            api_key: None,
            max_tokens_limit: None,
        }]);
        assert!(state.providers.is_empty());
        // But the route is still in the router
        assert_eq!(state.router.routes().len(), 1);
    }

    #[cfg(all(feature = "ollama", feature = "llamacpp", feature = "lmstudio"))]
    #[test]
    fn app_state_multiple_providers() {
        let state = make_state(vec![
            ProviderRoute {
                provider: ProviderType::Ollama,
                priority: 1,
                model_patterns: vec!["llama*".into()],
                enabled: true,
                base_url: "http://localhost:11434".into(),
                api_key: None,
                max_tokens_limit: None,
            },
            ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 2,
                model_patterns: vec!["gguf-*".into()],
                enabled: true,
                base_url: "http://localhost:8080".into(),
                api_key: None,
                max_tokens_limit: None,
            },
            ProviderRoute {
                provider: ProviderType::LmStudio,
                priority: 3,
                model_patterns: vec![],
                enabled: true,
                base_url: "http://localhost:1234".into(),
                api_key: None,
                max_tokens_limit: None,
            },
        ]);
        assert_eq!(state.providers.len(), 3);
        assert_eq!(state.router.routes().len(), 3);
    }

    #[test]
    fn app_state_route_selection_still_works() {
        let state = make_state(vec![
            ProviderRoute {
                provider: ProviderType::Ollama,
                priority: 1,
                model_patterns: vec!["llama*".into()],
                enabled: true,
                base_url: "http://localhost:11434".into(),
                api_key: None,
                max_tokens_limit: None,
            },
            ProviderRoute {
                provider: ProviderType::OpenAi,
                priority: 2,
                model_patterns: vec!["gpt-*".into()],
                enabled: true,
                base_url: "https://api.openai.com".into(),
                api_key: None,
                max_tokens_limit: None,
            },
        ]);

        let route = state.router.select("llama3").unwrap();
        assert_eq!(route.provider, ProviderType::Ollama);

        let route = state.router.select("gpt-4o").unwrap();
        assert_eq!(route.provider, ProviderType::OpenAi);

        // OpenAI is now registered as a remote provider
        #[cfg(feature = "openai")]
        assert!(
            state
                .providers
                .get(ProviderType::OpenAi, "https://api.openai.com")
                .is_some()
        );
        #[cfg(not(feature = "openai"))]
        assert!(
            state
                .providers
                .get(ProviderType::OpenAi, "https://api.openai.com")
                .is_none()
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end server tests (full HTTP stack with mock backend)
// ---------------------------------------------------------------------------

mod e2e {
    use axum::{
        Json, Router,
        routing::{get, post},
    };
    use serde_json::json;
    use tokio::net::TcpListener;

    use crate::budget::TokenPool;
    use crate::cache::CacheConfig;
    use crate::client::HooshClient;
    use crate::provider::ProviderType;
    use crate::router::{ProviderRoute, RoutingStrategy};
    use crate::server::ServerConfig;

    /// Start a mock OpenAI-compatible backend and return its URL.
    async fn start_mock_backend() -> String {
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat))
            .route("/v1/models", get(mock_models));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    async fn mock_chat(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
        let model = body["model"].as_str().unwrap_or("mock-model");
        let stream = body["stream"].as_bool().unwrap_or(false);
        assert!(!stream, "mock does not support streaming");

        Json(json!({
            "id": "chatcmpl-e2e",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "E2E mock response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
    }

    async fn mock_models() -> Json<serde_json::Value> {
        Json(json!({
            "object": "list",
            "data": [
                {"id": "mock-model", "object": "model", "owned_by": "mock"}
            ]
        }))
    }

    /// Start hoosh server pointing at a mock backend, return the hoosh URL.
    async fn start_hoosh(backend_url: &str) -> String {
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: backend_url.to_string(),
                api_key: None,
                max_tokens_limit: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![
                TokenPool::new("default", 100_000),
                TokenPool::new("limited", 50),
            ],
            whisper_model: None,
            tts_model: None,
        };

        let app = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_health() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);

        assert!(client.health().await.unwrap());
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_list_models() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);

        let models = client.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mock-model");
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_infer() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);

        let req = crate::InferenceRequest {
            model: "mock-model".into(),
            prompt: "Hello".into(),
            ..Default::default()
        };
        let resp = client.infer(&req).await.unwrap();
        assert_eq!(resp.text, "E2E mock response");
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_infer_no_matching_model() {
        let backend = start_mock_backend().await;
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec!["llama*".into()],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: vec![TokenPool::new("default", u64::MAX)],
            whisper_model: None,
            tts_model: None,
        };

        let app = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = HooshClient::new(format!("http://127.0.0.1:{}", addr.port()));
        let req = crate::InferenceRequest {
            model: "gpt-4".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let result = client.infer(&req).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_budget_enforcement() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;

        let client = reqwest::Client::new();
        // "limited" pool has 50 tokens — requesting 1024 should fail
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "pool": "limited"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 429);

        // Small request should succeed
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
                "pool": "limited"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_budget_tracks_usage() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();

        // Initial state
        let pools: Vec<serde_json::Value> = client
            .get(format!("{hoosh_url}/v1/tokens/pools"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let default_pool = pools.iter().find(|p| p["name"] == "default").unwrap();
        assert_eq!(default_pool["used"], 0);

        // Inference
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Pool should show actual usage (mock returns total_tokens=15)
        let pools: Vec<serde_json::Value> = client
            .get(format!("{hoosh_url}/v1/tokens/pools"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let default_pool = pools.iter().find(|p| p["name"] == "default").unwrap();
        assert_eq!(default_pool["used"], 15);
        assert_eq!(default_pool["reserved"], 0);
    }

    /// Mock backend that supports SSE streaming.
    async fn start_mock_streaming_backend() -> String {
        async fn mock_stream_chat(Json(body): Json<serde_json::Value>) -> axum::response::Response {
            let model = body["model"].as_str().unwrap_or("mock-model").to_owned();
            let stream = body["stream"].as_bool().unwrap_or(false);

            if !stream {
                return Json(json!({
                    "id": "chatcmpl-e2e",
                    "object": "chat.completion",
                    "model": &model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "non-stream"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
                })).into_response();
            }

            let s = async_stream::stream! {
                let tokens = ["Hello", " ", "world", "!"];
                for token in &tokens {
                    let chunk = json!({
                        "id": "chatcmpl-stream-mock",
                        "object": "chat.completion.chunk",
                        "model": &model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": serde_json::Value::Null}]
                    });
                    yield Ok::<_, std::convert::Infallible>(
                        axum::response::sse::Event::default().data(chunk.to_string())
                    );
                }
                let done_chunk = json!({
                    "id": "chatcmpl-stream-mock",
                    "object": "chat.completion.chunk",
                    "model": &model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                });
                yield Ok(axum::response::sse::Event::default().data(done_chunk.to_string()));
                yield Ok(axum::response::sse::Event::default().data("[DONE]"));
            };

            axum::response::sse::Sse::new(s).into_response()
        }

        use axum::response::IntoResponse;
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_stream_chat))
            .route("/v1/models", get(mock_models));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_streaming() {
        let backend = start_mock_streaming_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);

        let req = crate::InferenceRequest {
            model: "mock-model".into(),
            prompt: "Hello".into(),
            stream: true,
            ..Default::default()
        };
        let mut rx = client.infer_stream(&req).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(result) = rx.recv().await {
            tokens.push(result.unwrap());
        }
        assert!(!tokens.is_empty(), "should receive tokens");
        let full = tokens.join("");
        assert_eq!(full, "Hello world!");
    }

    // --- Server validation tests ---

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_validation_empty_model() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_validation_empty_messages() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": []
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_validation_bad_temperature() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5.0
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_validation_bad_top_p() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": -0.5
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_validation_nonexistent_pool() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "pool": "does-not-exist"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_no_provider_for_model() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);
        // wildcard route matches everything, so use a server with restricted patterns
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec!["only-this*".into()],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: vec![TokenPool::new("default", 10_000_000)],
            whisper_model: None,
            tts_model: None,
        };
        let app = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        let http = reqwest::Client::new();
        let resp = http
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "unmatched-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }
}
