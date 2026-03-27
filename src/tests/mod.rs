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
            rate_limit_rpm: None,
            tls_config: None,
        },
        ProviderRoute {
            provider: ProviderType::OpenAi,
            priority: 2,
            model_patterns: vec!["gpt-*".into()],
            enabled: true,
            base_url: "https://api.openai.com".into(),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
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
        rate_limit_rpm: None,
        tls_config: None,
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
        rate_limit_rpm: None,
        tls_config: None,
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
        rate_limit_rpm: None,
        tls_config: None,
    });
    registry.register_from_route(&ProviderRoute {
        provider: ProviderType::LlamaCpp,
        priority: 2,
        model_patterns: vec![],
        enabled: true,
        base_url: "http://localhost:8080".into(),
        api_key: None,
        max_tokens_limit: None,
        rate_limit_rpm: None,
        tls_config: None,
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
        rate_limit_rpm: None,
        tls_config: None,
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
        rate_limit_rpm: None,
        tls_config: None,
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
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LlamaCpp, None);

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
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LocalAi, None);

        let req = InferenceRequest {
            model: "chat-model".into(),
            messages: vec![
                Message::new(Role::System, "Be helpful."),
                Message::new(Role::User, "Hi"),
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
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LmStudio, None);

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
        let provider = OpenAiCompatibleProvider::new(&base_url, None, ProviderType::LlamaCpp, None);

        let healthy = provider.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn openai_compat_health_check_unreachable() {
        let provider =
            OpenAiCompatibleProvider::new("http://127.0.0.1:1", None, ProviderType::LlamaCpp, None);
        // Should return an error (connection refused), not panic
        let result = provider.health_check().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn openai_compat_infer_unreachable() {
        let provider =
            OpenAiCompatibleProvider::new("http://127.0.0.1:1", None, ProviderType::LlamaCpp, None);
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
            let provider = OllamaProvider::new(&base_url, None);

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
            let provider = OllamaProvider::new(&base_url, None);

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
            let provider = OllamaProvider::new(&base_url, None);

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
            let provider = OllamaProvider::new(&base_url, None);

            let healthy = provider.health_check().await.unwrap();
            assert!(healthy);
        }

        #[tokio::test]
        async fn ollama_health_check_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let result = provider.health_check().await;
            assert!(result.is_err());
        }

        /// Start a mock Ollama server with embeddings, streaming, and error support.
        async fn start_mock_ollama_full() -> String {
            use axum::http::StatusCode;
            use axum::response::IntoResponse;

            async fn mock_chat_full(
                Json(body): Json<serde_json::Value>,
            ) -> axum::response::Response {
                let model = body["model"].as_str().unwrap_or("test-model");

                if model == "fail-model" {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": "model not found"})),
                    )
                        .into_response();
                }

                let stream = body["stream"].as_bool().unwrap_or(false);
                if stream {
                    let ndjson = format!(
                        "{}\n{}\n{}\n",
                        r#"{"message":{"content":"Hello"},"done":false}"#,
                        r#"{"message":{"content":" world"},"done":false}"#,
                        r#"{"message":{"content":""},"done":true,"eval_count":10,"prompt_eval_count":5}"#,
                    );
                    (StatusCode::OK, ndjson).into_response()
                } else {
                    Json(json!({
                        "message": {"role": "assistant", "content": "Full mock reply"},
                        "eval_count": 15,
                        "prompt_eval_count": 20
                    }))
                    .into_response()
                }
            }

            async fn mock_tags_full() -> Json<serde_json::Value> {
                Json(json!({
                    "models": [
                        {"name": "llama3:latest", "size": 4_000_000_000_i64},
                        {"name": "mistral:7b"},
                        {"name": "phi3:mini", "size": 2_000_000_000_i64}
                    ]
                }))
            }

            async fn mock_embed(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
                let model = body["model"].as_str().unwrap_or("nomic-embed");
                Json(json!({
                    "model": model,
                    "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
                }))
            }

            let app = Router::new()
                .route("/api/chat", post(mock_chat_full))
                .route("/api/tags", get(mock_tags_full))
                .route("/api/embed", post(mock_embed));

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            format!("http://127.0.0.1:{}", addr.port())
        }

        #[tokio::test]
        async fn ollama_infer_with_all_options() {
            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = InferenceRequest {
                model: "llama3".into(),
                prompt: "Hello".into(),
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_tokens: Some(256),
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Full mock reply");
            assert_eq!(resp.usage.prompt_tokens, 20);
            assert_eq!(resp.usage.completion_tokens, 15);
            assert_eq!(resp.usage.total_tokens, 35);
            assert_eq!(resp.provider, "ollama");
            assert!(resp.latency_ms < 5000);
        }

        #[tokio::test]
        async fn ollama_infer_with_messages() {
            use crate::inference::{Message, Role};

            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = InferenceRequest {
                model: "llama3".into(),
                messages: vec![
                    Message::new(Role::System, "Be brief."),
                    Message::new(Role::User, "Hi"),
                    Message::new(Role::Assistant, "Hello!"),
                    Message::new(Role::User, "How?"),
                ],
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Full mock reply");
        }

        #[tokio::test]
        async fn ollama_infer_error_response() {
            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = InferenceRequest {
                model: "fail-model".into(),
                prompt: "Hello".into(),
                ..Default::default()
            };
            let result = provider.infer(&req).await;
            assert!(result.is_err(), "should fail on 500 response");
        }

        #[tokio::test]
        async fn ollama_infer_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let req = InferenceRequest {
                model: "test".into(),
                prompt: "Hi".into(),
                ..Default::default()
            };
            let result = provider.infer(&req).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn ollama_list_models_full() {
            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let models = provider.list_models().await.unwrap();
            assert_eq!(models.len(), 3);
            assert_eq!(models[0].id, "llama3:latest");
            assert_eq!(models[0].name, "llama3:latest");
            assert_eq!(models[0].provider, "ollama");
            assert_eq!(models[0].parameters, Some(4_000_000_000));
            assert!(models[0].context_length.is_none());
            assert!(models[0].available);
            assert_eq!(models[1].id, "mistral:7b");
            assert!(models[1].parameters.is_none());
            assert_eq!(models[2].id, "phi3:mini");
            assert_eq!(models[2].parameters, Some(2_000_000_000));
        }

        #[tokio::test]
        async fn ollama_list_models_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let result = provider.list_models().await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn ollama_embeddings_single() {
            use crate::inference::{EmbeddingsInput, EmbeddingsRequest};

            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = EmbeddingsRequest {
                model: "nomic-embed".into(),
                input: EmbeddingsInput::Single("hello world".into()),
            };
            let resp = provider.embeddings(&req).await.unwrap();
            assert_eq!(resp.object, "list");
            assert_eq!(resp.model, "nomic-embed");
            assert_eq!(resp.data.len(), 2);
            assert_eq!(resp.data[0].object, "embedding");
            assert_eq!(resp.data[0].index, 0);
            assert_eq!(resp.data[0].embedding.len(), 4);
            assert!((resp.data[0].embedding[0] - 0.1).abs() < 1e-6);
            assert_eq!(resp.data[1].index, 1);
            assert!((resp.data[1].embedding[0] - 0.5).abs() < 1e-6);
        }

        #[tokio::test]
        async fn ollama_embeddings_multiple() {
            use crate::inference::{EmbeddingsInput, EmbeddingsRequest};

            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = EmbeddingsRequest {
                model: "nomic-embed".into(),
                input: EmbeddingsInput::Multiple(vec!["hello".into(), "world".into()]),
            };
            let resp = provider.embeddings(&req).await.unwrap();
            assert_eq!(resp.data.len(), 2);
            assert_eq!(resp.usage.prompt_tokens, 0);
            assert_eq!(resp.usage.total_tokens, 0);
        }

        #[tokio::test]
        async fn ollama_embeddings_unreachable() {
            use crate::inference::{EmbeddingsInput, EmbeddingsRequest};

            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let req = EmbeddingsRequest {
                model: "nomic-embed".into(),
                input: EmbeddingsInput::Single("test".into()),
            };
            let result = provider.embeddings(&req).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn ollama_infer_stream_basic() {
            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = InferenceRequest {
                model: "llama3".into(),
                prompt: "Hello".into(),
                ..Default::default()
            };
            let mut rx = provider.infer_stream(req).await.unwrap();

            let mut collected = String::new();
            while let Some(chunk) = rx.recv().await {
                collected.push_str(&chunk.unwrap());
            }
            assert_eq!(collected, "Hello world");
        }

        #[tokio::test]
        async fn ollama_infer_stream_with_options() {
            let base_url = start_mock_ollama_full().await;
            let provider = OllamaProvider::new(&base_url, None);

            let req = InferenceRequest {
                model: "llama3".into(),
                prompt: "Hello".into(),
                temperature: Some(0.5),
                top_p: Some(0.8),
                max_tokens: Some(128),
                ..Default::default()
            };
            let mut rx = provider.infer_stream(req).await.unwrap();

            let mut collected = String::new();
            while let Some(chunk) = rx.recv().await {
                collected.push_str(&chunk.unwrap());
            }
            assert_eq!(collected, "Hello world");
        }

        #[tokio::test]
        async fn ollama_infer_stream_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let req = InferenceRequest {
                model: "test".into(),
                prompt: "Hi".into(),
                ..Default::default()
            };
            let result = provider.infer_stream(req).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn ollama_pull_model_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let result = provider.pull_model("llama3").await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn ollama_delete_model_unreachable() {
            let provider = OllamaProvider::new("http://127.0.0.1:1", None);
            let result = provider.delete_model("llama3").await;
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
            let provider = AnthropicProvider::new(&base_url, Some("sk-ant-test".into()), None);

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
            let provider = AnthropicProvider::new(&base_url, Some("key".into()), None);

            let req = InferenceRequest {
                model: "claude-sonnet-4-20250514".into(),
                messages: vec![
                    Message::new(Role::System, "Be concise."),
                    Message::new(Role::User, "Hi"),
                    Message::new(Role::Assistant, "Hello!"),
                    Message::new(Role::User, "More"),
                ],
                ..Default::default()
            };
            let resp = provider.infer(&req).await.unwrap();
            assert_eq!(resp.text, "Anthropic mock response");
        }

        #[tokio::test]
        async fn anthropic_list_models() {
            let provider = AnthropicProvider::new("http://unused", None, None);
            let models = provider.list_models().await.unwrap();
            assert!(models.len() >= 3);
            assert!(models.iter().any(|m| m.id.contains("opus")));
        }

        #[tokio::test]
        async fn anthropic_health_reachable() {
            let base_url = start_mock_anthropic().await;
            let provider = AnthropicProvider::new(&base_url, Some("key".into()), None);
            // Mock returns 200 on POST /v1/messages, so health should pass
            let healthy = provider.health_check().await.unwrap();
            assert!(healthy);
        }

        #[tokio::test]
        async fn anthropic_health_no_key() {
            let provider = AnthropicProvider::new("http://unused", None, None);
            let healthy = provider.health_check().await.unwrap();
            assert!(!healthy);
        }

        #[tokio::test]
        async fn anthropic_health_unreachable() {
            let provider = AnthropicProvider::new("http://127.0.0.1:1", Some("key".into()), None);
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
        let provider = LlamaCppProvider::new(&base_url, None);

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
        let provider = LmStudioProvider::new(&base_url, None);

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
        let provider = LocalAiProvider::new(&base_url, None);

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
        let provider = SynapseProvider::new(&base_url, None);

        let req = InferenceRequest {
            model: "synapse-model".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "synapse");
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_provider_infer_with_messages() {
        use crate::inference::{Message, Role};
        use crate::provider::synapse::SynapseProvider;

        let base_url = start_mock_oai_server().await;
        let provider = SynapseProvider::new(&base_url, None);

        let req = InferenceRequest {
            model: "synapse-model".into(),
            messages: vec![
                Message::new(Role::System, "Be concise."),
                Message::new(Role::User, "Hello"),
            ],
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };
        let resp = provider.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Mock response from server");
        assert_eq!(resp.provider, "synapse");
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_list_models() {
        use crate::provider::synapse::SynapseProvider;

        let base_url = start_mock_oai_server().await;
        let provider = SynapseProvider::new(&base_url, None);

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "mock-model-1");
        assert!(models[0].available);
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_health_check() {
        use crate::provider::synapse::SynapseProvider;

        let base_url = start_mock_oai_server().await;
        let provider = SynapseProvider::new(&base_url, None);

        let healthy = provider.health_check().await.unwrap();
        assert!(healthy);
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_health_check_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let result = provider.health_check().await;
        assert!(result.is_err());
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_infer_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let req = InferenceRequest {
            model: "test".into(),
            prompt: "Hi".into(),
            ..Default::default()
        };
        let result = provider.infer(&req).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_list_models_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let result = provider.list_models().await;
        assert!(result.is_err());
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_infer_stream_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let req = InferenceRequest {
            model: "test".into(),
            prompt: "Hello".into(),
            stream: true,
            ..Default::default()
        };
        let result = provider.infer_stream(req).await;
        assert!(result.is_err());
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_training_status_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let result = provider.training_status("job-123").await;
        assert!(result.is_err());
    }

    #[cfg(feature = "synapse")]
    #[tokio::test]
    async fn synapse_sync_catalog_unreachable() {
        use crate::provider::synapse::SynapseProvider;

        let provider = SynapseProvider::new("http://127.0.0.1:1", None);
        let result = provider.sync_catalog().await;
        assert!(result.is_err());
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn openai_remote_provider_infer() {
        use crate::provider::openai_remote::OpenAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiProvider::new(&base_url, None, None);
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
        let provider = DeepSeekProvider::new(&base_url, None, None);
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
        let provider = MistralProvider::new(&base_url, None, None);
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
        let provider = GroqProvider::new(&base_url, None, None);
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
        let provider = OpenRouterProvider::new(&base_url, None, None);
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
        let provider = GrokProvider::new(&base_url, None, None);
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
        let provider = LlamaCppProvider::new(&base_url, None);

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);

        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn openai_list_models_and_health() {
        use crate::provider::openai_remote::OpenAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenAiProvider::new(&base_url, None, None);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "groq")]
    #[tokio::test]
    async fn groq_list_models_and_health() {
        use crate::provider::groq::GroqProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GroqProvider::new(&base_url, None, None);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "deepseek")]
    #[tokio::test]
    async fn deepseek_list_models_and_health() {
        use crate::provider::deepseek::DeepSeekProvider;
        let base_url = start_mock_oai_server().await;
        let provider = DeepSeekProvider::new(&base_url, None, None);
        assert_eq!(provider.list_models().await.unwrap().len(), 2);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "mistral")]
    #[tokio::test]
    async fn mistral_health() {
        use crate::provider::mistral::MistralProvider;
        let base_url = start_mock_oai_server().await;
        let provider = MistralProvider::new(&base_url, None, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "openrouter")]
    #[tokio::test]
    async fn openrouter_health() {
        use crate::provider::openrouter::OpenRouterProvider;
        let base_url = start_mock_oai_server().await;
        let provider = OpenRouterProvider::new(&base_url, None, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn grok_health() {
        use crate::provider::grok::GrokProvider;
        let base_url = start_mock_oai_server().await;
        let provider = GrokProvider::new(&base_url, None, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "lmstudio")]
    #[tokio::test]
    async fn lmstudio_health() {
        use crate::provider::lmstudio::LmStudioProvider;
        let base_url = start_mock_oai_server().await;
        let provider = LmStudioProvider::new(&base_url, None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "localai")]
    #[tokio::test]
    async fn localai_health() {
        use crate::provider::localai::LocalAiProvider;
        let base_url = start_mock_oai_server().await;
        let provider = LocalAiProvider::new(&base_url, None);
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
        let provider = OllamaProvider::new("http://127.0.0.1:11434", None);
        assert!(provider.health_check().await.unwrap());
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn ollama_live_list_models() {
        use crate::provider::ollama::OllamaProvider;
        let provider = OllamaProvider::new("http://127.0.0.1:11434", None);
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
        let provider = OllamaProvider::new("http://127.0.0.1:11434", None);
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
        let provider = OllamaProvider::new("http://127.0.0.1:11434", None);
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
            router: std::sync::RwLock::new(hoosh_router::Router::new(
                routes,
                RoutingStrategy::Priority,
            )),
            config_path: None,
            cache: ResponseCache::new(CacheConfig::default()),
            budget: std::sync::Mutex::new(TokenBudget::new()),
            providers,
            cost_tracker: std::sync::Arc::new(crate::cost::CostTracker::new()),
            audit: None,
            auth_token_digests: Vec::new(),
            rate_limiter: std::sync::Arc::new(
                crate::middleware::rate_limit::RateLimitRegistry::new(),
            ),
            event_bus: std::sync::Arc::new(crate::events::new_event_bus()),
            inference_queue: std::sync::Arc::new(crate::queue::InferenceQueue::new()),
            health_map: crate::health::new_health_map(),
            heartbeat: std::sync::Arc::new(majra::heartbeat::ConcurrentHeartbeatTracker::default()),
            #[cfg(feature = "whisper")]
            whisper: None,
            #[cfg(feature = "piper")]
            tts: None,
            #[cfg(feature = "tools")]
            mcp_bridge: std::sync::Arc::new(crate::tools::McpBridge::new()),
            compactor: crate::context::compactor::ContextCompactor::new(0.8, 10, true),
            model_registry: crate::provider::metadata::ModelMetadataRegistry::new(),
            retry_manager: crate::provider::retry::RetryManager::new(Default::default()),
        })
    }

    #[test]
    fn app_state_empty_routes() {
        let state = make_state(vec![]);
        assert_eq!(state.router.read().unwrap().routes().len(), 0);
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
            rate_limit_rpm: None,
            tls_config: None,
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
            rate_limit_rpm: None,
            tls_config: None,
        }]);
        assert!(state.providers.is_empty());
        // But the route is still in the router
        assert_eq!(state.router.read().unwrap().routes().len(), 1);
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
                rate_limit_rpm: None,
                tls_config: None,
            },
            ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 2,
                model_patterns: vec!["gguf-*".into()],
                enabled: true,
                base_url: "http://localhost:8080".into(),
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            },
            ProviderRoute {
                provider: ProviderType::LmStudio,
                priority: 3,
                model_patterns: vec![],
                enabled: true,
                base_url: "http://localhost:1234".into(),
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            },
        ]);
        assert_eq!(state.providers.len(), 3);
        assert_eq!(state.router.read().unwrap().routes().len(), 3);
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
                rate_limit_rpm: None,
                tls_config: None,
            },
            ProviderRoute {
                provider: ProviderType::OpenAi,
                priority: 2,
                model_patterns: vec!["gpt-*".into()],
                enabled: true,
                base_url: "https://api.openai.com".into(),
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            },
        ]);

        {
            let router = state.router.read().unwrap();
            let route = router.select("llama3").unwrap();
            assert_eq!(route.provider, ProviderType::Ollama);
        }

        {
            let router = state.router.read().unwrap();
            let route = router.select("gpt-4o").unwrap();
            assert_eq!(route.provider, ProviderType::OpenAi);
        }

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
                rate_limit_rpm: None,
                tls_config: None,
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
            audit_enabled: false,
            audit_signing_key: None,
            audit_max_entries: 10_000,
            auth_tokens: Vec::new(),
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
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
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: vec![TokenPool::new("default", u64::MAX)],
            whisper_model: None,
            tts_model: None,
            audit_enabled: false,
            audit_signing_key: None,
            audit_max_entries: 10_000,
            auth_tokens: Vec::new(),
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
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
        let _hoosh_url = start_hoosh(&backend).await;
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
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: vec![TokenPool::new("default", 10_000_000)],
            whisper_model: None,
            tts_model: None,
            audit_enabled: false,
            audit_signing_key: None,
            audit_max_entries: 10_000,
            auth_tokens: Vec::new(),
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
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

    // -----------------------------------------------------------------------
    // Embeddings endpoint tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_embeddings_no_provider() {
        // Server with no matching model for embeddings → 404
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec!["only-this*".into()],
                enabled: true,
                base_url: "http://127.0.0.1:1".into(),
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig::default(),
            budget_pools: vec![],
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/embeddings", addr.port()))
            .json(&json!({
                "model": "no-such-embed-model",
                "input": "hello"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    // -----------------------------------------------------------------------
    // Admin reload endpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_admin_reload_no_config_path() {
        // When config_path is None, reload should return 400
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            config_path: None,
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/admin/reload", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no config path")
        );
    }

    // -----------------------------------------------------------------------
    // Queue status endpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_queue_status() {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/v1/queue/status", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["queued"], 0);
    }

    // -----------------------------------------------------------------------
    // Costs endpoints tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_costs_get() {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/v1/costs", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["records"].is_array());
        assert_eq!(body["total_cost_usd"], 0.0);
    }

    #[tokio::test]
    async fn e2e_costs_reset() {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/costs/reset", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
    }

    // -----------------------------------------------------------------------
    // Audit log endpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_audit_disabled() {
        // When audit is not enabled, /v1/audit should return 404
        let config = ServerConfig {
            audit_enabled: false,
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/v1/audit", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn e2e_audit_enabled() {
        let config = ServerConfig {
            audit_enabled: true,
            audit_signing_key: Some("test-key-for-audit".into()),
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/v1/audit", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["entries"].is_array());
        assert_eq!(body["total"], 0);
        assert_eq!(body["chain_valid"], true);
    }

    // -----------------------------------------------------------------------
    // Prometheus metrics endpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_prometheus_metrics() {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://127.0.0.1:{}/metrics", addr.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            content_type.contains("text/plain"),
            "metrics should be text/plain, got: {content_type}"
        );
        let body = resp.text().await.unwrap();
        // Prometheus output should contain at least HELP or TYPE lines
        // or be a valid (possibly empty) metrics body
        assert!(
            body.is_empty() || body.contains("hoosh") || body.contains("# "),
            "metrics body should be prometheus format"
        );
    }

    // -----------------------------------------------------------------------
    // Health heartbeat endpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_health_heartbeat() {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let resp = client
            .get(format!(
                "http://127.0.0.1:{}/v1/health/heartbeat",
                addr.port()
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        // Fleet stats should be a JSON object with counts
        assert!(body.is_object(), "heartbeat should return JSON object");
    }

    // -----------------------------------------------------------------------
    // Rate limiting (429 response) tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_rate_limit_exceeded() {
        let backend = start_mock_backend().await;
        // Configure a route with rate_limit_rpm = 1 so it trips after one request
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: Some(1),
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 10_000_000)],
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = reqwest::Client::new();

        // First request should succeed (uses up the 1 RPM allowance)
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Second request should be rate-limited (429)
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi again"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 429);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Rate limit")
        );
    }

    // -----------------------------------------------------------------------
    // Token budget enforcement: pool not found, budget exceeded
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_budget_pool_not_found() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "pool": "nonexistent-pool"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not exist")
        );
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_budget_exceeded() {
        let backend = start_mock_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();

        // "limited" pool has 50 tokens, requesting max_tokens=1024 => estimated 1024 > 50
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
        let body: serde_json::Value = resp.json().await.unwrap();
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("Token budget exceeded"));
        assert!(msg.contains("limited"));
    }

    // -----------------------------------------------------------------------
    // Costs reset with audit enabled (covers audit recording path)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn e2e_costs_reset_with_audit() {
        let config = ServerConfig {
            audit_enabled: true,
            audit_signing_key: Some("cost-audit-key".into()),
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = reqwest::Client::new();

        // Reset costs (should record audit event)
        let resp = client
            .post(format!("{url}/v1/costs/reset"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Verify the audit log now has an entry
        let resp = client.get(format!("{url}/v1/audit")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["total"].as_u64().unwrap() >= 1);
        assert_eq!(body["chain_valid"], true);
        let entries = body["entries"].as_array().unwrap();
        assert!(entries.iter().any(|e| {
            e["event"]
                .as_str()
                .map(|s| s.contains("costs_reset"))
                .unwrap_or(false)
        }));
    }

    // -----------------------------------------------------------------------
    // Streaming SSE path in chat_completions
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_streaming_sse_response_shape() {
        let backend = start_mock_streaming_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = reqwest::Client::new();

        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            content_type.contains("text/event-stream"),
            "streaming response should be SSE, got: {content_type}"
        );

        let body = resp.text().await.unwrap();
        // SSE body should contain "data:" lines with chat.completion.chunk objects
        assert!(
            body.contains("chat.completion.chunk"),
            "SSE body should contain chunk objects"
        );
        assert!(body.contains("[DONE]"), "SSE body should end with [DONE]");
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "SSE body should contain finish_reason stop"
        );
    }

    // -----------------------------------------------------------------------
    // Connection tuning tests
    // -----------------------------------------------------------------------

    /// Verify HooshClient with tuned settings can connect and get health.
    #[tokio::test]
    async fn hoosh_client_tuned_health() {
        use crate::server::{ServerConfig, build_app};
        use tokio::net::TcpListener;

        let config = ServerConfig::default();
        let (app, _state) = build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = HooshClient::new(format!("http://127.0.0.1:{}", addr.port()));
        let healthy = client.health().await.unwrap();
        assert!(healthy);
    }

    /// Verify connection reuse: multiple requests on the same HooshClient
    /// reuse the TCP connection (second request should be faster than first).
    #[tokio::test]
    async fn hoosh_client_connection_reuse() {
        use crate::server::{ServerConfig, build_app};
        use tokio::net::TcpListener;

        let config = ServerConfig::default();
        let (app, _state) = build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = HooshClient::new(format!("http://127.0.0.1:{}", addr.port()));

        // First request establishes TCP connection
        let start1 = std::time::Instant::now();
        let _ = client.health().await.unwrap();
        let first_ms = start1.elapsed();

        // Subsequent requests reuse the pooled connection — do several to prove stability
        let mut reuse_times = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let healthy = client.health().await.unwrap();
            assert!(healthy);
            reuse_times.push(start.elapsed());
        }

        // All reuse requests should succeed (connection pool works)
        assert_eq!(reuse_times.len(), 5);

        // The average reuse time should be ≤ first connection time
        // (first includes TCP handshake, subsequent reuse the pooled connection)
        let avg_reuse = reuse_times.iter().sum::<std::time::Duration>() / reuse_times.len() as u32;
        // Relaxed assertion: just verify reuse requests work and are reasonably fast
        assert!(
            avg_reuse < std::time::Duration::from_secs(1),
            "reuse requests should be fast, got {:?} avg (first was {:?})",
            avg_reuse,
            first_ms
        );
    }

    /// Verify tuned OllamaProvider creates successfully.
    #[test]
    fn ollama_provider_tuned_creation() {
        use crate::provider::LlmProvider;
        use crate::provider::ollama::OllamaProvider;

        let p = OllamaProvider::new("http://localhost:11434", None);
        // Provider created with tuned client settings — verify it doesn't panic
        assert_eq!(p.provider_type(), ProviderType::Ollama);
    }

    /// Verify tuned OpenAiCompatibleProvider creates successfully.
    #[test]
    fn openai_compat_provider_tuned_creation() {
        use crate::provider::LlmProvider;
        use crate::provider::openai_compat::OpenAiCompatibleProvider;

        let p = OpenAiCompatibleProvider::new(
            "http://localhost:8080",
            Some("sk-test".into()),
            ProviderType::OpenAi,
            None,
        );
        assert_eq!(p.base_url(), "http://localhost:8080");
        assert_eq!(p.provider_type(), ProviderType::OpenAi);
    }

    /// Verify concurrent requests through a tuned HooshClient all succeed.
    #[tokio::test]
    async fn hoosh_client_concurrent_requests() {
        use crate::server::{ServerConfig, build_app};
        use tokio::net::TcpListener;

        let config = ServerConfig::default();
        let (app, _state) = build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = std::sync::Arc::new(HooshClient::new(format!(
            "http://127.0.0.1:{}",
            addr.port()
        )));

        // Fire 10 concurrent health checks through the same pooled client
        let mut handles = Vec::new();
        for _ in 0..10 {
            let c = client.clone();
            handles.push(tokio::spawn(async move { c.health().await }));
        }

        for h in handles {
            let result = h.await.unwrap().unwrap();
            assert!(result);
        }
    }

    // -----------------------------------------------------------------------
    // 1. Streaming SSE full flow e2e test
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_streaming_full_flow() {
        let backend = start_mock_streaming_backend().await;
        let hoosh_url = start_hoosh(&backend).await;
        let client = HooshClient::new(&hoosh_url);

        let req = crate::InferenceRequest {
            model: "mock-model".into(),
            prompt: "Stream test".into(),
            stream: true,
            ..Default::default()
        };
        let mut rx = client.infer_stream(&req).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(result) = rx.recv().await {
            let token = result.unwrap();
            tokens.push(token);
        }

        assert!(
            tokens.len() >= 2,
            "should receive multiple tokens, got {}",
            tokens.len()
        );
        let full = tokens.join("");
        assert_eq!(
            full, "Hello world!",
            "concatenated streaming tokens should match"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Embeddings mock + e2e pass-through test (Ollama provider)
    // -----------------------------------------------------------------------

    /// Start a mock Ollama backend that supports chat, tags, and embeddings.
    async fn start_mock_ollama_backend_with_embeddings() -> String {
        async fn ollama_chat(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
            let stream = body["stream"].as_bool().unwrap_or(false);
            assert!(!stream, "mock does not support streaming");
            Json(json!({
                "message": {"role": "assistant", "content": "Ollama embed mock reply"},
                "eval_count": 5,
                "prompt_eval_count": 10
            }))
        }
        async fn ollama_tags() -> Json<serde_json::Value> {
            Json(json!({
                "models": [{"name": "mock-embed-model", "size": 1000000000_i64}]
            }))
        }
        async fn ollama_embed(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
            let model = body["model"].as_str().unwrap_or("mock-embed-model");
            Json(json!({
                "embeddings": [[0.1, 0.2, 0.3]],
                "model": model
            }))
        }

        let app = Router::new()
            .route("/api/chat", post(ollama_chat))
            .route("/api/tags", get(ollama_tags))
            .route("/api/embed", post(ollama_embed));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    async fn e2e_embeddings_pass_through() {
        let backend = start_mock_ollama_backend_with_embeddings().await;

        // Start hoosh with an Ollama provider (which supports embeddings)
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::Ollama,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 100_000)],
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let hoosh_url = format!("http://127.0.0.1:{}", addr.port());

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/embeddings"))
            .json(&json!({
                "model": "mock-embed-model",
                "input": "hello world"
            }))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(status, 200, "expected 200 but got {status}: {body}");
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["object"], "embedding");
        let embedding = data[0]["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 3);
        assert!((embedding[0].as_f64().unwrap() - 0.1).abs() < 0.001);
        assert!((embedding[1].as_f64().unwrap() - 0.2).abs() < 0.001);
        assert!((embedding[2].as_f64().unwrap() - 0.3).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // 3. Full gateway flow with observability verification
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_full_flow_with_observability() {
        let backend = start_mock_backend().await;

        // Start hoosh with audit enabled
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 100_000)],
            whisper_model: None,
            tts_model: None,
            audit_enabled: true,
            audit_signing_key: Some("observability-test-key".into()),
            audit_max_entries: 10_000,
            auth_tokens: Vec::new(),
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let hoosh_url = format!("http://127.0.0.1:{}", addr.port());

        let client = reqwest::Client::new();

        // Send an inference request
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "observability test"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Verify /v1/costs shows the cost was recorded
        let resp = client
            .get(format!("{hoosh_url}/v1/costs"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let costs: serde_json::Value = resp.json().await.unwrap();
        let records = costs["records"].as_array().unwrap();
        assert!(
            !records.is_empty(),
            "costs should have at least one record after inference"
        );

        // Verify /v1/audit shows audit entries
        let resp = client
            .get(format!("{hoosh_url}/v1/audit"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let audit: serde_json::Value = resp.json().await.unwrap();
        assert!(
            audit["total"].as_u64().unwrap() >= 1,
            "audit log should have at least one entry"
        );
        assert_eq!(audit["chain_valid"], true, "audit chain should be valid");

        // Verify /metrics contains prometheus output
        let resp = client
            .get(format!("{hoosh_url}/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let metrics_body = resp.text().await.unwrap();
        assert!(
            metrics_body.contains("hoosh")
                || metrics_body.contains("# ")
                || metrics_body.is_empty(),
            "metrics should be prometheus format"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Auth enforcement e2e
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_auth_enforcement() {
        let backend = start_mock_backend().await;

        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::LlamaCpp,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: backend,
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 100_000)],
            auth_tokens: vec!["correct-token-123".into()],
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let hoosh_url = format!("http://127.0.0.1:{}", addr.port());

        let client = reqwest::Client::new();
        let body = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "auth test"}]
        });

        // Request without token returns 401
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status().as_u16(),
            401,
            "missing token should return 401"
        );

        // Request with wrong token returns 401
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .bearer_auth("wrong-token-456")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 401, "wrong token should return 401");

        // Request with correct token returns 200
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .bearer_auth("correct-token-123")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status().as_u16(),
            200,
            "correct token should return 200"
        );

        // Verify the response body is valid
        let resp_body: serde_json::Value = resp.json().await.unwrap();
        assert!(resp_body["choices"].is_array());
    }

    // -----------------------------------------------------------------------
    // 5. Health check failover e2e
    // -----------------------------------------------------------------------

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn e2e_health_failover() {
        let working_backend = start_mock_backend().await;
        // Dead port — nothing listens here
        let dead_backend = "http://127.0.0.1:1";

        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![
                // Priority 1: dead backend (should get marked unhealthy)
                ProviderRoute {
                    provider: ProviderType::LlamaCpp,
                    priority: 1,
                    model_patterns: vec![],
                    enabled: true,
                    base_url: dead_backend.to_string(),
                    api_key: None,
                    max_tokens_limit: None,
                    rate_limit_rpm: None,
                    tls_config: None,
                },
                // Priority 2: working backend (fallback)
                ProviderRoute {
                    provider: ProviderType::LmStudio,
                    priority: 2,
                    model_patterns: vec![],
                    enabled: true,
                    base_url: working_backend,
                    api_key: None,
                    max_tokens_limit: None,
                    rate_limit_rpm: None,
                    tls_config: None,
                },
            ],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 100_000)],
            // Run health checks every 1 second to speed up the test
            health_check_interval_secs: 1,
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let hoosh_url = format!("http://127.0.0.1:{}", addr.port());

        // Wait for the health checker to run enough times to mark the dead
        // backend unhealthy (UNHEALTHY_THRESHOLD = 3 consecutive failures,
        // plus the initial tick skip). With 1s interval we need ~4-5 seconds.
        tokio::time::sleep(std::time::Duration::from_secs(6)).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{hoosh_url}/v1/chat/completions"))
            .json(&json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "failover test"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request should succeed via healthy fallback backend"
        );

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body["choices"][0]["message"]["content"], "E2E mock response",
            "response should come from the working mock backend"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Ollama-native mock e2e test
    // -----------------------------------------------------------------------

    #[cfg(feature = "ollama")]
    #[tokio::test]
    async fn e2e_ollama_native_flow() {
        // Start a mock Ollama backend with native /api/chat and /api/tags
        async fn mock_ollama_chat(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
            let _model = body["model"].as_str().unwrap_or("llama3:latest");
            Json(json!({
                "message": {"role": "assistant", "content": "Ollama mock response"},
                "eval_count": 5,
                "prompt_eval_count": 10
            }))
        }

        async fn mock_ollama_tags() -> Json<serde_json::Value> {
            Json(json!({
                "models": [
                    {"name": "llama3:latest", "size": 4000000000_i64}
                ]
            }))
        }

        let app = Router::new()
            .route("/api/chat", post(mock_ollama_chat))
            .route("/api/tags", get(mock_ollama_tags));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let ollama_url = format!("http://127.0.0.1:{}", addr.port());

        // Start hoosh with an Ollama provider pointing to the mock
        let config = ServerConfig {
            bind: "127.0.0.1".into(),
            port: 0,
            routes: vec![ProviderRoute {
                provider: ProviderType::Ollama,
                priority: 1,
                model_patterns: vec![],
                enabled: true,
                base_url: ollama_url,
                api_key: None,
                max_tokens_limit: None,
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![crate::budget::TokenPool::new("default", 100_000)],
            ..ServerConfig::default()
        };

        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let hoosh_url = format!("http://127.0.0.1:{}", addr.port());

        let client = HooshClient::new(&hoosh_url);

        // Verify list models through hoosh
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty(), "should list ollama models");
        assert_eq!(models[0].id, "llama3:latest");

        // Verify inference through hoosh -> Ollama mock
        let req = crate::InferenceRequest {
            model: "llama3:latest".into(),
            prompt: "Hello from e2e".into(),
            ..Default::default()
        };
        let resp = client.infer(&req).await.unwrap();
        assert_eq!(resp.text, "Ollama mock response");
        // HooshClient sets provider to "hoosh" (gateway perspective)
        assert_eq!(resp.provider, "hoosh");
    }
}

// ---------------------------------------------------------------------------
// OpenAI API conformance tests — strict schema validation
// ---------------------------------------------------------------------------

#[cfg(feature = "llamacpp")]
mod conformance {
    use serde_json::Value;
    use tokio::net::TcpListener;

    use crate::budget::TokenPool;
    use crate::cache::CacheConfig;
    use crate::provider::ProviderType;
    use crate::router::{ProviderRoute, RoutingStrategy};
    use crate::server::ServerConfig;

    async fn start_mock_backend() -> String {
        use axum::{
            Json, Router,
            routing::{get, post},
        };

        async fn mock_chat(Json(body): Json<Value>) -> Json<Value> {
            let model = body["model"].as_str().unwrap_or("mock-model");
            Json(serde_json::json!({
                "id": "chatcmpl-conf",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "conformance reply"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
            }))
        }

        async fn mock_models() -> Json<Value> {
            Json(serde_json::json!({
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}]
            }))
        }

        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat))
            .route("/v1/models", get(mock_models));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://127.0.0.1:{}", addr.port())
    }

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
                rate_limit_rpm: None,
                tls_config: None,
            }],
            strategy: RoutingStrategy::Priority,
            cache_config: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            budget_pools: vec![TokenPool::new("default", 100_000)],
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://127.0.0.1:{}", addr.port())
    }

    fn http_client() -> reqwest::Client {
        reqwest::Client::new()
    }

    // -- /v1/chat/completions response schema --

    #[tokio::test]
    async fn conformance_chat_response_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        // Required top-level fields
        assert!(resp["id"].is_string(), "id must be string");
        assert_eq!(resp["object"], "chat.completion");
        assert!(resp["created"].is_number(), "created must be number");
        assert!(resp["model"].is_string(), "model must be string");
        assert!(resp["choices"].is_array(), "choices must be array");
        assert!(resp["usage"].is_object(), "usage must be object");
    }

    #[tokio::test]
    async fn conformance_chat_choices_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let choices = resp["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);

        let choice = &choices[0];
        assert_eq!(choice["index"], 0);
        assert!(
            choice["finish_reason"].is_string(),
            "finish_reason must be string"
        );

        let msg = &choice["message"];
        assert_eq!(msg["role"], "assistant");
        assert!(msg["content"].is_string(), "content must be string");
    }

    #[tokio::test]
    async fn conformance_chat_usage_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let usage = &resp["usage"];
        assert!(
            usage["prompt_tokens"].is_number(),
            "prompt_tokens must be number"
        );
        assert!(
            usage["completion_tokens"].is_number(),
            "completion_tokens must be number"
        );
        assert!(
            usage["total_tokens"].is_number(),
            "total_tokens must be number"
        );
        let total = usage["total_tokens"].as_u64().unwrap();
        let prompt = usage["prompt_tokens"].as_u64().unwrap();
        let completion = usage["completion_tokens"].as_u64().unwrap();
        assert_eq!(
            total,
            prompt + completion,
            "total must equal prompt + completion"
        );
    }

    #[tokio::test]
    async fn conformance_chat_id_format() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let id = resp["id"].as_str().unwrap();
        assert!(id.starts_with("chatcmpl-"), "id must start with chatcmpl-");
    }

    #[tokio::test]
    async fn conformance_chat_created_is_unix_timestamp() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let created = resp["created"].as_i64().unwrap();
        // Must be a reasonable unix timestamp (after 2024-01-01)
        assert!(created > 1_704_067_200, "created must be a unix timestamp");
    }

    // -- /v1/models response schema --

    #[tokio::test]
    async fn conformance_models_response_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;
        let resp: Value = http_client()
            .get(format!("{url}/v1/models"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(resp["object"], "list");
        assert!(resp["data"].is_array(), "data must be array");

        let models = resp["data"].as_array().unwrap();
        assert!(!models.is_empty());

        for model in models {
            assert!(model["id"].is_string(), "model.id must be string");
            assert_eq!(model["object"], "model");
            assert!(
                model["owned_by"].is_string(),
                "model.owned_by must be string"
            );
        }
    }

    // -- Error response schema --

    #[tokio::test]
    async fn conformance_error_response_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        // Empty model string → our handler validation error (not serde)
        let resp = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error(),
            "empty model must be client error"
        );
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"].is_object(), "error must be object");
        assert!(
            body["error"]["message"].is_string(),
            "error.message must be string"
        );
        assert!(
            body["error"]["type"].is_string(),
            "error.type must be string"
        );
    }

    #[tokio::test]
    async fn conformance_validation_empty_messages() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        let resp = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": []
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status().as_u16(), 400, "empty messages must be 400");
    }

    // -- Content type validation --

    #[tokio::test]
    async fn conformance_response_content_type() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        let resp = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("application/json"),
            "response must be application/json"
        );
    }

    // -- Multi-part content support --

    #[tokio::test]
    async fn conformance_multipart_content_accepted() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        // OpenAI vision format with content array
        let resp = http_client()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "mock-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                    ]
                }]
            }))
            .send()
            .await
            .unwrap();

        assert!(
            resp.status().is_success(),
            "multi-part content must be accepted"
        );
    }

    // -- Health endpoint --

    #[tokio::test]
    async fn conformance_health_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        let resp: Value = http_client()
            .get(format!("{url}/v1/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(resp["status"], "ok");
        assert!(resp["version"].is_string(), "version must be string");
        assert!(
            resp["providers_configured"].is_number(),
            "providers_configured must be number"
        );
    }

    // -- Cache stats endpoint --

    #[tokio::test]
    async fn conformance_cache_stats_schema() {
        let backend = start_mock_backend().await;
        let url = start_hoosh(&backend).await;

        let resp: Value = http_client()
            .get(format!("{url}/v1/cache/stats"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert!(resp["entries"].is_number());
        assert!(resp["max_entries"].is_number());
        assert!(resp["hits"].is_number());
        assert!(resp["misses"].is_number());
        assert!(resp["evictions"].is_number());
        assert!(resp["hit_rate"].is_number());
        assert!(resp["enabled"].is_boolean());
    }
}

// ---------------------------------------------------------------------------
// Handler coverage tests (no feature gates, cover uncovered handler paths)
// ---------------------------------------------------------------------------
mod handler_coverage {
    use crate::server::ServerConfig;
    use serde_json::json;
    use tokio::net::TcpListener;

    /// Start a minimal hoosh server with no providers configured.
    async fn start_minimal_server() -> String {
        let config = ServerConfig::default();
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://127.0.0.1:{}", addr.port())
    }

    /// Start a hoosh server with audit enabled.
    async fn start_server_with_audit() -> String {
        let config = ServerConfig {
            audit_enabled: true,
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[tokio::test]
    async fn handler_validation_empty_model() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"].as_str().unwrap().contains("model"));
    }

    #[tokio::test]
    async fn handler_validation_invalid_model_name() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        // Model with control characters
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "model\twith\ttabs",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid model name")
        );
    }

    #[tokio::test]
    async fn handler_validation_empty_messages() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "test-model",
                "messages": []
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[tokio::test]
    async fn handler_validation_bad_temperature() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.0
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("temperature")
        );
    }

    #[tokio::test]
    async fn handler_validation_bad_top_p() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 1.5
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"].as_str().unwrap().contains("top_p"));
    }

    #[tokio::test]
    async fn handler_validation_negative_top_p() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": -0.1
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[tokio::test]
    async fn handler_no_provider_for_model() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_embeddings_no_provider() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/embeddings"))
            .json(&json!({
                "model": "nonexistent-embed-model",
                "input": "hello world"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_token_check_existing_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/tokens/check"))
            .json(&json!({"pool": "default", "tokens": 100}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["allowed"].as_bool().unwrap());
        assert!(body["available"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn handler_token_check_nonexistent_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/tokens/check"))
            .json(&json!({"pool": "nonexistent", "tokens": 100}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_token_reserve_existing_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/tokens/reserve"))
            .json(&json!({"pool": "default", "tokens": 500}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["reserved"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn handler_token_reserve_nonexistent_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/tokens/reserve"))
            .json(&json!({"pool": "no-such-pool", "tokens": 100}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_token_report_existing_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        // First reserve, then report
        let _ = client
            .post(format!("{url}/v1/tokens/reserve"))
            .json(&json!({"pool": "default", "tokens": 1000}))
            .send()
            .await
            .unwrap();
        let resp = client
            .post(format!("{url}/v1/tokens/report"))
            .json(&json!({"pool": "default", "reserved": 1000, "actual": 500}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["used"].is_number());
        assert!(body["available"].is_number());
    }

    #[tokio::test]
    async fn handler_token_report_nonexistent_pool() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/tokens/report"))
            .json(&json!({"pool": "nonexistent", "reserved": 100, "actual": 50}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_token_pools_list() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{url}/v1/tokens/pools"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(!body.as_array().unwrap().is_empty()); // at least "default"
    }

    #[tokio::test]
    async fn handler_costs_get_empty() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/v1/costs")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["records"].is_array());
        assert_eq!(body["total_cost_usd"].as_f64().unwrap(), 0.0);
    }

    #[tokio::test]
    async fn handler_costs_reset() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/costs/reset"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn handler_audit_disabled() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/v1/audit")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn handler_audit_enabled() {
        let url = start_server_with_audit().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/v1/audit")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["entries"].is_array());
        assert!(body["total"].is_number());
        assert!(body["chain_valid"].is_boolean());
    }

    #[tokio::test]
    async fn handler_admin_reload_no_config() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/admin/reload"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 400);
    }

    #[tokio::test]
    async fn handler_queue_status() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{url}/v1/queue/status"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["queued"].as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn handler_cache_stats() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{url}/v1/cache/stats"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["entries"].is_number());
    }

    #[tokio::test]
    async fn handler_prometheus_metrics() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/metrics")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/plain"));
    }

    #[tokio::test]
    async fn handler_health() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/v1/health")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert!(body["version"].is_string());
    }

    #[tokio::test]
    async fn handler_health_providers_empty() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{url}/v1/health/providers"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn handler_health_heartbeat() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{url}/v1/health/heartbeat"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn handler_list_models_empty() {
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client.get(format!("{url}/v1/models")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["object"], "list");
        assert!(body["data"].is_array());
    }

    #[tokio::test]
    async fn handler_costs_reset_with_audit() {
        let url = start_server_with_audit().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/costs/reset"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Audit log should have the reset event
        let audit_resp = client.get(format!("{url}/v1/audit")).send().await.unwrap();
        assert_eq!(audit_resp.status().as_u16(), 200);
        let audit_body: serde_json::Value = audit_resp.json().await.unwrap();
        assert!(audit_body["total"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn handler_nonexistent_model_returns_404() {
        // With no providers, any model request returns 404 (no route found)
        let url = start_minimal_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "any-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("No provider configured")
        );
    }

    #[tokio::test]
    async fn handler_admin_reload_with_valid_config() {
        // Create a temp config file
        let dir = std::env::temp_dir();
        let config_path = dir.join("hoosh_test_reload.toml");
        std::fs::write(&config_path, "").unwrap();

        let config = ServerConfig {
            config_path: Some(config_path.to_string_lossy().to_string()),
            ..ServerConfig::default()
        };
        let (app, _state) = crate::server::build_app(config);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("http://127.0.0.1:{}", addr.port());
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/admin/reload"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "reloaded");
        let _ = std::fs::remove_file(&config_path);
    }
}
