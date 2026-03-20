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
        },
        ProviderRoute {
            provider: ProviderType::OpenAi,
            priority: 2,
            model_patterns: vec!["gpt-*".into()],
            enabled: true,
            base_url: "https://api.openai.com".into(),
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
    assert_eq!(cache.get("prompt-hash").unwrap(), "cached response");
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
