use criterion::{Criterion, criterion_group, criterion_main};

use hoosh::inference::{InferenceRequest, Message, Role};
use hoosh::provider::openai_compat::OpenAiCompatibleProvider;
use hoosh::provider::{ProviderRegistry, ProviderType};
use hoosh::router::ProviderRoute;

// ---------------------------------------------------------------------------
// ProviderRegistry benchmarks
// ---------------------------------------------------------------------------

fn make_routes(n: usize) -> Vec<ProviderRoute> {
    let types = [
        ProviderType::Ollama,
        ProviderType::LlamaCpp,
        ProviderType::LmStudio,
        ProviderType::LocalAi,
        ProviderType::Synapse,
    ];
    (0..n)
        .map(|i| ProviderRoute {
            provider: types[i % types.len()],
            priority: i as u32,
            model_patterns: vec![format!("model-{}*", i)],
            enabled: true,
            base_url: format!("http://provider-{}:{}", i, 8080 + i),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
        })
        .collect()
}

fn bench_registry_register(c: &mut Criterion) {
    let routes = make_routes(20);
    c.bench_function("registry_register_20_routes", |b| {
        b.iter(|| {
            let mut registry = ProviderRegistry::new();
            for route in &routes {
                registry.register_from_route(route);
            }
            registry
        })
    });
}

fn bench_registry_lookup(c: &mut Criterion) {
    let routes = make_routes(20);
    let mut registry = ProviderRegistry::new();
    for route in &routes {
        registry.register_from_route(route);
    }

    c.bench_function("registry_lookup_hit", |b| {
        b.iter(|| registry.get(ProviderType::Ollama, "http://provider-0:8080"))
    });

    c.bench_function("registry_lookup_miss", |b| {
        b.iter(|| registry.get(ProviderType::Whisper, "http://nonexistent:9999"))
    });
}

// ---------------------------------------------------------------------------
// build_chat_body (via OpenAiCompatibleProvider construction + body building)
// ---------------------------------------------------------------------------

fn bench_provider_construction(c: &mut Criterion) {
    c.bench_function("openai_compat_new", |b| {
        b.iter(|| {
            OpenAiCompatibleProvider::new("http://localhost:8080", None, ProviderType::LlamaCpp, None)
        })
    });

    c.bench_function("openai_compat_new_with_key", |b| {
        b.iter(|| {
            OpenAiCompatibleProvider::new(
                "http://localhost:8080/",
                Some("sk-test-key-12345".into()),
                ProviderType::OpenAi,
                None,
            )
        })
    });
}

fn bench_inference_request_construction(c: &mut Criterion) {
    c.bench_function("inference_request_simple", |b| {
        b.iter(|| InferenceRequest {
            model: "llama3".into(),
            prompt: "Explain Rust ownership.".into(),
            ..Default::default()
        })
    });

    c.bench_function("inference_request_with_messages", |b| {
        b.iter(|| InferenceRequest {
            model: "llama3".into(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: "You are a helpful assistant.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Explain Rust ownership.".into(),
                },
                Message {
                    role: Role::Assistant,
                    content: "Rust uses an ownership model...".into(),
                },
                Message {
                    role: Role::User,
                    content: "Give me an example.".into(),
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            ..Default::default()
        })
    });
}

// ---------------------------------------------------------------------------
// Serde serialization (request → JSON, as providers do internally)
// ---------------------------------------------------------------------------

fn bench_request_serialization(c: &mut Criterion) {
    let simple = InferenceRequest {
        model: "llama3".into(),
        prompt: "Hello world".into(),
        ..Default::default()
    };

    let complex = InferenceRequest {
        model: "llama3".into(),
        messages: (0..10)
            .map(|i| Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                content: format!("Message number {} with some content to serialize.", i),
            })
            .collect(),
        temperature: Some(0.7),
        max_tokens: Some(2048),
        top_p: Some(0.9),
        stream: true,
        ..Default::default()
    };

    c.bench_function("serialize_simple_request", |b| {
        b.iter(|| serde_json::to_string(&simple).unwrap())
    });

    c.bench_function("serialize_10_message_request", |b| {
        b.iter(|| serde_json::to_string(&complex).unwrap())
    });
}

// ---------------------------------------------------------------------------
// Cache benchmarks
// ---------------------------------------------------------------------------

fn bench_cache_operations(c: &mut Criterion) {
    use hoosh::cache::{CacheConfig, ResponseCache};

    let cache = ResponseCache::new(CacheConfig {
        max_entries: 10_000,
        ttl_secs: 300,
        enabled: true,
    });

    // Pre-populate
    for i in 0..1000 {
        cache.insert(format!("key-{i}"), format!("value-{i}"));
    }

    c.bench_function("cache_get_hit", |b| b.iter(|| cache.get("key-500")));

    c.bench_function("cache_get_miss", |b| b.iter(|| cache.get("nonexistent")));

    c.bench_function("cache_insert", |b| {
        let mut i = 2000u64;
        b.iter(|| {
            cache.insert(format!("bench-{i}"), "value".into());
            i += 1;
        })
    });

    // Cache key generation
    use hoosh::cache::cache_key;
    use hoosh::inference::{Message, Role};
    let msgs: Vec<Message> = (0..5)
        .map(|i| Message {
            role: if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            content: format!("message {i}"),
        })
        .collect();

    c.bench_function("cache_key_5_messages", |b| {
        b.iter(|| cache_key("llama3", &msgs))
    });
}

// ---------------------------------------------------------------------------
// Token budget benchmarks
// ---------------------------------------------------------------------------

fn bench_budget_operations(c: &mut Criterion) {
    use hoosh::budget::{TokenBudget, TokenPool};

    c.bench_function("budget_reserve_report_cycle", |b| {
        let mut budget = TokenBudget::new();
        budget.add_pool(TokenPool::new("bench", 1_000_000_000));
        b.iter(|| {
            budget.reserve("bench", 1000);
            budget.report("bench", 1000, 800);
        })
    });

    c.bench_function("budget_check", |b| {
        let mut budget = TokenBudget::new();
        budget.add_pool(TokenPool::new("bench", 1_000_000));
        b.iter(|| budget.check("bench", 100))
    });

    c.bench_function("pool_available", |b| {
        let pool = TokenPool::new("bench", 1_000_000);
        b.iter(|| pool.available())
    });
}

// ---------------------------------------------------------------------------
// Config parsing benchmarks
// ---------------------------------------------------------------------------

fn bench_config_parsing(c: &mut Criterion) {
    let minimal_toml = "";
    let full_toml = r#"
[server]
bind = "0.0.0.0"
port = 9000
strategy = "round_robin"

[cache]
max_entries = 5000
ttl_secs = 600
enabled = true

[[providers]]
type = "Ollama"
base_url = "http://gpu-box:11434"
priority = 1
models = ["llama*", "mistral*", "qwen*"]

[[providers]]
type = "OpenAi"
api_key = "sk-test-key"
priority = 10
models = ["gpt-*", "o1-*"]
max_tokens_limit = 4096

[[providers]]
type = "Anthropic"
api_key = "sk-ant-test"
priority = 10
models = ["claude-*"]

[[budgets]]
name = "default"
capacity = 1000000

[[budgets]]
name = "agents"
capacity = 500000
"#;

    c.bench_function("config_parse_minimal", |b| {
        b.iter(|| {
            let _: hoosh::config::HooshConfig = toml::from_str(minimal_toml).unwrap();
        })
    });

    c.bench_function("config_parse_full", |b| {
        b.iter(|| {
            let _: hoosh::config::HooshConfig = toml::from_str(full_toml).unwrap();
        })
    });

    c.bench_function("config_to_server_config", |b| {
        b.iter(|| {
            let config: hoosh::config::HooshConfig = toml::from_str(full_toml).unwrap();
            config.into_server_config(None, None, None)
        })
    });
}

// ---------------------------------------------------------------------------
// Cache eviction benchmark
// ---------------------------------------------------------------------------

fn bench_cache_eviction(c: &mut Criterion) {
    use hoosh::cache::{CacheConfig, ResponseCache};

    c.bench_function("cache_insert_at_capacity", |b| {
        let cache = ResponseCache::new(CacheConfig {
            max_entries: 100,
            ttl_secs: 300,
            enabled: true,
        });
        // Fill to capacity
        for i in 0..100 {
            cache.insert(format!("pre-{i}"), format!("val-{i}"));
        }
        let mut i = 1000u64;
        b.iter(|| {
            cache.insert(format!("evict-{i}"), "new-value".into());
            i += 1;
        })
    });
}

criterion_group!(
    benches,
    bench_registry_register,
    bench_registry_lookup,
    bench_provider_construction,
    bench_inference_request_construction,
    bench_request_serialization,
    bench_cache_operations,
    bench_budget_operations,
    bench_config_parsing,
    bench_cache_eviction,
);
criterion_main!(benches);
