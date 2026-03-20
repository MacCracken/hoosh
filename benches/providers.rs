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
            OpenAiCompatibleProvider::new("http://localhost:8080", None, ProviderType::LlamaCpp)
        })
    });

    c.bench_function("openai_compat_new_with_key", |b| {
        b.iter(|| {
            OpenAiCompatibleProvider::new(
                "http://localhost:8080/",
                Some("sk-test-key-12345".into()),
                ProviderType::OpenAi,
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

criterion_group!(
    benches,
    bench_registry_register,
    bench_registry_lookup,
    bench_provider_construction,
    bench_inference_request_construction,
    bench_request_serialization,
);
criterion_main!(benches);
