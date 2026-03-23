//! Benchmarks for per-request hot-path operations: auth, rate limiting,
//! cost tracking, audit chain, event bus, and health-aware routing.

use criterion::{Criterion, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// Auth token hashing & verification
// ---------------------------------------------------------------------------

fn bench_auth(c: &mut Criterion) {
    use hoosh::middleware::auth::hash_token;

    let digest = hash_token("sk-test-token-abcdef1234567890");

    c.bench_function("auth_hash_token", |b| {
        b.iter(|| hash_token("sk-test-token-abcdef1234567890"))
    });

    // verify_token is private, but hash_token + constant-time compare is the
    // same cost as what auth_middleware does. Benchmark hash + XOR compare.
    c.bench_function("auth_verify_token", |b| {
        b.iter(|| {
            let candidate = hash_token("sk-test-token-abcdef1234567890");
            candidate
                .iter()
                .zip(digest.iter())
                .fold(0u8, |acc, (x, y)| acc | (x ^ y))
                == 0
        })
    });

    // Worst case: wrong token (same cost due to constant-time)
    c.bench_function("auth_verify_wrong_token", |b| {
        b.iter(|| {
            let candidate = hash_token("sk-wrong-token-xyz");
            candidate
                .iter()
                .zip(digest.iter())
                .fold(0u8, |acc, (x, y)| acc | (x ^ y))
                == 0
        })
    });
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

fn bench_rate_limit(c: &mut Criterion) {
    use hoosh::middleware::rate_limit::RateLimitRegistry;

    c.bench_function("rate_limit_check_unconfigured", |b| {
        let reg = RateLimitRegistry::new();
        b.iter(|| reg.check("unknown-provider"))
    });

    c.bench_function("rate_limit_check_within_limit", |b| {
        let reg = RateLimitRegistry::new();
        reg.configure("ollama:http://localhost:11434", 10_000);
        b.iter(|| reg.check("ollama:http://localhost:11434"))
    });

    c.bench_function("rate_limit_check_at_capacity", |b| {
        let reg = RateLimitRegistry::new();
        reg.configure("test-provider", 100);
        // Fill the window
        for _ in 0..100 {
            reg.check("test-provider");
        }
        b.iter(|| reg.check("test-provider"))
    });
}

// ---------------------------------------------------------------------------
// Cost tracking (pricing lookup + DashMap record)
// ---------------------------------------------------------------------------

fn bench_cost_tracking(c: &mut Criterion) {
    use hoosh::cost::CostTracker;
    use hoosh::inference::TokenUsage;
    use hoosh::provider::ProviderType;

    let usage = TokenUsage {
        prompt_tokens: 500,
        completion_tokens: 200,
        total_tokens: 700,
    };

    c.bench_function("cost_record_known_model", |b| {
        let tracker = CostTracker::new();
        b.iter(|| {
            tracker.record(
                ProviderType::OpenAi,
                "https://api.openai.com",
                "gpt-4o",
                &usage,
            )
        })
    });

    c.bench_function("cost_record_prefix_match", |b| {
        let tracker = CostTracker::new();
        b.iter(|| {
            tracker.record(
                ProviderType::Anthropic,
                "https://api.anthropic.com",
                "claude-sonnet-4-20250514",
                &usage,
            )
        })
    });

    c.bench_function("cost_record_fallback_pricing", |b| {
        let tracker = CostTracker::new();
        b.iter(|| {
            tracker.record(
                ProviderType::OpenAi,
                "https://api.openai.com",
                "unknown-future-model",
                &usage,
            )
        })
    });

    c.bench_function("cost_record_local_free", |b| {
        let tracker = CostTracker::new();
        b.iter(|| {
            tracker.record(
                ProviderType::Ollama,
                "http://localhost:11434",
                "llama3:8b",
                &usage,
            )
        })
    });

    c.bench_function("cost_total_10_providers", |b| {
        let tracker = CostTracker::new();
        for i in 0..10 {
            tracker.record(
                ProviderType::OpenAi,
                &format!("https://api-{i}.openai.com"),
                "gpt-4o",
                &usage,
            );
        }
        b.iter(|| tracker.total_cost())
    });
}

// ---------------------------------------------------------------------------
// Audit chain (HMAC-SHA256 record + verify)
// ---------------------------------------------------------------------------

fn bench_audit(c: &mut Criterion) {
    use hoosh::audit::AuditChain;

    c.bench_function("audit_record", |b| {
        let chain = AuditChain::new(b"bench-signing-key-32bytes!!!!!!!!", 100_000);
        b.iter(|| {
            chain.record(
                "inference.response",
                "info",
                "Inference completed for model llama3",
                Some("ollama"),
                Some("llama3"),
                None,
            )
        })
    });

    c.bench_function("audit_record_with_metadata", |b| {
        let chain = AuditChain::new(b"bench-signing-key-32bytes!!!!!!!!", 100_000);
        let meta = serde_json::json!({
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
            "cost_usd": 0.0075,
        });
        b.iter(|| {
            chain.record(
                "inference.response",
                "info",
                "Inference completed",
                Some("openai"),
                Some("gpt-4o"),
                Some(meta.clone()),
            )
        })
    });

    c.bench_function("audit_verify_100_entries", |b| {
        let chain = AuditChain::new(b"bench-key", 100_000);
        for i in 0..100 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }
        b.iter(|| chain.verify())
    });

    c.bench_function("audit_verify_1000_entries", |b| {
        let chain = AuditChain::new(b"bench-key", 100_000);
        for i in 0..1000 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }
        b.iter(|| chain.verify())
    });

    c.bench_function("audit_recent_10_of_1000", |b| {
        let chain = AuditChain::new(b"bench-key", 100_000);
        for i in 0..1000 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }
        b.iter(|| chain.recent(10))
    });

    c.bench_function("audit_snapshot_100_of_1000", |b| {
        let chain = AuditChain::new(b"bench-key", 100_000);
        for i in 0..1000 {
            chain.record("event", "info", &format!("entry {i}"), None, None, None);
        }
        b.iter(|| chain.snapshot(100))
    });
}

// ---------------------------------------------------------------------------
// Event bus publish
// ---------------------------------------------------------------------------

fn bench_event_bus(c: &mut Criterion) {
    use hoosh::events::{ProviderEvent, new_event_bus, topics};

    c.bench_function("event_publish_no_subscribers", |b| {
        let bus = new_event_bus();
        b.iter(|| {
            bus.publish(
                topics::INFERENCE,
                ProviderEvent::InferenceCompleted {
                    provider: "ollama".into(),
                    model: "llama3".into(),
                    latency_ms: 42,
                    tokens: 700,
                },
            )
        })
    });

    c.bench_function("event_publish_with_subscriber", |b| {
        let bus = new_event_bus();
        let _rx = bus.subscribe(topics::INFERENCE);
        b.iter(|| {
            bus.publish(
                topics::INFERENCE,
                ProviderEvent::InferenceCompleted {
                    provider: "ollama".into(),
                    model: "llama3".into(),
                    latency_ms: 42,
                    tokens: 700,
                },
            )
        })
    });
}

// ---------------------------------------------------------------------------
// Routing with health map filtering
// ---------------------------------------------------------------------------

fn bench_routing_with_health(c: &mut Criterion) {
    use hoosh::health::{ProviderHealthState, new_health_map};
    use hoosh::provider::ProviderType;
    use hoosh::router::{ProviderRoute, Router, RoutingStrategy};

    let routes: Vec<ProviderRoute> = (0..20)
        .map(|i| ProviderRoute {
            provider: if i % 2 == 0 {
                ProviderType::Ollama
            } else {
                ProviderType::OpenAi
            },
            priority: i as u32,
            model_patterns: vec![format!("model-{}*", i)],
            enabled: true,
            base_url: format!("http://provider-{}", i),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
        })
        .collect();

    // Half providers unhealthy
    let health_map = new_health_map();
    for (i, route) in routes.iter().enumerate() {
        let key = (route.provider, route.base_url.clone());
        health_map.insert(
            key,
            ProviderHealthState {
                is_healthy: i % 3 != 0, // every 3rd provider unhealthy
                last_check: std::time::Instant::now(),
                consecutive_failures: if i % 3 == 0 { 3 } else { 0 },
                last_error: None,
            },
        );
    }

    let mut router = Router::new(routes, RoutingStrategy::Priority);
    router.set_health_map(health_map);

    c.bench_function("route_select_with_health_map", |b| {
        b.iter(|| router.select("model-15-large"))
    });

    // Lowest-latency strategy with health map
    let routes2: Vec<ProviderRoute> = (0..10)
        .map(|i| ProviderRoute {
            provider: ProviderType::Ollama,
            priority: 1,
            model_patterns: vec!["*".into()],
            enabled: true,
            base_url: format!("http://provider-{}", i),
            api_key: None,
            max_tokens_limit: None,
            rate_limit_rpm: None,
            tls_config: None,
        })
        .collect();
    let health_map2 = new_health_map();
    for route in &routes2 {
        let key = (route.provider, route.base_url.clone());
        health_map2.insert(
            key,
            ProviderHealthState {
                is_healthy: true,
                last_check: std::time::Instant::now(),
                consecutive_failures: 0,
                last_error: None,
            },
        );
    }
    let mut router2 = Router::new(routes2, RoutingStrategy::LowestLatency);
    router2.set_health_map(health_map2);
    // Seed some latencies
    for i in 0..10 {
        router2.report_latency(
            ProviderType::Ollama,
            &format!("http://provider-{}", i),
            (i * 10 + 5) as u64,
        );
    }

    c.bench_function("route_lowest_latency_10_providers", |b| {
        b.iter(|| router2.select("any-model"))
    });
}

// ---------------------------------------------------------------------------
// Queue operations
// ---------------------------------------------------------------------------

fn bench_queue(c: &mut Criterion) {
    use hoosh::inference::InferenceRequest;
    use hoosh::queue::InferenceQueue;

    // Re-export from majra
    use majra::queue::Priority;

    c.bench_function("queue_enqueue_dequeue_cycle", |b| {
        let queue = InferenceQueue::new();
        let req = hoosh::queue::QueuedRequest {
            request: InferenceRequest::default(),
            model: "llama3".into(),
            pool: "default".into(),
            request_id: "bench-req".into(),
        };
        b.iter(|| {
            queue.enqueue(req.clone(), Priority::Normal);
            queue.dequeue()
        })
    });

    c.bench_function("queue_enqueue_priority_sort", |b| {
        let queue = InferenceQueue::new();
        let req = hoosh::queue::QueuedRequest {
            request: InferenceRequest::default(),
            model: "test".into(),
            pool: "default".into(),
            request_id: "r".into(),
        };
        b.iter(|| {
            queue.enqueue(req.clone(), Priority::Background);
            queue.enqueue(req.clone(), Priority::Critical);
            queue.enqueue(req.clone(), Priority::Normal);
            queue.dequeue();
            queue.dequeue();
            queue.dequeue();
        })
    });
}

criterion_group!(
    benches,
    bench_auth,
    bench_rate_limit,
    bench_cost_tracking,
    bench_audit,
    bench_event_bus,
    bench_routing_with_health,
    bench_queue,
);
criterion_main!(benches);
