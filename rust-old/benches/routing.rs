use criterion::{Criterion, criterion_group, criterion_main};

use hoosh::provider::ProviderType;
use hoosh::router::{ProviderRoute, Router, RoutingStrategy};

fn make_routes(n: usize) -> Vec<ProviderRoute> {
    (0..n)
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
        .collect()
}

fn bench_routing(c: &mut Criterion) {
    let routes = make_routes(20);
    let router = Router::new(routes, RoutingStrategy::Priority);

    c.bench_function("route_select_20_providers", |b| {
        b.iter(|| router.select("model-15-large"))
    });
}

fn bench_routing_round_robin(c: &mut Criterion) {
    let routes = make_routes(10);
    let router = Router::new(routes, RoutingStrategy::RoundRobin);

    c.bench_function("route_round_robin_10", |b| {
        b.iter(|| router.select("model-5-base"))
    });
}

criterion_group!(benches, bench_routing, bench_routing_round_robin);
criterion_main!(benches);
