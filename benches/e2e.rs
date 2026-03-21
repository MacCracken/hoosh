//! End-to-end benchmarks — measures full round-trip through the hoosh server.
//!
//! These benchmarks measure what downstream consumers (AgnosAI, tarang, daimon)
//! actually experience: HooshClient → hoosh HTTP server → Ollama → response.
//!
//! Run with:
//!   cargo bench --bench e2e
//!
//! Requires Ollama running on localhost:11434 with at least one model pulled.
//! Skips gracefully if Ollama is not reachable.

use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::net::TcpListener;

use hoosh::client::HooshClient;
use hoosh::config::HooshConfig;
use hoosh::inference::InferenceRequest;
use hoosh::provider::LlmProvider;

#[cfg(feature = "ollama")]
use hoosh::provider::ollama::OllamaProvider;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Spin up a hoosh server with Ollama configured, return the HooshClient and port.
fn start_hoosh_server(rt: &tokio::runtime::Runtime) -> Option<(HooshClient, u16)> {
    // First check Ollama is available
    let ollama = OllamaProvider::new("http://127.0.0.1:11434");
    let healthy = rt.block_on(ollama.health_check()).ok()?;
    if !healthy {
        return None;
    }

    let config_toml = r#"
[server]
bind = "127.0.0.1"
port = 0
strategy = "priority"

[cache]
enabled = false

[[providers]]
type = "Ollama"
priority = 1
models = ["*"]
"#;
    let config: HooshConfig = toml::from_str(config_toml).unwrap();
    let server_config = config.into_server_config(None, None);
    let app = hoosh::server::build_app(server_config);

    let (port_tx, port_rx) = std::sync::mpsc::channel();

    // Spawn server on a separate runtime so it doesn't block the bench runtime
    std::thread::spawn(move || {
        let server_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        server_rt.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            port_tx.send(port).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });

    let port = port_rx.recv_timeout(Duration::from_secs(5)).ok()?;
    let client = HooshClient::new(format!("http://127.0.0.1:{port}"));

    // Wait for server to be ready
    for _ in 0..50 {
        if rt.block_on(client.health()).unwrap_or(false) {
            return Some((client, port));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

/// Get the first available model from Ollama.
#[cfg(feature = "ollama")]
fn first_model(rt: &tokio::runtime::Runtime) -> Option<String> {
    let ollama = OllamaProvider::new("http://127.0.0.1:11434");
    let models = rt.block_on(ollama.list_models()).ok()?;
    models.first().map(|m| m.id.clone())
}

// ---------------------------------------------------------------------------
// End-to-end: HooshClient → hoosh server → Ollama
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_health(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping e2e health bench — Ollama or server not available");
        return;
    };

    c.bench_function("e2e_health_check", |b| {
        b.iter(|| rt.block_on(client.health()).unwrap())
    });
}

#[cfg(feature = "ollama")]
fn bench_e2e_list_models(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping e2e list_models bench — not available");
        return;
    };

    c.bench_function("e2e_list_models", |b| {
        b.iter(|| rt.block_on(client.list_models()).unwrap())
    });
}

#[cfg(feature = "ollama")]
fn bench_e2e_infer(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping e2e infer bench — not available");
        return;
    };
    let Some(model) = first_model(&rt) else {
        eprintln!("Skipping e2e infer bench — no models available");
        return;
    };

    let mut group = c.benchmark_group("e2e_infer");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Short prompt — measures connection + gateway + inference overhead
    let req_short = InferenceRequest {
        model: model.clone(),
        prompt: "Say hi.".into(),
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..Default::default()
    };
    group.bench_with_input(
        BenchmarkId::new("short_prompt_5_tokens", &model),
        &req_short,
        |b, req| b.iter(|| rt.block_on(client.infer(req)).unwrap()),
    );

    // Medium prompt
    let req_medium = InferenceRequest {
        model: model.clone(),
        prompt: "Explain what Rust ownership is in two sentences.".into(),
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..Default::default()
    };
    group.bench_with_input(
        BenchmarkId::new("medium_prompt_50_tokens", &model),
        &req_medium,
        |b, req| b.iter(|| rt.block_on(client.infer(req)).unwrap()),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Connection reuse: cold vs warm connections
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_connection_cold_vs_warm(c: &mut Criterion) {
    let rt = runtime();
    let Some((_client, port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping cold/warm bench — not available");
        return;
    };

    let mut group = c.benchmark_group("e2e_connection");
    group.sample_size(20);

    // Cold connection: create a new HooshClient each time (new TCP connection)
    group.bench_function("cold_new_client_per_request", |b| {
        b.iter(|| {
            let fresh_client = HooshClient::new(format!("http://127.0.0.1:{port}"));
            rt.block_on(fresh_client.health()).unwrap()
        })
    });

    // Warm connection: reuse existing client (pooled TCP connection)
    let warm_client = HooshClient::new(format!("http://127.0.0.1:{port}"));
    // Warm up the connection pool
    rt.block_on(warm_client.health()).unwrap();

    group.bench_function("warm_reused_connection", |b| {
        b.iter(|| rt.block_on(warm_client.health()).unwrap())
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Concurrent requests: parallel agents hitting hoosh
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_concurrent_health(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping concurrent bench — not available");
        return;
    };
    let client = Arc::new(client);

    let mut group = c.benchmark_group("e2e_concurrent");
    group.sample_size(20);

    for concurrency in [1, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("health_concurrent", concurrency),
            &concurrency,
            |b, &n| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::new();
                        for _ in 0..n {
                            let c = client.clone();
                            handles.push(tokio::spawn(async move { c.health().await }));
                        }
                        for h in handles {
                            h.await.unwrap().unwrap();
                        }
                    })
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Latency breakdown: measure gateway overhead separately from inference
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_gateway_overhead(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping gateway overhead bench — not available");
        return;
    };
    let Some(model) = first_model(&rt) else {
        eprintln!("Skipping gateway overhead bench — no models");
        return;
    };

    // Measure direct Ollama vs through-hoosh to isolate gateway overhead
    let ollama_direct = OllamaProvider::new("http://127.0.0.1:11434");
    let req = InferenceRequest {
        model: model.clone(),
        prompt: "Say hi.".into(),
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..Default::default()
    };

    let mut group = c.benchmark_group("e2e_gateway_overhead");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Direct to Ollama (baseline)
    group.bench_with_input(
        BenchmarkId::new("direct_ollama", &model),
        &req,
        |b, req| b.iter(|| rt.block_on(ollama_direct.infer(req)).unwrap()),
    );

    // Through hoosh server (gateway overhead = this - baseline)
    group.bench_with_input(
        BenchmarkId::new("through_hoosh", &model),
        &req,
        |b, req| b.iter(|| rt.block_on(client.infer(req)).unwrap()),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Streaming end-to-end
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_stream(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping e2e stream bench — not available");
        return;
    };
    let Some(model) = first_model(&rt) else {
        eprintln!("Skipping e2e stream bench — no models");
        return;
    };

    let mut group = c.benchmark_group("e2e_stream");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let req = InferenceRequest {
        model: model.clone(),
        prompt: "Count from 1 to 10.".into(),
        max_tokens: Some(50),
        temperature: Some(0.0),
        stream: true,
        ..Default::default()
    };

    // Streaming through hoosh: measures time-to-first-token and total throughput
    group.bench_with_input(
        BenchmarkId::new("stream_50_tokens_through_hoosh", &model),
        &req,
        |b, req| {
            b.iter(|| {
                rt.block_on(async {
                    let mut rx = client.infer_stream(req).await.unwrap();
                    let mut count = 0;
                    while let Some(result) = rx.recv().await {
                        result.unwrap();
                        count += 1;
                    }
                    count
                })
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Sequential multi-request (simulates single-agent-single-task scenario)
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
fn bench_e2e_sequential_requests(c: &mut Criterion) {
    let rt = runtime();
    let Some((client, _port)) = start_hoosh_server(&rt) else {
        eprintln!("Skipping sequential bench — not available");
        return;
    };
    let Some(model) = first_model(&rt) else {
        eprintln!("Skipping sequential bench — no models");
        return;
    };

    let mut group = c.benchmark_group("e2e_sequential");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    // 3 sequential requests (simulates a simple agent loop)
    let requests = vec![
        InferenceRequest {
            model: model.clone(),
            prompt: "What is 2+2?".into(),
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        },
        InferenceRequest {
            model: model.clone(),
            prompt: "What is the capital of France?".into(),
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        },
        InferenceRequest {
            model: model.clone(),
            prompt: "Say goodbye.".into(),
            max_tokens: Some(10),
            temperature: Some(0.0),
            ..Default::default()
        },
    ];

    group.bench_with_input(
        BenchmarkId::new("3_sequential_requests", &model),
        &requests,
        |b, reqs| {
            b.iter(|| {
                rt.block_on(async {
                    for req in reqs {
                        client.infer(req).await.unwrap();
                    }
                })
            })
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion group assembly
// ---------------------------------------------------------------------------

#[cfg(feature = "ollama")]
criterion_group!(
    benches,
    bench_e2e_health,
    bench_e2e_list_models,
    bench_e2e_infer,
    bench_e2e_connection_cold_vs_warm,
    bench_e2e_concurrent_health,
    bench_e2e_gateway_overhead,
    bench_e2e_stream,
    bench_e2e_sequential_requests,
);

#[cfg(not(feature = "ollama"))]
fn no_op(_c: &mut Criterion) {
    eprintln!("Ollama feature not enabled — skipping e2e benchmarks");
}

#[cfg(not(feature = "ollama"))]
criterion_group!(benches, no_op);

criterion_main!(benches);
