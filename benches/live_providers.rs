//! Live provider benchmarks — requires running backends.
//!
//! Run with:
//!   cargo bench --bench live_providers
//!
//! Requires Ollama running on localhost:11434 with at least one model pulled.
//! Skips gracefully if Ollama is not reachable.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use hoosh::inference::{InferenceRequest, Message, Role};
use hoosh::provider::LlmProvider;

#[cfg(feature = "ollama")]
use hoosh::provider::ollama::OllamaProvider;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Returns (provider, first_model_id) if Ollama is reachable and has models.
#[cfg(feature = "ollama")]
fn ollama_if_available() -> Option<(OllamaProvider, String)> {
    let rt = runtime();
    let provider = OllamaProvider::new("http://127.0.0.1:11434");
    let healthy = rt.block_on(provider.health_check()).ok()?;
    if !healthy {
        return None;
    }
    let models = rt.block_on(provider.list_models()).ok()?;
    let first = models.first()?.id.clone();
    Some((provider, first))
}

#[cfg(feature = "ollama")]
fn bench_ollama_health(c: &mut Criterion) {
    let Some((provider, _)) = ollama_if_available() else {
        eprintln!("Skipping Ollama health bench — not reachable");
        return;
    };
    let rt = runtime();

    c.bench_function("ollama_health_check", |b| {
        b.iter(|| rt.block_on(provider.health_check()).unwrap())
    });
}

#[cfg(feature = "ollama")]
fn bench_ollama_list_models(c: &mut Criterion) {
    let Some((provider, _)) = ollama_if_available() else {
        eprintln!("Skipping Ollama list_models bench — not reachable");
        return;
    };
    let rt = runtime();

    c.bench_function("ollama_list_models", |b| {
        b.iter(|| rt.block_on(provider.list_models()).unwrap())
    });
}

#[cfg(feature = "ollama")]
fn bench_ollama_infer(c: &mut Criterion) {
    let Some((provider, model)) = ollama_if_available() else {
        eprintln!("Skipping Ollama infer bench — not reachable");
        return;
    };
    let rt = runtime();

    let mut group = c.benchmark_group("ollama_infer");
    // Inference is slow — reduce sample size and increase measurement time
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Short prompt, constrained output
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
        |b, req| b.iter(|| rt.block_on(provider.infer(req)).unwrap()),
    );

    // Medium prompt, moderate output
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
        |b, req| b.iter(|| rt.block_on(provider.infer(req)).unwrap()),
    );

    group.finish();
}

#[cfg(feature = "ollama")]
fn bench_ollama_infer_multiturn(c: &mut Criterion) {
    let Some((provider, model)) = ollama_if_available() else {
        eprintln!("Skipping Ollama multiturn bench — not reachable");
        return;
    };
    let rt = runtime();

    let mut group = c.benchmark_group("ollama_infer_multiturn");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    let req = InferenceRequest {
        model: model.clone(),
        messages: vec![
            Message {
                role: Role::System,
                content: "Answer in one sentence.".into(),
            },
            Message {
                role: Role::User,
                content: "What is Rust?".into(),
            },
            Message {
                role: Role::Assistant,
                content:
                    "Rust is a systems programming language focused on safety and performance."
                        .into(),
            },
            Message {
                role: Role::User,
                content: "What about its type system?".into(),
            },
        ],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..Default::default()
    };

    group.bench_with_input(
        BenchmarkId::new("4_message_history", &model),
        &req,
        |b, req| b.iter(|| rt.block_on(provider.infer(req)).unwrap()),
    );

    group.finish();
}

#[cfg(feature = "ollama")]
fn bench_ollama_stream_throughput(c: &mut Criterion) {
    let Some((provider, model)) = ollama_if_available() else {
        eprintln!("Skipping Ollama stream bench — not reachable");
        return;
    };
    let rt = runtime();

    let mut group = c.benchmark_group("ollama_stream");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    let req = InferenceRequest {
        model: model.clone(),
        prompt: "Count from 1 to 10.".into(),
        max_tokens: Some(50),
        temperature: Some(0.0),
        stream: true,
        ..Default::default()
    };

    group.bench_with_input(
        BenchmarkId::new("stream_50_tokens", &model),
        &req,
        |b, req| {
            b.iter(|| {
                rt.block_on(async {
                    let mut rx = provider.infer_stream(req.clone()).await.unwrap();
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
// Hardware detection benchmarks (ai-hwaccel)
// ---------------------------------------------------------------------------

#[cfg(feature = "hwaccel")]
fn bench_hwaccel_detect(c: &mut Criterion) {
    use hoosh::hardware::HardwareManager;

    c.bench_function("hwaccel_detect", |b| b.iter(|| HardwareManager::detect()));
}

#[cfg(feature = "hwaccel")]
fn bench_hwaccel_summary(c: &mut Criterion) {
    use hoosh::hardware::HardwareManager;

    let hw = HardwareManager::detect();
    c.bench_function("hwaccel_summary", |b| b.iter(|| hw.summary()));
}

#[cfg(feature = "hwaccel")]
fn bench_hwaccel_recommend_placement(c: &mut Criterion) {
    use hoosh::hardware::HardwareManager;

    let hw = HardwareManager::detect();
    let providers = vec![
        "ollama".to_string(),
        "llamacpp".to_string(),
        "localai".to_string(),
    ];

    let mut group = c.benchmark_group("hwaccel_recommend_placement");

    // Small model (1B params)
    group.bench_function("1b_params", |b| {
        b.iter(|| hw.recommend_placement(1_000_000_000, &providers))
    });

    // Medium model (7B params)
    group.bench_function("7b_params", |b| {
        b.iter(|| hw.recommend_placement(7_000_000_000, &providers))
    });

    // Large model (70B params)
    group.bench_function("70b_params", |b| {
        b.iter(|| hw.recommend_placement(70_000_000_000, &providers))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion group assembly
// ---------------------------------------------------------------------------

#[cfg(all(feature = "ollama", feature = "hwaccel"))]
criterion_group!(
    benches,
    bench_ollama_health,
    bench_ollama_list_models,
    bench_ollama_infer,
    bench_ollama_infer_multiturn,
    bench_ollama_stream_throughput,
    bench_hwaccel_detect,
    bench_hwaccel_summary,
    bench_hwaccel_recommend_placement,
);

#[cfg(all(feature = "ollama", not(feature = "hwaccel")))]
criterion_group!(
    benches,
    bench_ollama_health,
    bench_ollama_list_models,
    bench_ollama_infer,
    bench_ollama_infer_multiturn,
    bench_ollama_stream_throughput,
);

#[cfg(all(not(feature = "ollama"), feature = "hwaccel"))]
criterion_group!(
    benches,
    bench_hwaccel_detect,
    bench_hwaccel_summary,
    bench_hwaccel_recommend_placement,
);

#[cfg(not(any(feature = "ollama", feature = "hwaccel")))]
fn no_op(_c: &mut Criterion) {
    eprintln!("No live features enabled — skipping live benchmarks");
}

#[cfg(not(any(feature = "ollama", feature = "hwaccel")))]
criterion_group!(benches, no_op);

criterion_main!(benches);
