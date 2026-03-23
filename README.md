# hoosh

**AI inference gateway for Rust.**

Multi-provider LLM routing, local model serving, speech-to-text, and token budget management — in a single crate. OpenAI-compatible HTTP API. Built on [ai-hwaccel](https://crates.io/crates/ai-hwaccel) for hardware-aware model placement.

> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI.
> Extracted from the [AGNOS](https://github.com/MacCracken/agnosticos) LLM gateway as a standalone, reusable engine.

[![Crates.io](https://img.shields.io/crates/v/hoosh.svg)](https://crates.io/crates/hoosh)
[![CI](https://github.com/MacCracken/hoosh/actions/workflows/ci.yml/badge.svg)](https://github.com/MacCracken/hoosh/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

---

## What it does

hoosh is the **inference backend** — it routes, caches, rate-limits, and budget-tracks LLM requests across providers. It is not a model trainer (that's [Synapse](https://github.com/MacCracken/synapse)) or a model file manager. Applications build their AI features on top of hoosh.

| Capability | Details |
|------------|---------|
| **14 LLM providers** | Ollama, llama.cpp, Synapse, LM Studio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter, Whisper |
| **OpenAI-compatible API** | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` — streaming SSE |
| **Provider routing** | Priority, round-robin, lowest-latency (EMA), direct — with model pattern matching |
| **Authentication** | Bearer token auth middleware with constant-time comparison |
| **Rate limiting** | Per-provider sliding window RPM limits |
| **Token budgets** | Per-agent named pools with reserve/commit/release lifecycle |
| **Cost tracking** | Per-provider/model cost accumulation with static pricing table |
| **Observability** | Prometheus `/metrics`, OpenTelemetry (feature-gated), cryptographic audit log |
| **Health checks** | Background periodic checks, automatic failover, heartbeat tracking (majra) |
| **Response caching** | Thread-safe DashMap cache with TTL eviction |
| **Request queuing** | Priority queue for inference requests (majra) |
| **Event bus** | Pub/sub for provider health changes, inference events (majra) |
| **Hot-reload** | SIGHUP or `POST /v1/admin/reload` — no restart required |
| **TLS security** | Certificate pinning for remote providers, mTLS for local |
| **Speech** | whisper.cpp STT + TTS via HTTP backend (feature-gated) |
| **Hardware-aware** | ai-hwaccel detects GPUs/TPUs/NPUs for model placement |
| **Local-first** | Prefers on-device inference; remote APIs as fallback |

---

## Architecture

```
Clients (tarang, daimon, agnoshi, consumer apps)
    │
    ▼
Auth ──▶ Rate Limiter ──▶ Router (priority, round-robin, lowest-latency)
                              │
    ┌─────────────────────────┤
    │                         │
    ▼                         ▼
Local backends            Remote APIs (TLS pinned / mTLS)
(Ollama, llama.cpp, …)   (OpenAI, Anthropic, DeepSeek, …)
    │                         │
    └────────┬────────────────┘
             ▼
    Cache ◀── Budget ◀── Cost Tracker
             │
    Metrics ◀── Audit Log ◀── Event Bus (majra)
```

See [docs/architecture/overview.md](docs/architecture/overview.md) for the full architecture document.

---

## Quick start

### As a library

```toml
[dependencies]
hoosh = "0.21"
```

```rust
use hoosh::{HooshClient, InferenceRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = HooshClient::new("http://localhost:8088");

    let response = client.infer(&InferenceRequest {
        model: "llama3".into(),
        prompt: "Explain Rust ownership in one sentence.".into(),
        ..Default::default()
    }).await?;

    println!("{}", response.text);
    Ok(())
}
```

### As a server

```bash
# Start the gateway
hoosh serve --port 8088

# One-shot inference
hoosh infer --model llama3 "What is Rust?"

# List models across all providers
hoosh models

# System info (hardware, providers)
hoosh info
```

### OpenAI-compatible API

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Features

| Feature | Backend | Default |
|---------|---------|---------|
| `ollama` | Ollama REST API | yes |
| `llamacpp` | llama.cpp server | yes |
| `synapse` | Synapse server | yes |
| `lmstudio` | LM Studio API | yes |
| `localai` | LocalAI API | yes |
| `openai` | OpenAI API | yes |
| `anthropic` | Anthropic Messages API | yes |
| `deepseek` | DeepSeek API | yes |
| `mistral` | Mistral API | yes |
| `groq` | Groq API | yes |
| `openrouter` | OpenRouter API | yes |
| `grok` | xAI Grok API | yes |
| `whisper` | whisper.cpp STT | no |
| `piper` | Piper TTS | no |
| `hwaccel` | ai-hwaccel hardware detection | yes |
| `otel` | OpenTelemetry tracing | no |
| `all-providers` | All LLM providers | yes |

```toml
# Minimal: just Ollama + llama.cpp for local inference
hoosh = { version = "0.20", default-features = false, features = ["ollama", "llamacpp"] }

# With speech-to-text
hoosh = { version = "0.20", features = ["whisper"] }
```

---

## Key types

### `HooshClient`

HTTP client for downstream consumers. Speaks the OpenAI-compatible API.

```rust
let client = hoosh::HooshClient::new("http://localhost:8088");
let healthy = client.health().await?;
let models = client.list_models().await?;
```

### `InferenceRequest` / `InferenceResponse`

```rust
use hoosh::{InferenceRequest, InferenceResponse};

let req = InferenceRequest {
    model: "claude-sonnet-4-20250514".into(),
    prompt: "Summarise this document.".into(),
    system: Some("You are a technical writer.".into()),
    max_tokens: Some(500),
    temperature: Some(0.3),
    stream: false,
    ..Default::default()
};
```

### `Router`

Provider selection with model pattern matching:

```rust
use hoosh::router::{Router, ProviderRoute, RoutingStrategy};
use hoosh::ProviderType;

let routes = vec![
    ProviderRoute {
        provider: ProviderType::Ollama,
        priority: 1,
        model_patterns: vec!["llama*".into(), "mistral*".into()],
        enabled: true,
        base_url: "http://localhost:11434".into(),
        api_key: None,
        max_tokens_limit: None,
        rate_limit_rpm: None,
        tls_config: None,
    },
];
let router = Router::new(routes, RoutingStrategy::Priority);
let selected = router.select("llama3"); // → Ollama
```

### `TokenBudget`

Per-agent token accounting:

```rust
use hoosh::{TokenBudget, TokenPool};

let mut budget = TokenBudget::new();
budget.add_pool(TokenPool::new("agent-123", 50_000));

// Before inference: reserve estimated tokens
budget.reserve("agent-123", 2000);

// After inference: report actual usage
budget.report("agent-123", 2000, 1847);
```

---

## Dependencies

| Crate | Role |
|-------|------|
| [ai-hwaccel](https://crates.io/crates/ai-hwaccel) | Hardware detection for model placement |
| [majra](https://crates.io/crates/majra) | Priority queues, pub/sub events, heartbeat tracking |
| [axum](https://crates.io/crates/axum) | HTTP server |
| [reqwest](https://crates.io/crates/reqwest) | HTTP client for remote providers (rustls-tls) |
| [prometheus](https://crates.io/crates/prometheus) | Metrics endpoint |
| [dashmap](https://crates.io/crates/dashmap) | Thread-safe caches and registries |
| [hmac](https://crates.io/crates/hmac) + [sha2](https://crates.io/crates/sha2) | Audit chain cryptography |
| [whisper-rs](https://crates.io/crates/whisper-rs) | whisper.cpp Rust bindings (optional) |
| [tokio](https://crates.io/crates/tokio) | Async runtime |

---

## Who uses this

| Project | Usage |
|---------|-------|
| **[AGNOS](https://github.com/MacCracken/agnosticos)** (llm-gateway) | Wraps hoosh as the system-wide inference gateway |
| **[tarang](https://crates.io/crates/tarang)** | Transcription, content description, AI media analysis |
| **[aethersafta](https://github.com/MacCracken/aethersafta)** | Real-time transcription/captioning for streams |
| **[AgnosAI](https://github.com/MacCracken/agnosai)** | Agent crew LLM routing |
| **[Synapse](https://github.com/MacCracken/synapse)** | Inference backend + model management |
| **All AGNOS consumer apps** | Via daimon or direct HTTP |

---

## Roadmap

| Version | Milestone | Status |
|---------|-----------|--------|
| **0.20.3** | Core gateway + providers | Done |
| **0.21.5** | Auth, observability, messaging | Done |
| **0.23.3** | Tool use, context management, privacy routing | Next |
| **0.24.0** | Speech & audio improvements | Planned |
| **1.0.0** | Stable API, 90%+ coverage | Target |

Full details: [docs/development/roadmap.md](docs/development/roadmap.md)

---

## Building from source

```bash
git clone https://github.com/MacCracken/hoosh.git
cd hoosh

# Build (all default providers, no whisper)
cargo build

# Build with whisper support (requires whisper.cpp system lib)
cargo build --features whisper

# Run tests
cargo test

# Run all CI checks locally
make check
```

---

## Versioning

Pre-1.0 releases use `0.D.M` (day.month) SemVer — e.g. `0.20.3` = March 20th.
Post-1.0 follows standard SemVer.

The `VERSION` file is the single source of truth. Use `./scripts/version-bump.sh <version>` to update.

---

## License

AGPL-3.0-only. See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork and create a feature branch
2. Run `make check` (fmt + clippy + test + audit)
3. Open a PR against `main`
