# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-04-13

Complete rewrite from Rust to Cyrius. Binary drops from multi-MB to 636KB. All core gateway functionality preserved and ported.

### Added — Core Gateway
- **18 Cyrius modules** — types, ratelimit, route, router, budget, cache, metrics, auth, http_server, http_client, provider, compact, audit, retry, hardware, handlers, config, main
- **13 provider backends** — Ollama (native `/api/chat`), LlamaCPP, Synapse, LMStudio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter — all via OpenAI-compatible forwarding (Ollama uses native API)
- **SSE streaming** — `stream:true` in `/v1/chat/completions` proxies NDJSON (Ollama) or SSE (OpenAI-compat) from backend to client as OpenAI-format `chat.completion.chunk` events
- **Provider routing** — Priority, RoundRobin, LowestLatency strategies; model pattern matching with glob (`llama*`, `gpt-*`)
- **Token budget system** — named pools with capacity, reserve/commit lifecycle; `/v1/tokens/check`, `/v1/tokens/reserve`, `/v1/tokens/report`, `/v1/tokens/pools`
- **HMAC-SHA256 audit chain** — cryptographically linked log entries with tamper detection and verification; `/v1/audit` endpoint with chain validation
- **Retry with exponential backoff** — jittered delays (nanosecond clock bits for jitter), configurable max_retries/base_delay_ms/max_delay_ms via `[[retry]]` config section
- **Per-provider rate limiting** — RPM token bucket with continuous refill; `rate_limit` field in `[[providers]]` config
- **Response cache with LRU eviction** — timestamp-based access tracking, evict-oldest-on-full; hit/miss/eviction counters at `/v1/cache/stats`
- **Context compaction** — preserves system message, keeps recent N messages within token budget; runs before inference to prevent oversized requests
- **Bearer token auth** — constant-time comparison via sigil; skips `/v1/health` and `/metrics`
- **CORS** — full preflight handling on all endpoints

### Added — Hardware
- **ai-hwaccel 2.0.0 integration** — git tag dep (kybernet-style), 27 modules for hardware detection across 18 accelerator types (CUDA, ROCm, Metal, Vulkan, TPU, Gaudi, Neuron, Intel NPU, AMD XDNA, etc.)
- **`/v1/hardware`** — device summary JSON (count, memory, best device, all profiles)
- **`/v1/hardware/placement`** — model placement recommendation given model_params and quantization
- **`/v1/hardware/models`** — compatibility matrix for common model sizes (1B–405B) against detected hardware
- **Hardware on startup** — device count and best device shown in server banner and `hoosh info`

### Added — API Endpoints
- `POST /v1/chat/completions` — streaming + non-streaming inference
- `GET /v1/models` — list configured providers
- `GET /v1/health` — first provider connectivity check
- `GET /v1/health/providers` — per-provider health with TCP probe
- `GET /v1/health/heartbeat` — node status
- `POST /v1/embeddings` — routed through provider system (not hardcoded)
- `GET /v1/costs` — request/token counters per provider
- `POST /v1/costs/reset` — reset counters
- `GET /v1/cache/stats` — hit/miss/eviction stats
- `GET /v1/tokens/pools` — pool capacity/usage
- `GET /v1/queue/status` — queue depth
- `GET /v1/audit` — audit chain with verification
- `POST /v1/admin/reload` — hot-reload config
- `GET /v1/hardware`, `POST /v1/hardware/placement`, `GET /v1/hardware/models`
- `GET /metrics` — Prometheus format
- `GET /` — server info

### Added — CLI
- `hoosh serve [port]` — start gateway (default: 8088)
- `hoosh models` — list configured providers with URLs
- `hoosh health` — check provider connectivity
- `hoosh infer <model> <prompt>` — one-shot inference from CLI
- `hoosh info` — system info with hardware summary
- `hoosh help` / `hoosh version`

### Added — Configuration
- `hoosh.toml` with sections: `[[server]]`, `[[providers]]` (type, base_url, priority, models, api_key, rate_limit), `[[budgets]]`, `[[auth]]`, `[[retry]]`, `[[cache]]`
- `cyrius.toml` with `[package]`, `[build]`, `[deps]` (stdlib + ai-hwaccel git tag dep)

### Changed
- **Language**: Rust → Cyrius (cyrius 3.10.0)
- **Binary size**: multi-MB → 636KB
- **Dependencies**: 200+ crates → 29 Cyrius deps (stdlib + ai-hwaccel)
- **HTTP server**: axum/tokio → raw TCP sockets with syscalls
- **Build system**: cargo → `cyrius build`
- **Dep management**: Cargo.toml → cyrius.toml with git tag deps (kybernet-style)

### Removed
- Rust codebase (preserved in `rust-old/` for reference)
- axum, tokio, reqwest, serde, and all Rust dependencies
- Feature flags (all features compiled in)
- OpenTelemetry integration (deferred to v2.1)
- DLP content filtering (deferred to v2.1)
- TLS/mTLS support (blocked on Cyrius TLS lib)
- Audio endpoints (deferred to svara migration)
- Tool calling / MCP bridge (deferred to v2.1)
- Multi-threaded concurrency (single-threaded accept loop)

---

## Rust-era releases (pre-Cyrius port)

See `rust-old/` for source. These versions used Rust + axum + tokio.

- **1.2.0** (2026-04-03) — License change to GPL-3.0, binary size optimization, TLS provider decoupling
- **1.1.0** (2026-03-29) — GPU telemetry heartbeats, heartbeat eviction, majra ConcurrentPriorityQueue
- **1.0.0** (2026-03-27) — Context management, model metadata (63 models), semantic cache, retry manager, batch inference, cost optimizer, DLP scanner, multi-modal support, ai-hwaccel 1.0.0, 613 tests
- **0.23.4** (2026-03-23) — Tool use & MCP via bote/szál, model metadata registry, hot_path benchmarks
- **0.23.3** (2026-03-23) — Sentiment analysis via bhava
- **0.21.5** (2026-03-21) — Auth, rate limiting, TLS pinning, Prometheus, OpenTelemetry, audit chain, health checks, heartbeat, event bus, queue
- **0.21.3** (2026-03-21) — E2E benchmarks, connection tuning, HTTP/2, documentation
- **0.20.4** (2026-03-21) — Benchmark suite, CI/CD pipelines, version management
- **0.20.3** (2026-03-20) — Initial release: 14 backends, routing, caching, budgets, streaming, hardware placement, CLI, 185 tests
