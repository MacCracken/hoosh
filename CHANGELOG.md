# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [0.21.3] — 2026-03-21

### Added
- **End-to-end benchmark suite** (`benches/e2e.rs`) — 8 benchmark groups measuring full round-trip: HooshClient → hoosh server → Ollama → response
  - `e2e_health_check`, `e2e_list_models`, `e2e_infer` (short/medium)
  - `e2e_connection` — cold (new client) vs warm (pooled) connection comparison
  - `e2e_concurrent` — 1/4/8/16 parallel requests through shared client
  - `e2e_gateway_overhead` — direct Ollama vs through hoosh (isolates gateway cost)
  - `e2e_stream` — streaming inference through hoosh SSE proxy
  - `e2e_sequential` — 3 back-to-back requests (simulates agent loop)
- **Connection tuning tests** — 5 new tests for pooled connection reuse, concurrent requests, and tuned provider creation
- **Documentation infrastructure**
  - `docs/development/performance.md` — dedicated performance doc with all benchmark results, connection tuning table, hot-path analysis, and update instructions
  - `docs/index.md` — documentation portal linking all docs
  - `docs/decisions/001-http-gateway.md` — ADR explaining HTTP gateway design choice
  - `CONTRIBUTING.md` — development workflow, code style, commit conventions, provider guide
- **Makefile targets** — `bench`, `vet`, `coverage`, `release` (matching agnosai/dhvani/ranga)
- **CI benchmark job** — runs synthetic benchmarks on every push/PR, uploads criterion artifacts (30-day retention)

### Changed
- **HTTP/2 support** — added `http2` feature to reqwest for multiplexed connections
- **Connection pooling tuned on all HTTP clients** (HooshClient, OllamaProvider, OpenAiCompatibleProvider, AnthropicProvider, TtsProvider):
  - `TCP_NODELAY` — disables Nagle's algorithm, eliminates up to 40ms batching delay per request
  - `tcp_keepalive(60s)` — OS-level keepalive probes prevent connection drops
  - `pool_idle_timeout(600s)` — keep connections alive 10 min (was reqwest default 90s)
  - `pool_max_idle_per_host(32)` — allow more concurrent pooled connections
  - `http2_adaptive_window(true)` — adaptive flow control for multiplexed requests
- **HooshClient** now uses a tuned `reqwest::Client::builder()` instead of bare `reqwest::Client::new()`
- **BENCHMARKS.md** — added e2e results, connection reuse data, concurrency scaling, gateway overhead measurements
- **Makefile** — added `bench`, `vet`, `coverage`, `release`, `clean` targets; `check` now includes `deny`

### Performance
- Gateway overhead through hoosh: **~0 ms** (within measurement noise of direct Ollama)
- Warm connection reuse: **52 µs** (2.9x faster than cold 151 µs)
- 16 concurrent health checks: **306 µs total** (19 µs per request)
- Sequential 3-request agent loop: **1.32 s** (48% improvement over first run baseline)
- Streaming through hoosh: **1.46 s** for 50 tokens (23% improvement)

## [0.20.4] — 2026-03-21

### Added
- Benchmark suite with Criterion: routing, providers, live_providers
- `BENCHMARKS.md` — performance matrix with hardware specs and results
- `scripts/version-bump.sh` — version management script
- `VERSION` file — single source of truth for version
- `.github/workflows/ci.yml` — CI pipeline (check, security, deny, test, msrv, coverage)
- `.github/workflows/release.yml` — tag-triggered release (cross-compile, crates.io publish, GitHub release)

### Changed
- Bug fixes and stability improvements

## [0.20.3] — 2026-03-20

### Core Gateway
- `InferenceRequest` / `InferenceResponse` with OpenAI-compatible serialization
- `Message`, `Role`, `TokenUsage` for multi-turn conversations
- `ModelInfo` with parameter count, context length, provider
- `LlmProvider` trait (infer, infer_stream, list_models, health_check)
- `ProviderType` enum (14 backends) with `is_local()` / `supports_streaming()`
- `Router` with Priority, RoundRobin, LowestLatency, Direct strategies
- Model pattern matching (glob: `llama*`, `gpt-*`)
- `ResponseCache` with TTL, DashMap, forced eviction, `Arc<String>` values, `cache_key()` helper
- `TokenBudget` with named pools, reserve/commit/release lifecycle, saturating arithmetic

### Local Provider Backends
- **Ollama** — native `/api/chat` + `/api/tags`, NDJSON streaming, pull/delete, max_tokens/top_p support
- **llama.cpp** — OpenAI-compatible wrapper (default port 8080)
- **Synapse** — OpenAI-compatible wrapper with training_status/sync_catalog
- **LM Studio** — OpenAI-compatible wrapper (default port 1234)
- **LocalAI** — OpenAI-compatible wrapper (default port 8080)
- **OpenAI-compatible base** (`openai_compat.rs`) — shared by 9 providers

### Remote Provider Backends
- **OpenAI** — OpenAI-compatible with bearer auth
- **Anthropic** — native Messages API, system prompt extraction, configurable API version (`ANTHROPIC_API_VERSION`)
- **DeepSeek** — OpenAI-compatible
- **Mistral** — OpenAI-compatible
- **Groq** — OpenAI-compatible
- **OpenRouter** — OpenAI-compatible
- **Grok (xAI)** — OpenAI-compatible

### Provider Registry
- `ProviderRegistry` with `register_from_route()` and `get(type, base_url)`
- Feature-gated provider construction for all 12 backends
- API key passthrough from config to remote providers

### Hardware-Aware Placement
- `HardwareManager` using ai-hwaccel with vulkan, rocm, cuda backends
- `recommend_placement()` for model-to-provider assignment
- `summary()` for `hoosh info` display

### Configuration
- `hoosh.toml` config file with auto-load
- `[server]` — bind, port, strategy
- `[cache]` — max_entries, ttl_secs, enabled
- `[[providers]]` — type, base_url (auto-defaults), api_key (`$ENV_VAR` resolution with warnings), priority, models, max_tokens_limit
- `[[budgets]]` — named token pools with capacity
- `[whisper]` — model path
- `[tts]` — backend URL
- CLI `--port`/`--bind`/`--config` overrides

### Server
- axum HTTP server with OpenAI-compatible API
- `POST /v1/chat/completions` — streaming SSE + non-streaming JSON
- `GET /v1/models` — live provider queries with pattern fallback
- `GET /v1/health` + `GET /v1/health/providers` — live health checks
- `POST /v1/tokens/check` / `reserve` / `report`, `GET /v1/tokens/pools`
- `POST /v1/audio/transcriptions` — whisper STT (feature-gated)
- `POST /v1/audio/speech` — TTS via HTTP backend (feature-gated)
- Token budget integration: reserve → infer → report, pool validation, 429 on exceed
- `StreamBudgetGuard` drop guard for budget reporting on client disconnect
- Input validation: model, messages (max 256), temperature (0-2), top_p (0-1)
- Per-provider max_tokens clamping
- Per-route body limits: 1MB JSON API, 50MB audio
- Request ID logging for tracing
- Content-Type validation on audio endpoints

### Client
- `HooshClient` with proper OpenAI-format requests
- `infer()` — non-streaming chat completions
- `infer_stream()` — SSE streaming with token-by-token delivery
- Content-Type validation on SSE responses

### CLI
- `hoosh serve` — with config file loading
- `hoosh models` / `hoosh health` / `hoosh info`
- `hoosh infer --model llama3 "prompt"` — non-streaming
- `hoosh infer --model llama3 "prompt" --stream` — streaming with live output
- `hoosh transcribe <file>` — speech-to-text
- `hoosh speak "text" --output file.wav` — text-to-speech

### Speech
- **Whisper STT** — whisper-rs 0.16, WAV decoder with sample rate parsing, async via spawn_blocking
- **TTS** — HTTP backend provider (works with openedai-speech, OpenAI API), `SpeechRequest`/`SpeechResponse` types

### Security & Hardening (3 audit rounds)
- 50MB default body limit, 1MB for JSON routes
- 10M token default budget (not unlimited)
- Pool existence validation before budget reservation
- Mutex poisoning recovery (`unwrap_or_else(|e| e.into_inner())`)
- `StreamBudgetGuard` with `try_lock()` + async fallback
- 1MB SSE line buffer limit (all 4 stream parsers)
- WAV chunk position overflow protection (`checked_add`)
- Odd-length audio data rejection
- Request timeouts: 300s overall, 10s connect (all providers)
- Cache forced eviction when expired eviction insufficient
- Budget `available()` using `saturating_add` to prevent overflow
- Budget `commit()` using `saturating_add`
- API key redaction in config parse errors
- Missing `$ENV_VAR` warnings at config load time
- Temperature/top_p range validation
- Content-Type validation on SSE and audio endpoints

### Testing
- 185 tests (182 default + 3 piper-gated), 4 live Ollama tests
- E2E server tests with mock backends (health, models, infer, streaming, budget enforcement, validation)
- Mock servers for OpenAI-compatible, Ollama, and Anthropic APIs
- Unit tests for WAV decoding, message building, cache operations, config parsing, provider construction
- 66% total coverage, 74% lib-only coverage

### Benchmarks
- Synthetic: routing (199ns select), registry (36ns lookup), serialization (758ns 10-msg)
- Hardware: detect (31ms with GPU probing), placement (47ns)
- Live Ollama: health (223µs), infer short (413ms), medium (2.47s), streaming (1.39s)
- CPU vs Vulkan iGPU comparison documented

### Infrastructure
- `deny.toml` — cargo-deny for license/advisory/ban checks
- `hoosh.toml` — example config with Ollama + budget pools
- `BENCHMARKS.md` — performance matrix with hardware specs
- Feature flags: ollama, llamacpp, synapse, lmstudio, localai, openai, anthropic, deepseek, mistral, groq, openrouter, grok, whisper, piper, hwaccel
