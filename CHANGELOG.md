# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [1.2.0] — 2026-04-03

### Changed
- **License changed from AGPL-3.0-only to GPL-3.0-only** — updated `Cargo.toml`, `deny.toml`, `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`; added `LICENSE` file with full GPL-3.0 text
- **Binary size optimization** — added `[profile.release]` with `strip = true`, `lto = true`, `codegen-units = 1`, `opt-level = "s"`, `panic = "abort"`
- **TLS provider decoupling** — switched reqwest from `rustls` feature to `rustls-no-provider`, added explicit `rustls` 0.23 dependency with `ring` crypto provider; `install_crypto_provider()` initializes ring at startup
- All tests updated with explicit `install_crypto_provider()` calls for deterministic TLS initialization

## [1.1.0] — 2026-03-29

### Added
- **GPU telemetry on heartbeats** — local providers now forward GPU memory capacity from ai-hwaccel into majra's heartbeat tracker via `heartbeat_with_telemetry()` (feature-gated: `hwaccel`)
- **Heartbeat eviction policy** — persistently offline providers (5 consecutive offline sweep cycles) are automatically evicted from the heartbeat tracker and removed from the health map; eviction events are published to the event bus and logged
- **`dequeue_wait()` on `InferenceQueue`** — async blocking dequeue that parks until an item is enqueued, powered by majra's built-in `Notify`

### Changed
- **`InferenceQueue` → majra `ConcurrentPriorityQueue`** — replaced `std::sync::Mutex<PriorityQueue>` wrapper with majra's async-native `ConcurrentPriorityQueue` (tokio mutex + notify); `enqueue`, `dequeue`, `len`, `is_empty` are now `async fn`
- Dependency updates:
  - `majra` 0.21.3 → 1.0.2
  - `aws-lc-sys` 0.39.0 → 0.39.1
  - `cc` 1.2.57 → 1.2.58
  - `cmake` 0.1.57 → 0.1.58
  - `js-sys` 0.3.91 → 0.3.92
  - `mio` 1.1.1 → 1.2.0
  - `rustc-hash` 2.1.1 → 2.1.2
  - `uuid` 1.22.0 → 1.23.0
  - `wasm-bindgen` 0.2.114 → 0.2.115
  - `web-sys` 0.3.91 → 0.3.92
  - `zerocopy` 0.8.47 → 0.8.48

## [1.0.0] — 2026-03-27

### Added
- **Context management** (`context` module)
  - `TokenCounter` trait with `SimpleTokenCounter` (bytes/4 heuristic) and `ProviderTokenCounter` (per-provider ratios for OpenAI, Anthropic, etc.)
  - `ContextCompactor` — proactive context window management, truncates conversations at configurable threshold (default 80%), preserves system prompts + last N messages, binary search for optimal keep count
  - `compress_messages()` — mechanical prompt compression: whitespace collapse, stale tool-call pair pruning (keeps last 3 tool interactions)
  - `[context]` config section: `compaction_threshold`, `keep_last_messages`, `enabled`
  - Handler integration: token-counted budget estimation replaces hardcoded `1024`, compaction runs before inference
- **Model metadata registry expansion** — 21 → 63 models
  - `ModelTier` enum (Economy/Standard/Premium/Reasoning) with `#[non_exhaustive]`
  - `Modality` enum (Text/Vision/Audio/Embedding) with `#[non_exhaustive]`
  - `max_output_tokens`, `supports_system_prompt` fields on `ModelMetadata`
  - `by_tier()`, `by_modality()`, `len()`, `is_empty()` query methods
  - New models: GPT-4.1 family, o3/o4-mini, Gemini 2.5, Llama 4, Mistral Nemo/Codestral/Pixtral, Phi-3/4, Gemma 2/3, Command-R, embedding models (OpenAI, nomic, mxbai)
- **Cache improvements**
  - `CacheStats` with atomic hit/miss/eviction counters, `hit_rate` calculation
  - `GET /v1/cache/stats` endpoint
  - `SemanticCache` — cosine similarity lookup over stored embeddings, configurable threshold (default 0.92), full-scan with optional max_search cap
  - `CacheWarmer` — pre-populate cache on startup from `[[cache.warming_prompts]]` config
- **Provider acceleration**
  - `RetryManager` — jittered exponential backoff, `ErrorClass` (Retryable vs Permanent), configurable max_retries/base_delay/max_delay/jitter; integrated in non-streaming handler path
  - `HooshError::is_retryable()` — classifies 429/5xx/timeouts as retryable, 400/401/404/budget as permanent
  - `[retry]` config section: `max_retries`, `base_delay_ms`, `max_delay_ms`
  - `BatchManager` — concurrent inference with `Semaphore` concurrency control, `CancellationToken`, per-batch progress tracking, TTL-based eviction of completed batches
  - `CostOptimizer` — recommends cheapest capable model given request complexity (token count, tools, vision), tier classification, capability matching
  - `RequestProfile` and `ModelRecommendation` types
- **Privacy & DLP** (`dlp` feature flag, requires `regex`)
  - `DlpScanner` with `RegexSet` single-pass matching, 8 built-in PII patterns: email, phone (US, separator-required), SSN, credit card, IPv4, API keys, AWS access keys, GitHub tokens
  - `ClassificationLevel` enum: Public/Internal/Confidential/Restricted with `#[non_exhaustive]`
  - Custom patterns via `[[dlp.patterns]]` config with per-pattern classification
  - `DlpConfig` with enable/disable and default classification level
  - Privacy-aware routing: `Router::select_with_classification()` — Confidential routes to local-only providers, Restricted blocks entirely
  - `HooshError::DlpBlocked` variant (HTTP 403, `content_blocked` error code)
- **Multi-modal support**
  - `MessageContent` enum: `Text(String)` | `Parts(Vec<ContentPart>)` — serde-compatible with OpenAI format (deserializes from plain string or content array)
  - `ContentPart::Text` and `ContentPart::ImageUrl` with `#[non_exhaustive]`
  - `ImageUrl` struct with optional detail level
  - `MessageContent::text()` (returns `Cow<str>`), `has_images()`, `PartialEq<&str>` for ergonomic use
  - `ChatMessage.content` changed from `String` to `MessageContent` — vision requests pass through to providers
- **Hardware capabilities** (ai-hwaccel 1.0.0)
  - `detect_with_timing()` — per-backend probe timing for startup diagnostics
  - `from_cache(ttl)` — disk-cached detection via `DiskCachedRegistry`, skips re-probing on restart
  - `best_device()` — ranked device selection by memory × throughput
  - `devices_by_family()`, `gpus()`, `npus()`, `tpus()` — filter by accelerator family
  - `plan_sharding(model_params)` — multi-GPU model splitting plans (pipeline/tensor/data parallel)
  - `system_io()`, `has_fast_interconnect()`, `estimate_data_load_secs()` — I/O topology and throughput estimation
  - `ShardingSummary` and `ShardInfo` output types
  - Hardware summary now shows device names (e.g. "RTX 4090") and interconnect info
  - `hoosh info` uses timed detection, shows best device and interconnect bandwidth

### Changed
- Bump ai-hwaccel from 0.23.3 to 1.0.0 (path dep, pre-publish)
- `Message.content` type changed from `String` to `MessageContent` (backwards-compatible deserialization)
- `Role`, `RoutingStrategy`, `EmbeddingsInput` enums now `#[non_exhaustive]`
- `AppState` expanded: `compactor`, `model_registry`, `retry_manager` fields
- `ServerConfig` expanded: `context_config`, `retry_config` fields
- Budget estimation in handler uses `ProviderTokenCounter` (input tokens + output budget) instead of hardcoded `max_tokens.unwrap_or(1024)`
- Cost module restructured from `cost.rs` to `cost/mod.rs` + `cost/optimizer.rs`
- `lookup_pricing()` visibility changed from `fn` to `pub(crate) fn` for optimizer access

### Fixed
- `#[derive(Debug)]` added to `provider::whisper::DecodedAudio` (clippy fix)
- DLP scanner uses `map_err` instead of `expect()` for regex compilation (no panic in lib code)
- Cache warming uses locally-computed key for insertion (was using mismatched key from closure)
- Token counter documents bytes-per-token semantics (intentionally over-estimates for multi-byte UTF-8)
- Compactor truncation uses binary search O(log n) instead of linear decrement O(n)
- Batch manager `evict_completed(max_age)` prevents unbounded memory growth from finished batches
- Prompt compression uses `retain()` O(n) instead of `remove()` O(n²), `HashSet` for stale tool ID lookups
- Phone pattern requires separators to reduce false positives on numeric strings
- API key DLP pattern quantifier bounded to `{1,5}` (prevents ReDoS)
- Retry manager defaults unknown errors to non-retryable (was retrying permanently broken requests)
- Semantic cache redundant key clone removed

### Security
- DLP scanner rejects invalid custom regex patterns at construction time
- Privacy-aware routing enforces local-only inference for Confidential content
- Restricted classification blocks inference entirely (HTTP 403)
- ReDoS mitigation: bounded quantifiers on all DLP regex patterns

### CI/CD
- `cargo-semver-checks` job in CI pipeline — runs on PRs to detect accidental API breakage
- `make semver` Makefile target

### Documentation
- Runnable doc examples on `InferenceRequest`, `MessageContent`, `TokenPool`, `HooshError`, `ResponseCache` (6 doc-tests)

### Testing
- 613 tests (up from 388) + 6 doc-tests, all passing
- 82.8% line coverage (88% excluding untestable hardware-gated code)
- **OpenAI conformance suite** — 12 strict schema validation tests: response fields, choices/usage schema, `chatcmpl-` ID prefix, unix timestamp, content-type, error format, multi-part content, models/health/cache endpoints
- New unit test coverage: token counting (11), context compaction (8), prompt compression (5), cache stats (4), semantic cache (7), cache warming (6), retry manager (12), batch inference (13), cost optimizer (14), DLP scanner (13), hardware capabilities (6), multi-modal content (7), config (30), router DLP (6), provider TLS (6), client (10), handler conformance (25)
- Full audit round: 3 critical, 7 high, 12 medium, 12 low findings — all critical/high fixed

### Dependencies
- `regex` 1.x (optional, behind `dlp` feature)
- `tokio-util` 0.7 (for `CancellationToken` in batch inference)

## [0.23.4] — 2026-03-23

### Added
- **Tool use & function calling** (`tools` feature flag)
  - `ToolDefinition`, `ToolCall`, `ToolResult`, `ToolChoice` — unified tool types across providers
  - `tools::convert` — OpenAI/Anthropic format conversion and response parsing
  - `tools::McpBridge` — MCP integration via bote dispatcher + szal's 47 built-in tools
  - `POST /v1/tools/list` — list all registered MCP tools
  - `POST /v1/tools/call` — invoke MCP tools by name with arguments
  - `Role::Tool` variant for tool-result messages in multi-turn conversations
  - `tools`/`tool_choice` fields on `InferenceRequest`, `tool_calls` on `InferenceResponse`
  - `Message::new()` constructor for ergonomic message creation
  - Streaming tool call assembly — accumulates incremental OpenAI SSE deltas into complete ToolCalls
  - `szal_workflow_run` MCP tool — invoke szál workflow engine (sequential/parallel/DAG) as a single tool call
- **Error handling** — `HooshError::http_status_code()` and `error_code()` for consistent OpenAI-compatible error responses
- **Model metadata registry** — `provider::metadata::ModelMetadataRegistry` with 20+ models, context windows, capability flags (chat/streaming/tool_use/vision/embeddings), prefix matching
- `hot_path` benchmark suite — auth, rate limiting, cost tracking, audit chain, event bus, health-aware routing, queue operations (22 benchmarks)
- bote 0.22.3 and szal 0.23.4 as optional dependencies

### Changed
- **Server refactor** — split `server.rs` (1477 lines) into `server/` module: `mod.rs` (331), `types.rs` (263), `handlers.rs` (831), `audio.rs` (136)
- Bump ai-hwaccel from 0.21.3 to 0.23.3
- Remove local path dependency on bhava (now resolves from crates.io)

## [0.23.3] — 2026-03-23

### Added
- `sentiment` feature flag — optional bhava integration for response sentiment analysis
- `SentimentAnalysis` struct on `InferenceResponse` — valence, confidence, is_positive, is_negative (feature-gated)
- `analyze_response_sentiment()` — analyze response text for sentiment using bhava's keyword engine
- bhava 0.23.3 as optional dependency (sentiment analysis with negation, intensity modifiers)

## [0.21.5] — 2026-03-21

### Added
- **Authentication & security**
  - Bearer token auth middleware (`[auth] tokens` in config)
  - Per-provider rate limiting with sliding window RPM (`rate_limit_rpm` per provider)
  - TLS certificate pinning for remote providers (`tls_pinned_certs`)
  - mTLS client certificate support for local providers (`client_cert`, `client_key`)
- **Observability**
  - Prometheus metrics endpoint (`GET /metrics`) with request counters, latency histograms, token counters, provider gauges
  - OpenTelemetry trace propagation (feature-gated: `otel`) with OTLP export
  - Per-provider cost tracking with static pricing table (`GET /v1/costs`, `POST /v1/costs/reset`)
  - Cryptographic audit log — HMAC-SHA256 linked chain with tamper detection (`GET /v1/audit`)
- **Server improvements**
  - `/v1/embeddings` pass-through (OpenAI-compatible and Ollama native)
  - Hot-reload config via SIGHUP or `POST /v1/admin/reload` (uses `arc-swap` + `RwLock`)
  - Periodic background health checks with automatic failover (3-strike unhealthy marking)
  - Latency tracking for `LowestLatency` routing strategy (exponential moving average)
  - Priority request queue via `majra` (`GET /v1/queue/status`)
  - Provider event bus via `majra::pubsub` (health changes, inference events, errors)
  - Provider heartbeat tracking via `majra::heartbeat` with Online→Suspect→Offline FSM (`GET /v1/health/heartbeat`)
- **New modules**: `audit`, `cost`, `events`, `health`, `metrics`, `middleware/auth`, `middleware/rate_limit`, `queue`, `telemetry`
- **New dependencies**: `hmac`, `sha2`, `hex`, `rand`, `prometheus`, `arc-swap`, `majra`; optional: `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`

### Changed
- Bump `ai-hwaccel` dependency from 0.20.3 to 0.21.3
- All provider constructors now accept optional `TlsConfig` and use shared `build_provider_client()` utility
- `Router` uses `RwLock` for hot-reload support; `LowestLatency` uses O(n) min scan instead of sort
- `StrategyValue` → `RoutingStrategy` conversion uses `From` impl (eliminates duplicate match blocks)
- `AppState` fields: added `cost_tracker`, `audit`, `auth_token_digests`, `rate_limiter`, `event_bus`, `inference_queue`, `health_map`, `heartbeat`, `config_path`
- `ServerConfig` extended with audit, auth, telemetry, health check configuration
- `build_app()` returns `(Router, Arc<AppState>)` for SIGHUP handler access

### Fixed
- Audit chain eviction now uses `VecDeque::pop_front()` (O(1)) instead of `Vec::remove(0)` (O(n))
- Audit verification works correctly after eviction via `first_valid_hash` tracking
- Audit `/v1/audit` endpoint uses single-lock `snapshot()` instead of 3 separate lock acquisitions
- Health checker failure handling deduplicated into `handle_check_failure()` helper

### Security
- Auth token comparison uses constant-time SHA-256 digest comparison (no length leak)
- Pre-hash auth tokens at startup — 1 hash per request instead of 2N
- Config `Debug` impls redact `api_key`, `signing_key`, and auth `tokens`
- TLS cert pinning failure logs error instead of silently degrading
- Inference error messages sanitized — internal details logged, generic message to clients
- Health endpoint no longer exposes raw provider errors (internal IPs/hostnames)
- Model name validation rejects control characters and limits length to 256
- Audit signing key supports `$ENV_VAR` resolution
- Cost reset records an audit entry
- Startup warning when authentication is disabled
- Unrecognized provider type logs warning during registration
- `prometheus` upgraded 0.13 → 0.14 (fixes RUSTSEC-2024-0437 in transitive `protobuf` dependency)

### Performance
- SSE streaming uses reusable `String` buffer with `write!` instead of `serde_json::json!` per token
- Audit chain crypto (SHA-256, HMAC) moved outside mutex — lock only held for append
- Audit `generate_id()` uses `uuid::Uuid::new_v4()` (consistent with codebase)
- Router `LowestLatency` uses O(n) min scan instead of O(n log n) sort
- `StrategyValue` → `RoutingStrategy` uses `From` impl (eliminates duplicate match blocks)
- `CostTracker::all_with_total()` single-pass method replaces two iterations
- `StreamBudgetGuard` now records metrics and events on stream end (previously missed)
- Removed dead `ProviderType` import hack in events.rs

### Testing
- 328 tests (up from 187), 78.9% line coverage
- E2E integration tests: streaming SSE, embeddings pass-through, auth enforcement, health failover, Ollama native flow, full observability verification
- Mock servers: Ollama (chat, tags, embed, streaming), Synapse (infer, list, health, training, catalog)
- Unit tests: audit chain (100%), health checker (81%), router (98%), queue (83%), auth (100%), rate limiter (87%), cost tracker (90%), metrics (98%)

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
