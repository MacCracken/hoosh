# Changelog

All notable changes to hoosh are documented here.

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
- `BACKLOG.md` — engineering backlog (cleared)
- Feature flags: ollama, llamacpp, synapse, lmstudio, localai, openai, anthropic, deepseek, mistral, groq, openrouter, grok, whisper, piper, hwaccel
