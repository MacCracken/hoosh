# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v0.20.3 — Core Gateway + Local Provider Backends

Foundation: inference types, provider trait, routing, caching, token budgets, HTTP client, axum server, and local provider backends.

### Core types
- [x] `InferenceRequest` / `InferenceResponse` with OpenAI-compatible serialisation
- [x] `Message`, `Role`, `TokenUsage` for multi-turn conversations
- [x] `ModelInfo` with parameter count, context length, provider
- [x] `TranscriptionRequest` / `TranscriptionResponse` for speech-to-text

### Provider framework
- [x] `LlmProvider` trait (infer, infer_stream, list_models, health_check)
- [x] `ProviderType` enum (14 backends)
- [x] `is_local()` / `supports_streaming()` capability queries

### Routing
- [x] `Router` with Priority, RoundRobin, LowestLatency, Direct strategies
- [x] Model pattern matching (glob: `llama*`, `gpt-*`)
- [x] Disabled provider filtering

### Middleware
- [x] `ResponseCache` with TTL, DashMap, max entries
- [x] `TokenBudget` with named pools, reserve/commit/release lifecycle

### Client
- [x] `HooshClient` HTTP client for downstream consumers
- [x] `/v1/chat/completions`, `/v1/models`, `/v1/health`

### Server
- [x] axum HTTP server with OpenAI-compatible endpoints
- [x] `/v1/chat/completions` (streaming SSE + non-streaming JSON)
- [x] `/v1/models`, `/v1/health`, `/v1/health/providers`
- [x] `/v1/tokens/check`, `/v1/tokens/reserve`, `/v1/tokens/report`, `/v1/tokens/pools`
- [x] CORS, graceful shutdown

### CLI
- [x] `hoosh serve --port 8088`
- [x] `hoosh models` / `hoosh health` / `hoosh info`
- [x] `hoosh infer --model llama3 "prompt"`

### Ollama
- [ ] REST API client (`/api/chat`, `/api/generate`)
- [ ] Model pull / list / delete
- [ ] Streaming support via chunked response

### llama.cpp
- [ ] Server API client (OpenAI-compatible `/v1/chat/completions`)
- [ ] Model loading, context management
- [ ] Streaming via SSE

### Synapse
- [ ] Synapse API client (OpenAI-compatible)
- [ ] Training job status integration
- [ ] Model catalog sync

### LM Studio / LocalAI
- [ ] OpenAI-compatible API clients (reuse openai provider with custom base_url)

### Hardware-aware placement
- [ ] ai-hwaccel integration for detecting available GPUs/TPUs
- [ ] Automatic model-to-device assignment based on VRAM
- [ ] Quantisation recommendation per hardware

---

## v0.21.3 — Remote Provider Backends

### OpenAI
- [ ] Chat completions (streaming + non-streaming)
- [ ] Embeddings API
- [ ] Token counting (tiktoken-compatible)

### Anthropic
- [ ] Messages API with streaming
- [ ] System prompt handling
- [ ] Tool use / function calling

### DeepSeek / Mistral / Groq / Google / Grok / OpenRouter
- [ ] OpenAI-compatible API clients with provider-specific quirks
- [ ] Rate limiting per provider
- [ ] API key management (env vars + config file)

### Provider health monitoring
- [ ] Periodic health checks on all configured providers
- [ ] Automatic failover when provider goes unhealthy
- [ ] Latency tracking for LowestLatency routing

---

## v0.22.3 — Speech-to-Text (Whisper)

### whisper.cpp integration
- [ ] Rust bindings via `whisper-rs` (feature-gated)
- [ ] Model download and management (tiny, base, small, medium, large)
- [ ] Audio format support: WAV, MP3, OGG, FLAC
- [ ] Word-level timestamps

### Transcription API
- [ ] `POST /v1/transcribe` endpoint
- [ ] Language detection
- [ ] Streaming transcription (chunked audio input)

### tarang integration
- [ ] Tarang's `HooshClient` calls `/v1/transcribe` for audio analysis
- [ ] Batch transcription for media libraries

---

## v0.23.3 — Configuration & Server Hardening

### Configuration
- [ ] TOML config file for provider routes, budgets, cache settings
- [ ] Hot-reload config without restart
- [ ] Environment variable overrides

### Server
- [ ] `/v1/embeddings` pass-through
- [ ] Bearer token auth middleware
- [ ] Rate limiting middleware
- [ ] Per-agent budget enforcement in request pipeline

---

## v0.24.3 — Advanced Features

### Caching improvements
- [ ] Semantic cache (embedding similarity, not just exact match)
- [ ] Cache warming for common prompts
- [ ] Cache statistics API

### Accounting & observability
- [ ] Per-provider cost tracking (token × price)
- [ ] Prometheus metrics endpoint
- [ ] OpenTelemetry trace propagation
- [ ] Request/response audit log

### Provider acceleration
- [ ] Batch inference (group compatible requests)
- [ ] Speculative decoding hints
- [ ] Prompt compression for long conversations

### Security
- [ ] TLS certificate pinning for remote providers
- [ ] mTLS for local provider communication

---

## v1.0.0 Criteria

All of the following must be true before cutting 1.0:

- [ ] Public API reviewed and marked stable
- [ ] `LlmProvider` trait finalized
- [ ] Core types (`InferenceRequest`, `InferenceResponse`, `ModelInfo`) frozen
- [ ] 90%+ line coverage
- [ ] At least 5 provider backends fully implemented and tested
- [ ] OpenAI-compatible API passing conformance tests
- [ ] Token budget accounting verified against real provider billing
- [ ] docs.rs documentation complete with examples
- [ ] No `unsafe` blocks without `// SAFETY:` comments
- [ ] `cargo-semver-checks` in CI
- [ ] Whisper transcription tested against reference audio corpus

---

## Post-v1

### New backends
- [ ] **ONNX Runtime** — local ONNX model inference
- [ ] **vLLM** — high-throughput serving with PagedAttention
- [ ] **TensorRT-LLM** — NVIDIA-optimised inference
- [ ] **Candle** — pure-Rust inference (Hugging Face)
- [ ] **Custom model serving** — user-trained models via hoosh plugin API

### Advanced inference
- [ ] **Tool use / function calling** — unified across providers
- [ ] **Vision models** — image input support (GPT-4V, Claude, Gemini)
- [ ] **Embeddings server** — local embedding generation for RAG
- [ ] **Agent memory** — conversation history with vector retrieval
- [ ] **Multi-modal routing** — route text/image/audio to appropriate backends

### Speech
- [ ] **Text-to-speech** — piper-tts or bark integration
- [ ] **Real-time STT** — streaming audio → streaming text
- [ ] **Speaker diarization** — who-spoke-when from hoosh

### Platform
- [ ] **WASM target** — browser-based inference client
- [ ] **C FFI bindings** — `hoosh.h` for C/C++ consumers
- [ ] **Python bindings** — PyO3 package

---

## Non-goals

- **Model training** — hoosh is for inference. Training is Synapse's domain.
- **Model storage** — hoosh doesn't manage model files. Synapse, Ollama, or the filesystem own that.
- **Direct GPU compute** — hoosh delegates to backends that own the GPU. It uses ai-hwaccel for detection, not for running kernels.
- **Web UI** — hoosh is an API gateway. Dashboard/UI is a separate application.
