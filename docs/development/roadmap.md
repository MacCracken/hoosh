# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v0.20.3 — Core Gateway + Provider Backends ✅

All items complete. See CHANGELOG.md for details.

- [x] Core types (InferenceRequest, InferenceResponse, Message, Role, ModelInfo)
- [x] LlmProvider trait + ProviderType enum (14 backends)
- [x] Router (Priority, RoundRobin, LowestLatency, Direct)
- [x] ResponseCache with TTL + forced eviction
- [x] TokenBudget with named pools + reserve/commit/release
- [x] HooshClient (infer, infer_stream, list_models, health)
- [x] axum server with OpenAI-compatible API
- [x] 5 local providers (Ollama, llama.cpp, Synapse, LM Studio, LocalAI)
- [x] 7 remote providers (OpenAI, Anthropic, DeepSeek, Mistral, Groq, OpenRouter, Grok)
- [x] Hardware-aware placement (ai-hwaccel with vulkan/rocm/cuda)
- [x] TOML config file (hoosh.toml)
- [x] Whisper STT (whisper-rs, /v1/audio/transcriptions)
- [x] TTS endpoint (/v1/audio/speech via HTTP backend)
- [x] CLI (serve, models, health, info, infer, infer --stream, transcribe, speak)
- [x] Token budget integration in request pipeline
- [x] 3 security audit rounds, all CRITICAL/HIGH fixed
- [x] 185 tests, benchmarks, cargo-deny

---

## v0.21.3 — Server Hardening & Observability

All items complete. See CHANGELOG.md for details.

### Authentication & security
- [x] Bearer token auth middleware
- [x] Per-provider rate limiting middleware (sliding window RPM)
- [x] TLS certificate pinning for remote providers
- [x] mTLS for local provider communication

### Observability
- [x] Prometheus metrics endpoint (`/metrics`)
- [x] OpenTelemetry trace propagation (feature-gated: `otel`)
- [x] Per-provider cost tracking (token × price) — ported from secureyeoman's cost-calculator pattern
- [x] Request/response audit log — HMAC-SHA256 linked chain, ported from secureyeoman's sy-audit pattern

### Hardware acceleration
- [x] Review ai-hwaccel 0.21.x updates for improved integration — no breaking changes; new LazyRegistry, DiskCachedRegistry, cost module available

### Server improvements
- [x] `/v1/embeddings` pass-through
- [x] Hot-reload config without restart (SIGHUP + `POST /v1/admin/reload`)
- [x] Periodic health checks with automatic failover (3-strike unhealthy marking)
- [x] Latency tracking for LowestLatency routing

---

## v0.22.3 — Advanced Inference

### Caching improvements
- [ ] Semantic cache (embedding similarity, not just exact match)
- [ ] Cache warming for common prompts
- [ ] Cache statistics API

### Provider acceleration
- [ ] Batch inference (group compatible requests)
- [ ] Speculative decoding hints
- [ ] Prompt compression for long conversations

### Multi-modal
- [ ] Tool use / function calling — unified across providers
- [ ] Vision models — image input support (GPT-4V, Claude, Gemini)
- [ ] Multi-modal routing — route text/image/audio to appropriate backends

---

## v0.23.3 — Speech & Audio

### Whisper improvements
- [ ] Model download and management (tiny, base, small, medium, large)
- [ ] Audio format support beyond WAV: MP3, OGG, FLAC
- [ ] Streaming transcription (chunked audio input)
- [ ] Speaker diarization — who-spoke-when

### TTS improvements
- [ ] Native piper-rs integration (blocked on upstream ort compatibility)
- [ ] Multiple voice support with voice selection API
- [ ] Real-time streaming TTS
- [ ] Bark / XTTS integration for voice cloning

---

## v1.0.0 Criteria

All of the following must be true before cutting 1.0:

- [ ] Public API reviewed and marked stable
- [ ] `LlmProvider` trait finalized
- [ ] Core types (`InferenceRequest`, `InferenceResponse`, `ModelInfo`) frozen
- [ ] 90%+ line coverage
- [x] At least 5 provider backends fully implemented and tested (12 done)
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
- [ ] **TPU** — Google TPU inference via libtpu
- [ ] **Gaudi** — Intel Gaudi accelerator backend
- [ ] **Inferentia** — AWS Inferentia/Trainium via neuron SDK
- [ ] **OneAPI** — Intel Arc/Data Center GPU Max via oneAPI
- [ ] **Qualcomm AI 100** — Qualcomm Cloud AI 100 backend
- [ ] **AMD XDNA** — AMD Ryzen AI NPU backend
- [ ] **Metal** — Apple Metal Performance Shaders backend
- [ ] **Vulkan** — Vulkan compute backend
- [ ] **GGUF** — direct GGUF model loading (currently via llama.cpp)

### Ifran integration (library-mode API)
- [ ] **Embeddings trait** — expose embeddings generation through `LlmProvider` so Ifran can drop its per-backend embedding wrappers
- [ ] **Privacy-aware routing** — route-level privacy classification (local-only vs remote-allowed) so Ifran can enforce data residency without its own `BackendRouter`
- [ ] **Backend capability reporting** — expose accelerator type, context length, streaming support, vision support per provider so Ifran can query capabilities without `ai-hwaccel` directly
- [ ] **Programmatic server builder** — `HooshServer::builder()` API for embedding the gateway inside another Axum app (Ifran's API server) without spawning a separate process

### Advanced features
- [ ] **Embeddings server** — local embedding generation for RAG
- [ ] **Agent memory** — conversation history with vector retrieval

### Platform
- [ ] **WASM target** — browser-based inference client
- [ ] **C FFI bindings** — `hoosh.h` for C/C++ consumers
- [ ] **Python bindings** — PyO3 package

---

## Non-goals

- **Model training** — hoosh is for inference. Training is Ifran's domain.
- **Model storage** — hoosh doesn't manage model files. Ifran, Ollama, or the filesystem own that.
- **Direct GPU compute** — hoosh delegates to backends that own the GPU. It uses ai-hwaccel for detection, not for running kernels.
- **Web UI** — hoosh is an API gateway. Dashboard/UI is a separate application.
