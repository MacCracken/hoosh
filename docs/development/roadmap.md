# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v2.0.0 — Cyrius Port (current)

Hoosh rewritten from Rust to Cyrius. All core gateway functionality ported.

### Completed
- [x] 13 provider backends (Ollama, LlamaCPP, Synapse, LMStudio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter)
- [x] OpenAI-compatible `/v1/chat/completions` (streaming SSE + non-streaming)
- [x] Provider routing (priority, round-robin, lowest-latency)
- [x] Token budget system (check/reserve/report/pools)
- [x] Bearer token authentication (constant-time comparison)
- [x] HMAC-SHA256 audit chain (tamper-proof logging)
- [x] Retry with jittered exponential backoff
- [x] Per-provider rate limiting (RPM token bucket)
- [x] Response cache with LRU eviction
- [x] Context compaction (system message preserved + recent N)
- [x] Hardware detection via ai-hwaccel 2.0.0 (GPU/NPU/TPU)
- [x] Hardware placement, model compatibility endpoints
- [x] Embeddings pass-through routed via provider system
- [x] Prometheus metrics endpoint
- [x] CORS support
- [x] Config hot-reload via `/v1/admin/reload`
- [x] CLI: serve, models, health, infer, info, help, version

---

## v2.1.0 — Production Hardening

### Tool calling & MCP
- [ ] `/v1/tools/list` — list registered MCP tools
- [ ] `/v1/tools/call` — invoke MCP tools by name
- [ ] Streaming tool call assembly (incremental deltas)

### DLP (Data Loss Prevention)
- [ ] PII pattern scanner (email, phone, SSN, credit card, API keys)
- [ ] Classification levels (Public/Internal/Confidential/Restricted)
- [ ] Privacy-aware routing (Confidential → local-only, Restricted → block)

### TLS & Security
- [ ] TLS client support for remote providers (requires Cyrius TLS lib)
- [ ] Certificate pinning for remote endpoints
- [ ] mTLS for local provider authentication

### Concurrency
- [ ] Multi-threaded accept loop (thread pool or epoll)
- [ ] Connection pooling for backend sockets

### Observability
- [ ] OpenTelemetry trace propagation
- [ ] Per-provider latency histograms in Prometheus

---

## v2.2.0 — Extended Backends

### New backends
- [ ] vLLM — high-throughput serving with PagedAttention
- [ ] TensorRT-LLM — NVIDIA-optimised inference
- [ ] ONNX Runtime — local ONNX model inference

### Advanced features
- [ ] Semantic cache (cosine similarity over embeddings)
- [ ] Batch inference manager
- [ ] Cost optimizer (cheapest capable model recommendation)

---

## Deferred (external dependencies)

### svara — Speech/Audio (migration pending)
- STT (Whisper) and TTS (Piper) migrating from hoosh to svara
- Hoosh retains provider interface; svara owns audio pipeline
- Endpoints `/v1/audio/transcriptions` and `/v1/audio/speech` will not be ported

### Remote provider DNS/TLS
- Remote providers (OpenAI, Anthropic, etc.) require DNS resolution and TLS
- Blocked on Cyrius stdlib DNS and TLS support

---

## Non-goals

- **Model training** — hoosh is for inference
- **Model storage** — hoosh doesn't manage model files
- **Direct GPU compute** — hoosh delegates to backends; ai-hwaccel handles detection
- **Web UI** — hoosh is an API gateway; dashboard is separate
- **Audio pipeline** — speech processing belongs to svara
