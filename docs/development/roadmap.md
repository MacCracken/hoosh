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

## v2.1.0 — Cyrius 6.0 modernization (shipped 2026-06-04)

Toolchain/scaffolding brought up to current Cyrius (6.0.x) conventions. See
[CHANGELOG](../../CHANGELOG.md) and
[ADR 007](../decisions/007-cyrius-6-modernization.md).

- [x] Cyrius pin 4.5.0 → 6.0.57; ai-hwaccel 2.0.0 → 2.3.7 (distlib bundle)
- [x] `cyrius.toml` → `cyrius.cyml` (`${file:VERSION}`), retire `.cyrius-toolchain`
- [x] Source to green under 6.0.57 (sys_* wrappers, HMAC, toml-cstr config fix)
- [x] CI/release modernized (canonical installer, lib sync, fmt/lint/vet gates)
- [x] **Benchmarks mandatory** — hard CI release gate
- [x] De-Rust scripts (`bench-history.sh`, `version-bump.sh`) + `.gitignore`

---

## v2.1.x — Production Hardening (in progress)

The hardening features below continue on the 2.1.x line.

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

### Scaffolding modernization (backlog — from ai-hwaccel/patra review)
Conventions the modern sibling repos (ai-hwaccel 2.3.7, patra 1.10.3) follow that
hoosh has not adopted yet. None block 2.1.0; candidates for the 2.1.x line:
- [ ] `docs/development/state.md` — volatile state (version, test/assertion
      counts, binary size, recent releases), refreshed each release (patra pattern)
- [ ] `docs/doc-health.md` — doc inventory / freshness tracking (patra pattern)
- [ ] Split tests into `tests/tcyr/*.tcyr` + `benches/*.bcyr` per-topic units
      (ai-hwaccel pattern). Current single `hoosh.tcyr`/`hoosh.bcyr` is fine
      (patra-style) — only worth it if the suite keeps growing.
- [ ] Fuzz harnesses (`fuzz/*.fcyr`) + a CI fuzz step (both siblings)
- [ ] Security-pattern scan in CI (raw `execve`, hardcoded `/etc` paths — patra)
- [ ] `cyrius deny` policy file + CI gate (currently lint/vet only)

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
