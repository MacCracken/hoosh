# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v2.0.0 — Cyrius Port (current)

Hoosh rewritten from Rust to Cyrius. All core gateway functionality ported.

> **Transport status** (updated 2026-06-09): remote transport is wired (see
> [v2.2.0](#v220--remote-provider-transport-the-criticals)) — OpenAI-compatible
> cloud providers (OpenAI, DeepSeek, Mistral, Groq, Grok, OpenRouter) work
> end-to-end over TLS via sandhi, alongside the local backends (Ollama, LlamaCPP,
> Synapse, LMStudio, LocalAI). Anthropic/Gemini need request-response shaping
> still.

### Completed
- [x] 13 provider backends defined (Ollama, LlamaCPP, Synapse, LMStudio, LocalAI
      — wired; OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok,
      OpenRouter — defined, awaiting remote transport)
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

## v2.1.x — Production Hardening (shipped)

Released on this line: 2.1.1 (hardware planning endpoints), 2.1.2 (structured
logging), 2.1.3 (patra persistence), 2.1.4 (toolchain refresh — cyrius 6.1.18,
ai-hwaccel 2.3.9).

- [x] Hardware planning endpoints — `cost`, `training-estimate`,
      `compatible-models` (2.1.1)
- [x] Optional persistence via patra — audit chain + token budgets survive
      restarts; opt-in `[[storage]] path` (2.1.3)
- [x] Structured operational logging via sakshi (stderr, leveled; `[[logging]]`
      config) (2.1.2)

The line is closed for new scope. The **[v2.2.x parity arc](#v22x--port-parity-arc)**
is now the focus; everything else that was open on this roadmap is deferred
behind it to **[v2.3.x](#v23x--post-parity-deferred)**.

---

## v2.2.x — Port Parity Arc

The Cyrius port (v2.0.0) shipped the gateway control plane but deferred the
remote data plane and several Rust-era features. This arc closes the gap between
the `rust-old` reference and the Cyrius port (audit 2026-06-09). Ordered by
severity — **2.2.0 ships the criticals**; later point releases work the
remaining partials and missing features. (Audio STT/TTS is deliberately *not*
here — it is migrating to **svara**; see Deferred.)

### v2.2.0 — Remote provider transport (the criticals)

Without these, "multi-provider cloud routing" is unbacked. The forward client
(`src/lib/http_client.cyr`) hand-rolls raw sockets to `127.0.0.1`
(`sockaddr_in(0x0100007F, port)`), keeps just the **port** from each route's base
URL (host discarded), speaks plaintext HTTP/1.0, and injects **no auth**. The 8
cloud providers (OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok,
OpenRouter) are enum + default-URL entries the transport never reaches.

**No external blocker** — the cyrius stdlib already ships everything needed:
`lib/sandhi.cyr` (high-level HTTPS client: URL parse + DNS + TLS + connect via
`sandhi_http_post`), over `lib/tls.cyr` (TLS) and `lib/net.cyr` (sockets). This
is hoosh wiring, not a Cyrius gap.

**Status (landed):** OpenAI-compatible cloud providers (OpenAI, DeepSeek,
Mistral, Groq, Grok, OpenRouter) now work end-to-end — `https://` routes forward
through sandhi with TLS + `Bearer` auth, in both non-streaming and (buffered)
streaming modes. Remaining: Anthropic/Gemini request-response shaping, incremental
remote streaming, cert pinning (checklist below).

- [x] Add `tls`/`sandhi` (+ `mmap`/`dynlib`/`fdlopen`) to `cyrius.cyml` `[deps]`
      in include order (`net`/`http` already present)
- [x] Remote forward via **sandhi**'s high-level client (`sandhi_http_post` —
      URL parse + DNS + TLS + connect in one call), scheme-dispatched in
      `provider.cyr`: `https://` → sandhi (`route_is_remote`), local `http://`
      keeps the raw-socket fast path
- [x] TLS for `https://` routes (sandhi over `lib/tls.cyr`); plaintext for local
- [x] Auth header injection — `Authorization: Bearer` (OpenAI-compat) and
      `x-api-key` + `anthropic-version` (Anthropic), built from `route_api_key`
- [x] Token extraction for OpenAI `usage` objects (`extract_openai_tokens`,
      Ollama fields fall back)
- [x] Remote streaming: buffered fallback (full response → one SSE delta + stop)
      so `stream:true` clients work against remote providers
- [ ] **Anthropic request/response shaping** — `/v1/messages` body (`max_tokens`,
      system hoist) + `content[].text` extraction. Auth headers are wired; full
      body/response translation is the remaining gap (OpenAI-compat providers —
      OpenAI, DeepSeek, Mistral, Groq, Grok, OpenRouter — work end-to-end now).
- [ ] **Google/Gemini shaping** — `:generateContent` API + key-as-query-param
- [ ] **Incremental remote streaming** — read TLS via `sandhi_conn_recv` instead
      of `sys_read`, replacing the buffered fallback
- [ ] Certificate pinning + optional mTLS for local providers (hardening)

### v2.2.1 — Provider correctness & completeness
- [ ] Per-provider token-count ratios — restore `context/tokens.rs` behavior
      (`compact.cyr` currently hardcodes 4 chars/token for every provider)
- [ ] Provider lifecycle methods — Ollama `pull_model`/`delete_model`, Synapse
      `training_status`/`sync_catalog`
- [ ] Native Ollama `/api/tags` inbound route (list models)

### v2.2.2 — DLP (Data Loss Prevention)
Today only `ERR_DLP_BLOCKED` + a test-stub `@`-scan exist; the real scanner
(`dlp/scanner.rs`, 364 L) was not ported.
- [ ] PII pattern scanner (email, phone, SSN, credit card, API keys)
- [ ] Classification levels (Public/Internal/Confidential/Restricted)
- [ ] Privacy-aware routing (Confidential → local-only, Restricted → block)

### v2.2.3 — Cost & cache intelligence
- [ ] Cost optimizer — cheapest capable model recommendation (`cost/optimizer.rs`)
- [ ] Semantic cache — cosine similarity over embeddings (`cache/semantic.rs`);
      distinct from the exact-key LRU cache already ported
- [ ] Cache warming — startup pre-population (`cache/warming.rs`)
- [ ] Context compression — whitespace collapse, stale tool-pair prune, dedup
      (`context/compression.rs`); distinct from the compaction already ported

### v2.2.4 — Tool calling & MCP
- [ ] `/v1/tools/list` — list registered MCP tools
- [ ] `/v1/tools/call` — invoke MCP tools by name
- [ ] Streaming tool call assembly (incremental deltas)

### v2.2.5 — Throughput
- [ ] Inference batching manager (`inference/batch.rs`) — gated on the
      multi-threaded accept loop (BLOCKED; see v2.3.x concurrency)
- [ ] Connection pooling for backend sockets

### v2.2.6 — Observability
- [ ] OpenTelemetry trace propagation (`telemetry.rs`)
- [ ] Event pub/sub bus (`events.rs`) — HealthChanged / InferenceCompleted /
      InferenceFailed / RateLimited
- [ ] Per-provider latency histograms in Prometheus

---

## v2.3.x — Post-parity deferred

Everything that was open on the roadmap before the parity arc. Kicked back —
not started until the 2.2.x arc lands.

### Hardware planning (remaining ai-hwaccel surface)
- [ ] `POST /v1/hardware/model-format` — detect SafeTensors/GGUF/ONNX/PyTorch
      (ai-hwaccel `model_format.cyr`)
- [ ] `POST /v1/hardware/requirement-match` — scheduler requirement matching
      (ai-hwaccel `requirement.cyr`)
- [ ] Threaded detection at startup (`registry_detect_threaded`) — blocked:
      segfaults under the single-threaded runtime; revisit with the threaded
      accept loop.

### Concurrency
- [ ] Multi-threaded accept loop — **BLOCKED** (race audit 2026-06-04): the
      cyrius global allocator is not thread-safe — concurrent `alloc` corrupts
      memory (verified ~5000 corruptions across 4 threads). All stdlib allocation
      routes through the global allocator (the allocator-as-parameter convention
      in `alloc.cyr` keeps the global fns racy for back-compat), so per-thread
      arenas do not help and a global processing mutex would serialize all
      request handling. Threading primitives themselves (thread_create/join,
      mutex, channels) work. Revisit when cyrius provides thread-safe allocation.
      (Unblocks v2.2.5 batching.)
- [ ] Connection pooling for backend sockets

### New backends
- [ ] vLLM — high-throughput serving with PagedAttention
- [ ] TensorRT-LLM — NVIDIA-optimised inference
- [ ] ONNX Runtime — local ONNX model inference

### Scaffolding modernization (from ai-hwaccel/patra review)
Conventions the modern sibling repos (ai-hwaccel 2.3.9, patra 1.10.3) follow that
hoosh has not adopted yet.
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

## Deferred (external dependencies)

### svara — Speech/Audio (migration pending)
- STT (Whisper) and TTS (Piper) migrating from hoosh to svara
- Hoosh retains provider interface; svara owns audio pipeline
- Endpoints `/v1/audio/transcriptions` and `/v1/audio/speech` will not be ported

(Remote DNS/TLS is **not** deferred — the cyrius stdlib already ships it
(`lib/tls.cyr`, `lib/net.cyr`, `lib/sandhi.cyr`); the work is hoosh wiring,
tracked as the [v2.2.0](#v220--remote-provider-transport-the-criticals)
criticals.)

---

## Non-goals

- **Model training** — hoosh is for inference
- **Model storage** — hoosh doesn't manage model files
- **Direct GPU compute** — hoosh delegates to backends; ai-hwaccel handles detection
- **Web UI** — hoosh is an API gateway; dashboard is separate
- **Audio pipeline** — speech processing belongs to svara
