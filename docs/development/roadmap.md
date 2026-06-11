# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v2.0.0 — Cyrius Port (current)

Hoosh rewritten from Rust to Cyrius. All core gateway functionality ported.

> **Transport status** (updated 2026-06-09): remote transport is wired and
> unblocked (see [v2.2.0](#v220--remote-provider-transport-the-criticals)) — all
> cloud providers work end-to-end over TLS via sandhi: OpenAI-compatible (OpenAI,
> DeepSeek, Mistral, Groq, Grok, OpenRouter), **Anthropic** (`/v1/messages` +
> system-message hoist), and **Google/Gemini** (`:generateContent` shaping),
> alongside the local backends (Ollama, LlamaCPP, Synapse, LMStudio, LocalAI).
> Streaming is incremental for remote providers via `sandhi_http_stream`. The
> sandhi P1 repeated-HTTPS crash is fixed (cyrius 6.1.20 / sandhi 1.4.5).

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

### v2.2.0 — Remote provider transport (the criticals) — ✅ SHIPPED 2026-06-09

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

**Status (2026-06-09): SHIPPED in 2.2.0 — sandhi P1 fixed, criticals landed.** The
sandhi repeated-HTTPS SIGSEGV was fixed upstream (cyrius 6.1.20 / sandhi 1.4.5:
alloc heap moved off `brk` onto mmap, sandhi default-switched to the native TLS
backend); hoosh bumped its pin 6.1.18 → 6.1.20 and re-verified crash-free (6/6
sequential HTTPS, clean exit). With the unblock, the remaining shaping/streaming
criticals all landed: **Anthropic system-message hoist**, **Google/Gemini
shaping**, and **incremental remote streaming** (via `sandhi_http_stream`'s SSE
callback, with a buffered fallback when nothing streams). 278 tests green. **All
three cloud families are live-verified end-to-end** through the gateway —
OpenAI (`gpt-4o-mini`), Anthropic (`claude-haiku-4-5`, plus system-message hoist
+ streaming + repeated requests, no crash), and Google/Gemini
(`gemini-2.5-flash`). Live verification also caught and fixed a
whitespace-tolerance bug: OpenAI and Gemini pretty-print their REST responses, so
the compact-only extractors silently dropped their text/tokens (see CHANGELOG).

**Critical build requirement (found during live smoke testing):** the gateway
must run on sandhi's **native** TLS backend. Without it, hoosh runs on the
deprecated libssl fdlopen bridge, which **SIGSEGVs on the 2nd–4th remote
request** (the brk-malloc/TLS-arena family of the upstream P1) — this affected
stream *and* non-stream. Native loads no libssl at all and is crash-free.
Originally this required `-D CYRIUS_TLS_NATIVE` at build time; **as of cyrius
6.1.21 / sandhi native-default (hoosh 2.2.2), native is the default and no flag
is needed** — the libssl bridge is now the explicit opt-out (`-D
CYRIUS_TLS_LIBSSL`, which hoosh never passes). `main()` still asserts native via
`sandhi_tls_use_native()` and warns if a libssl-only build disabled it. See
CLAUDE.md.

The only deferred item is **cert pinning**, which needs an upstream sandhi
feature (the high-level client can't carry a TLS policy) — see below.

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
- [x] **Anthropic request/response shaping** — `/v1/messages` body (`max_tokens`)
      + `content[].text` / `usage.{input,output}_tokens` extraction. Live-verified
      against the real API.
- [x] **`$ENV` key expansion** in config (`_config_expand_env`) — `api_key =
      "$ANTHROPIC_API_KEY"` resolves from the environment; secret stays out of
      hoosh.cyml.
- [x] ✅ **UNBLOCKED: sandhi P1 SIGSEGV fixed** (cyrius 6.1.20 / sandhi 1.4.5;
      alloc heap off `brk`, native TLS default). Pin bumped 6.1.18 → 6.1.20,
      re-verified crash-free (6/6 sequential HTTPS, clean exit).
- [x] **Anthropic system-message hoist** — `role:"system"` turns lift to the
      top-level `system` field (multiple joined with `\n`); remaining turns pass
      through. `_build_anthropic_body` + `_json_obj_*`/`_obj_is_system` helpers.
- [x] **Google/Gemini shaping** — `_build_gemini_body` maps OpenAI messages →
      `contents` (assistant→model, system→`systemInstruction`); `_gemini_url`
      builds the model-scoped `:generateContent?key=` URL (key as query param, no
      auth header); `gemini_extract_text` + `extract_gemini_tokens` for the
      `candidates`/`usageMetadata` response.
- [x] **Incremental remote streaming** — `handle_chat_stream`'s remote branch now
      drives `sandhi_http_stream` with a per-event SSE callback (`_remote_stream_cb`),
      decoding each provider's delta (OpenAI `delta.content`, Anthropic
      `content_block_delta`, Gemini `:streamGenerateContent?alt=sse`) and
      re-emitting OpenAI SSE chunks. Falls back to the buffered path if the
      stream errors before emitting anything (no duplicate output).
- [ ] Certificate pinning + optional mTLS for local providers (hardening) —
      **deferred: blocked on a sandhi P1 (wiring gap).** sandhi *already* has live
      pinning/mTLS — `sandhi_tls_policy_new_pinned` enforced by
      `sandhi_conn_open_with_policy` — but the high-level `sandhi_http_post`/
      `_stream` client never threads a policy (`sandhi_http_options` has no policy
      field; the request path opens via plain `sandhi_conn_open_fully_timed_a`).
      So pinning is unreachable from the high-level API without hand-rolling
      HTTP+chunked+SSE over `sandhi_conn` (losing the streaming client). Filed P1
      on sandhi to thread a policy through `sandhi_http_options`:
      `sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`. Pick up
      hoosh-side once that lands.

### v2.2.1 — Provider correctness & completeness — ✅ SHIPPED 2026-06-09
- [x] Per-provider token-count ratios — restored `context/tokens.rs` behavior in
      `compact.cyr`: `chars_per_token_x10` (OpenAI-family 3.8, Anthropic 3.5,
      local 3.7, default 4.0, scaled ×10 for float-free integer math);
      `compact_messages` now takes the target provider's ratio via
      `route_provider(route)`. Unknown providers keep the prior 4.0 default.
- [x] Provider lifecycle methods — Ollama `pull_model`/`delete_model`, Synapse
      `training_status`/`sync_catalog`, exposed as `POST /v1/models/pull`,
      `POST /v1/models/delete`, `POST /v1/training/status`, `POST /v1/catalog/sync`.
      Each forwards to the configured backend (via `http_req_local`, generalized
      to GET/POST/DELETE) and 404s when the provider isn't configured.
      Live-verified against a mock backend (method/path/body correct).
- [x] Native Ollama `/api/tags` inbound route (`handle_ollama_tags`) — lists
      enabled routes' model patterns in Ollama's tags shape.

### v2.2.2 — DLP (Data Loss Prevention) — ✅ SHIPPED 2026-06-10
Ported `dlp/scanner.rs` (364 L) — previously only `ERR_DLP_BLOCKED` + a test
stub existed. Implemented in `src/lib/dlp.cyr` as hand-rolled byte-level matchers
(no regex engine in the Cyrius port, no new dependency).
- [x] PII pattern scanner — email, phone_us, ssn, credit_card, ipv4, api_key,
      aws_key, github_token (8 built-ins, `\b`-aware), live-verified end-to-end.
- [x] Classification levels (`DlpClass`: Public/Internal/Confidential/Restricted,
      ordered) + `[dlp]` config (`enabled`, `default_level`).
- [x] Privacy-aware routing in `handle_chat` — Restricted → block (403);
      Confidential → local provider only (`router_select_local`, 403 if none);
      Internal/Public pass.

### v2.2.3 — Cost & cache intelligence — ✅ SHIPPED 2026-06-10
- [x] **Response cache wired into `/v1/chat/completions`** — prerequisite found
      during this arc: the exact-key LRU cache was ported but **inert** (never
      read/written by `handle_chat`). Now non-streaming requests key off
      sha256(model+body), hit short-circuits before forward, miss stores the
      body. Live-verified (hits/misses in `/v1/cache/stats`).
- [x] **Cache warming** — startup pre-population (`cache/warming.rs`).
      `[[warming]]` config (model + prompt) + a synchronous startup warm loop
      (`warming_run` in `cmd_serve`; no background task — single-threaded
      runtime). Warms under the exact key a client request hits; live-verified
      (startup warmed 1/1 → client request cache hit).
- [x] **Cost optimizer** (`cost/optimizer.rs`) — cheapest *capable* model.
      `pricing.cyr` (16-model per-token table from `cost/mod.rs`, exact/prefix/
      fallback) + `metadata.cyr` (per-model tier/context/vision/tools/system +
      `classify_complexity`/`meets_requirements` from the registry slice).
      `POST /v1/cost/estimate` and `POST /v1/cost/recommend` (capability-filtered).
      Live-verified across tier/vision/tools/context profiles.
      (Note: per-token API pricing is hardcoded, *not* `data/cloud_pricing.json`,
      which is cloud-GPU $/hour for hardware planning.)
- [x] **Semantic cache** (`cache/semantic.rs`) — `semantic.cyr`: fixed-point
      cosine (`cosine_x1000`, integer-sqrt magnitudes), embedding store
      (`semantic_insert`/`semantic_find`, threshold + `max_search`), embeddings
      float-vector parser, `[semantic_cache]` config, **and chat-path wiring**
      (`_embed_query_body` in `handle_chat` — embed the query on a miss, search/
      store; silent fallthrough on embedding failure). Unit-tested + live-verified
      (paraphrased queries hit semantically).
- [x] Context compression (`context/compression.rs`) — whitespace collapse
      (`compress_ws_json`, JSON-aware) **and** stale tool-pair prune
      (`prune_tool_pairs`, completed in v2.2.4); `[compression]` opt-in, applied
      before compaction.

### v2.2.4 — Tool calling — ✅ SHIPPED 2026-06-10
Provider-side tool calling across all three remote families. (The MCP server
endpoints `/v1/tools/list` + `/v1/tools/call` moved to v2.2.5.)
- [x] **Provider-side tool calling — OpenAI, Anthropic, Gemini** — `handle_chat`
      forwards the request's `tools` (converting to each native format via
      `_tools_convert`: OpenAI verbatim, Anthropic `input_schema`, Gemini
      `functionDeclarations`) and surfaces tool calls back as OpenAI `tool_calls`
      + `finish_reason:"tool_calls"` (`_extract_openai_tool_calls`,
      `_anthropic_tool_calls`, `_gemini_tool_calls`). Ports `tools/convert.rs`.
      Live-verified against all three providers; unit-tested.
- [x] **Streaming tool call assembly (incremental deltas) — all three families**:
      `stream:true` forwards `tools` and converts each provider's streaming tool
      deltas to OpenAI `tool_calls` chunks (`_sse_tool_chunk`; OpenAI pass-through,
      Anthropic `_emit_anthropic_tool_delta`, Gemini via `_gemini_tool_calls`).
      Live-verified against all three.
- [x] **Stale tool-pair prune** in `compression.cyr` (`prune_tool_pairs`,
      deferred from 2.2.3) — drops assistant `tool_calls` turns + their matching
      `role:"tool"` results before the last 3 tool turns. Unit-tested. This
      completes the `context/compression.rs` port.

---

## v2.3.0 — MCP tool server — ✅ SHIPPED 2026-06-10

The last open item of the parity arc (was tracked as v2.2.5). With it landed,
the v2.2.x parity arc is complete, so MCP ships as **2.3.0**. Toolchain also
bumped to the latest Cyrius (**6.1.27**).

bote (`dist/bote-core.cyr`, the `[lib.core]` profile) is the JSON-RPC 2.0
`ToolRegistry`/`Dispatcher`/codec. The tool implementations (**szál**, 58
built-ins) are not yet a Cyrius distlib, so the registry holds a single built-in
`bote_echo` smoke tool until they ship — register them in `mcp_init`
(`src/lib/mcp.cyr`) alongside `bote_echo` and they appear with no transport
changes.
- [x] `GET /v1/tools/list` — list registered MCP tools (bote `ToolRegistry`),
      JSON-RPC result shape. Live-verified.
- [x] `POST /v1/tools/call` — invoke a tool by name (bote codec + `Dispatcher`);
      MCP JSON-RPC body, `initialize`/`tools/list` also accepted. Live-verified
      (echo round-trip, initialize handshake, unknown-tool error, empty-body 400).
- [x] `src/lib/mcp.cyr` wiring + `bote_echo` built-in; `mcp_init` from
      `cmd_serve`. Unit-tested (`mcp_tools`) + benched (`mcp_tools_list`/`_call`).
- [x] **bote vendored at `src/vendor/bote-core.cyr`** (not `[deps.bote]`) — bote's
      manifest declares libro/majra git sub-deps that `cyrius deps` resolves
      transitively into a colliding compile set; the self-contained core bundle
      sidesteps it. Re-sync: `./scripts/sync-bote.sh <tag>`.

### v2.3.1 — Concurrent batch inference — ✅ SHIPPED 2026-06-10
- [x] **`POST /v1/batch`** — concurrent batch executor (in-process worker pool,
      bounded waves, atomic-counter barrier). `handle_chat` refactored into a
      returns-body core; coarse `_chat_lock` around bookkeeping with the network
      forward unlocked; atomic `metrics_record`. Ports the concurrency core of
      `inference/batch.rs`. **No longer blocked** — the cyrius 6.1.27 allocator
      is thread-safe (v6.0.64 CAS spinlock; re-verified by a 4-thread alloc
      stress, 0 corruption). See [ADR 009](../decisions/009-concurrent-batch-inference.md).
- [ ] **Deferred — connection pooling for backend sockets.** Low near-term ROI:
      the local path connects to `127.0.0.1` (loopback connect ≪ inference
      latency); the high-value case is remote TLS-handshake reuse, which is gated
      on sandhi keep-alive/pooling support. Revisit when sandhi exposes it.

### v2.3.2 — Async batch inference — ✅ SHIPPED 2026-06-10
- [x] **`POST /v1/batch {"async":true}`** → returns a batch id; a background
      runner executes the batch (waves of 7 crypto-bank workers).
- [x] **`GET /v1/batch/{id}`** progress (queued/running/completed/cancelled,
      completed/failed counts, per-item results), **`POST /v1/batch/{id}/cancel`**.
- [x] **Sync pass** — `handle_chat` now takes `_chat_lock` so a chat served on
      the accept loop is safe concurrent with background batch workers
      (live-verified: 20/20 concurrent chats during a 28-item batch). One batch
      executes at a time (`_batch_exec_lock`) to keep crypto lanes exclusive.
      See [ADR 009](../decisions/009-concurrent-batch-inference.md) §"Update (2.3.2)".
      See [ADR 009](../decisions/009-concurrent-batch-inference.md) §"Update (2.3.2)".

### v2.3.3 — Concurrent batches + registry eviction — ✅ SHIPPED 2026-06-10
- [x] **Concurrent async batches** — replaced the one-at-a-time exec lock with a
      **global crypto-lane pool** (banks 1..7 shared across all batches; ≤7 live
      workers total). Multiple async batches run concurrently and interleave
      fairly. Live-verified (4 batches progressing simultaneously, 12/12 each).
- [x] **Registry eviction** — `BATCH_MAX_TRACKED`=64; oldest terminal batch
      evicted on overflow (evicted ids 404). Bounds the map (not heap — bump
      allocator never frees).
- [x] **De-spin** — lane acquire + barriers `sleep_ms(1)` instead of busy-spin,
      keeping the accept loop responsive to polls. Toolchain → **6.1.28**.
      See [ADR 009](../decisions/009-concurrent-batch-inference.md) §"Update (2.3.3)".

### v2.4.0 — Multi-threaded accept loop
Now **unblocked** (allocator thread-safe, verified in 2.3.1). Thread the accept
loop so *all* traffic runs concurrently — requires a shared-state synchronization
pass across every handler (cache/budget/audit/rate/cost/metrics). 2.3.2's sync
pass already locked the chat path; this extends it to the remaining mutating
handlers. Enables loopback-style batching and general throughput.

### v2.3.4 — Observability — ✅ SHIPPED 2026-06-10
- [x] **Per-provider latency histograms** — `hoosh_provider_latency_ms`
      Prometheus histogram (per provider) in `/metrics`.
- [x] **Event pub/sub bus** (`events.rs`) — majra pubsub carrying HealthChanged /
      InferenceCompleted / InferenceFailed / RateLimited, published as JSON; bus
      count in `/metrics`, recent events at **`GET /v1/events/recent`**.
- [x] **W3C `traceparent` propagation** (the lightweight OTel slice) — incoming
      traceparent forwarded to backends, generated when absent.
- [ ] **→ 2.3.5:** full OpenTelemetry OTLP span export (`telemetry.rs`) — needs a
      gRPC/protobuf exporter; its own effort.
      See [ADR 010](../decisions/010-observability.md).

### v2.3.5 — OpenTelemetry OTLP export
- [ ] OTLP span exporter + span emission around inference (deferred from 2.3.4;
      builds on the 2.3.4 traceparent propagation).

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
- [x] ✅ **UNBLOCKED: cyrius global allocator is now thread-safe.** The
      2026-06-04 race audit (~5000 corruptions across 4 threads) was fixed
      upstream by `alloc.cyr`'s **v6.0.64** process-wide CAS spinlock around
      `alloc()`/`alloc_reset()`. Re-verified under cyrius 6.1.27 with a 4-thread
      × 200k-alloc stress: **0 corruption**. This unblocked 2.3.1 batch inference
      and the multi-threaded accept loop (now scheduled as **v2.4.0** above).
      Note: the global alloc lock serializes `alloc` (uncontended-cheap, but a
      hot path under heavy threading) — a future per-thread-arena optimization is
      still possible.
- [ ] Multi-threaded accept loop — see [v2.4.0](#v240--multi-threaded-accept-loop).
- [ ] Connection pooling for backend sockets — see v2.3.1 (deferred; sandhi-gated).

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
