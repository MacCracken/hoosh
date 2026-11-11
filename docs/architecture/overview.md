# Hoosh Architecture

> AI inference gateway — multi-provider LLM routing and token-budget management,
> written in **Cyrius** (single static binary, port 8088).
>
> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI. Part of the
> [AGNOS](https://github.com/MacCracken/agnosticos) ecosystem; ported from Rust
> (the Rust tree, `rust-old/`, was removed in 2.9.2; it stays in git history at tag 2.7.1).

---

## Design Principles

1. **Provider-agnostic** — uniform OpenAI-compatible API across 18 backends
   (local, in-process + remote); model-pattern routing picks the backend.
2. **Local-first** — prefer on-device inference; remote APIs as fallback.
3. **Token-aware** — every request tracked against per-agent budgets.
4. **No magic** — every operation measurable, auditable, traceable.
5. **Lean** — single static binary, no external (non-AGNOS) dependencies.

---

## System Architecture

```
Clients (curl, daimon, AGNOS consumer apps)
    │  HTTP (OpenAI-compatible)
    ▼
Accept loop (accept + enqueue) → unified 7-worker pool ── main.cyr / pool.cyr
    │
    ├─ Auth (bearer, constant-time) ── auth.cyr
    ├─ Trace context (2.10.3: hoosh's span is the caller's child) ── trace.cyr
    ├─ Request arena (per thread, released at each job's start) ── reqmem.cyr
    ▼
Router (priority │ round-robin │ lowest-latency │ direct) ── router.cyr / route.cyr
    │  health-filtered: unhealthy routes withdrawn ── health.cyr
    │  hardware-aware: local deprioritized when the model exceeds free VRAM
    │
    ├─ Request validation + sampling params (temperature/top_p/max_tokens)
    ├─ DLP scan + privacy routing ── dlp.cyr
    ├─ Response cache (exact LRU + semantic) ── cache.cyr / semantic.cyr
    ├─ Rate limiter (per-provider RPM) ── ratelimit.cyr
    ├─ Token budget (reserve/commit) ── budget.cyr
    ├─ Compaction + compression ── compact.cyr / compression.cyr
    ▼
Provider forward ── provider.cyr
    ├─ Local: raw socket → host:port, bounded connect/IO timeouts (Ollama, llama.cpp, Synapse, LM Studio, LocalAI, vLLM, TensorRT-LLM, ONNX)
    ├─ In-process (2.11.0): a packed ternary model, decoded through tentib ── tentib_local.cyr (ADR 014)
    └─ Remote: sandhi HTTPS → (OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter)
    │
    ▼
Audit chain (HMAC-SHA256) │ Metrics (Prometheus) │ Events (majra) │ OTLP spans
    ── audit.cyr            ── metrics.cyr          ── events.cyr   ── otlp.cyr + otlp_export.cyr
Cost accumulation per (provider, base_url) ── pricing.cyr
   └─ 2.5.12: cost_record returns the micro-USD it accumulated; the chat
      response reports that same value as usage.cost_micro_usd, so a client's
      per-response sum reconciles with /v1/costs by construction

Background threads: health prober │ hardware re-detect │ coarse clock │ OTLP
                    export │ signal handler (SIGINT/SIGTERM/SIGHUP)

ai-hwaccel: hardware detection for placement / simulate / telemetry / cost
```

Speech (2.12.2, [ADR 016](../decisions/016-in-process-speech-through-svara.md)) takes its own path:

```
POST /v1/audio/speech ── handlers.cyr handle_audio_speech
    ├─ router_select_speech: svara routes only (a svara route serves no text) ── router.cyr
    ├─ DLP, the route's rate limit ── dlp.cyr / ratelimit.cyr
    ▼
svara_local.cyr ── one lock around every call into shabda or svara
    ├─ shabda: text → phoneme events (English G2P, one engine per process)
    ├─ the plan, in the request arena: svara's durations, nasalization, crossfades, the reply's length
    ├─ svara: each phoneme on a fresh synthesis context, over the thread's sample buffer
    └─ the crossfade, streamed: 16-bit PCM written 8192 samples at a time (WAV or raw, 24 kHz)
```

(The `whisper` provider type exists, but no endpoint uses it: AGNOS has no speech-recognition
engine in Cyrius yet.)

---

## Module Structure (`src/`)

`main.cyr` — CLI (`serve`/`models`/`health`/`infer`/`info`/`version`, with flags and
`--server` remote mode), HTTP route dispatch, signal handling (SIGINT/SIGTERM
shutdown, SIGHUP reload), and the `cmd_serve` accept loop + startup init.

| Module | Role |
|--------|------|
| `types.cyr` | Provider enum (`PROV_*`), `provider_name`, error codes, `HOOSH_VERSION` |
| `router.cyr` / `route.cyr` | Provider selection (4 strategies) over the routes that serve text (2.12.2: a second array, so a speech route costs text selection nothing), speech selection (`router_select_speech`), the first enabled route of a provider type (2.9.21) + route records |
| `jsonlite.cyr` | Byte-level JSON field readers (2.6.5): value position, string/int extraction, object and span ends, raw values, JSON escaping (2.9.22 gathered the last three here) |
| `provider.cyr` | Forward to backends (local raw socket / remote sandhi), request shaping (Anthropic/Gemini) + response extraction + tool-call surfacing; Ollama's embeddings answer as OpenAI's list (2.9.22) |
| `http_client.cyr` / `http_server.cyr` | Raw HTTP request/response, header building; per-thread scratch blocks for local responses and local streams (2.10.0); a local stream's keep-alive step (`sse_idle_next`, 2.9.22); the error envelope's writers and the deferred response descriptor (`_resp_*`, 2.10.2), and a response head on the stack |
| `handlers.cyr` | Request handlers — chat (`_chat_prep`/`_chat_forward`/`_chat_assemble`), streaming, speech (2.12.2), health, models, tokens, cost, metrics |
| `models.cyr` | The model listings (2.9.22 split from `handlers.cyr`): `/v1/models`' body (`models_list_json_a`), one entry per id, a local backend's own list, the catalog and table fallbacks, `_cat_literal`; since 2.10.2 `/v1/models/catalog`'s body too, built once per change and held (`models_catalog_body`, [ADR 013](../decisions/013-request-memory.md)) |
| `reqparse.cyr` | Pure request helpers (2.9.16/17 splits): body fields, the cache key and warming body, the hardware inputs, `/v1/batch`'s body, the POST-only path list (2.9.21, from `main.cyr`), `add_devices`' sizes (2.9.22) |
| `wq.cyr` | The work queue (2.9.20 split from `pool.cyr`): a bounded ring of job pointers under one mutex, the in-flight count the shutdown drain waits on, the job constructors, `/v1/queue/status`'s body (2.9.26), and `job_begin` (2.10.3: each job's trace context and a released request arena) |
| `pool.cyr` | Unified worker pool (v2.4.0; on a target with no second thread the accept loop serves instead, [ADR 011](../decisions/011-multithreaded-accept-loop.md)'s 2.10.1 addendum) — 7 workers draining `wq.cyr`'s queue (crypto lanes come from sigil's pool of 64 since 2.9.1); the accept loop enqueues, workers run every request concurrently |
| `batch.cyr` | `/v1/batch` sync + async — items enqueued as pool jobs; batch registry |
| `mcp.cyr` | `/v1/tools/*` via the vendored bote MCP core: `bote_echo`, and since 2.12.1 szál's tool groups that `[mcp] tools` allows (the host-reaching ones only with `[[auth]]` tokens), dispatched one call at a time on the request arena |
| `events.cyr` | Provider event bus (majra pubsub) + recent-events ring: 128 slots each event is copied into and reused as the ring wraps, events built on the stack (2.10.3) |
| `otlp.cyr` | OpenTelemetry OTLP/JSON spans (opt-in), the pure half: settings, the span ring (slots reused as it wraps since 2.11.2), span and batch JSON |
| `otlp_export.cyr` | The I/O half (2.9.14 split): the exporter thread and the POST — local `http` + remote `https` (worker-routed TLS) |
| `trace.cyr` | W3C `traceparent` propagation: per job, the caller's context and hoosh's own under it, in one block per thread; ids from a keyed permutation of a counter (2.10.3, [ADR 010](../decisions/010-observability.md)'s update) |
| `reqmem.cyr` | The request arena (2.10.3, [ADR 013](../decisions/013-request-memory.md)): one growable arena per thread, released when a job starts; builders with room in it; `mem_keep`, the copy a keeper stores |
| `keep.cyr` | The keep pool (2.11.2, [ADR 013](../decisions/013-request-memory.md) decision 3): size classes, four per power of two from 32 bytes to 1 MiB, that a bounded keeper frees into and allocates from again; the response cache's keys and values and the audit chain's model copies recycle through it |
| `metrics.cyr` | Prometheus counters + per-provider latency histograms; the `/metrics` body (`metrics_render`, 2.9.21) |
| `cache.cyr` / `semantic.cyr` | Exact-key LRU cache with TTL expiry, evicting in O(1) through two lists in each entry (2.10.0), and recycling what it evicts: entries on a free list, its copies of keys and values through its keep pool (2.11.2) + semantic (embedding cosine) cache |
| `budget.cyr` | Named token pools (reserve → commit/release) |
| `ratelimit.cyr` | Per-provider RPM token bucket (`hoosh_ratelimit_*` — majra exports colliding names) |
| `audit.cyr` | HMAC-SHA256 tamper-proof audit chain; since 2.11.2 its entries hold their hashes, evicted entries are reused, and a lock of its own covers records and `GET /v1/audit`'s reads |
| `dlp.cyr` | PII/secret scanner + privacy-aware routing |
| `compact.cyr` / `compression.cyr` | Context compaction + whitespace/tool-pair compression |
| `catalog.cyr` / `catalog_fetch.cyr` | The live model catalog (2.8.0, [ADR 012](../decisions/012-live-model-catalog.md)): per-provider list parsers (each saying `kind`, tools and vision where its provider does, 2.10.2), the published per-route snapshot, `[[models]]` operator entries, the per-model thinking plan and output ceiling, and since 2.9.25 an index over each published list (exact id and dated alias) so lookups do not scan (pure; tested for real) — and the refresh that asks each live route on a banked pool worker |
| `urlparse.cyr` | Scheme-aware host / port of a base url (2.8.0 — the remote health probe had been reading `https://` urls as `localhost:80`) |
| `provider_err.cyr` | The provider's own error kept and relayed (2.6.5), and the OpenAI envelope: completion id and head, `chat_rehead`, the `chat.completion` body (`_chat_completion_body`, 2.9.21 from `handlers.cyr`), and a stream's chunks from one reused buffer per thread (`sse_delta_send`, 2.10.3) |
| `outcome.cyr` | Finish reasons mapped from each provider's words, stream usage (Anthropic `message_*`, OpenAI `usage`, Gemini `usageMetadata`, Ollama's final line), the Anthropic tool-fragment guard |
| `keyindex.cyr` | An index over a compiled table of key-first rows (2.9.25): the exact key through a map, then the longest prefix by precomputed key lengths; pricing and metadata look up through it |
| `pricing.cyr` / `metadata.cyr` | Pricing table (with long-context tiers), per-(provider, base_url) cost accumulation, cost optimizer (cheapest capable model), the compiled model table — the catalog's fallback, refreshed 2026-10-09, and what `metadata_lend` fills into what an entry does not say (2.10.2) |
| `hardware.cyr` | ai-hwaccel planning endpoints (placement, cost, training estimate, model-format, requirement-match, simulate, telemetry); available-VRAM accounting + periodic re-detection |
| `config.cyr` | `hoosh.cyml` (TOML) loading: `load_config` and the per-section readers |
| `tentib_local.cyr` | In-process ternary inference (2.11.0, [ADR 014](../decisions/014-in-process-ternary-inference.md)): a `tentib` route's packed model loaded through tentib's library bundle (bounded, header-peeked, validated, skipped when unchanged, a same-shape reload reusing the model's memory since 2.11.1, loaded for serving only since 2.11.2), byte-token greedy decoding one locked step at a time (one cached decode step per token until the window slides, since 2.11.1), buffered replies and per-character streams, UTF-8 assembly with U+FFFD |
| `svara_local.cyr` | In-process speech (2.12.2, [ADR 016](../decisions/016-in-process-speech-through-svara.md)): the request's fields, the English G2P engine and three voices (built once), the plan in the request arena (svara's durations, nasalization and crossfades), the render on a fresh synthesis context per request with svara's crossfade streamed out as 16-bit PCM, the WAV header; one lock around every call into shabda or svara |
| `cfgparse.cyr` | config.cyr's self-contained helpers (2.9.18 split): strategy names, `$ENV` expansion, TOML string lists (read with bayan's array parser since 2.9.23), a provider's TLS policy and rate limit |
| `cli.cyr` | The command line through cmdit (2.12.0, [ADR 015](../decisions/015-command-line-through-cmdit.md)): the global and per-verb handles, and `cli_parse_argv`, a pure parse of the whole argv into a record `main()` acts on (help, version and usage errors before setup; exit 2 for a usage error) |
| `storage.cyr` | Optional patra persistence (audit chain + budgets) |
| `retry.cyr` | Jittered exponential backoff, gated on retryability (permanent 4xx are not retried) |
| `health.cyr` | Background provider probing + per-route circuit breaker; `router_select` routes around unhealthy backends; `/v1/health`'s verdict and body (2.9.21) |
| `auth.cyr` | Bearer-token auth (constant-time) |
| `logging.cyr` | Structured operational logging (sakshi) + the coarse clock ticker (`hoosh_now_ms`) |

`src/vendor/` — committed single-file distlib bundles: `bote-core.cyr` (MCP
JSON-RPC core), `majra.cyr` (pub/sub) and `szal-mcp.cyr` (szál's MCP tools, 2.12.1). Vendored rather than `[deps]` because
their manifests pull colliding transitive git deps; see
[ADR 005](../decisions/005-mcp-via-bote.md) / [ADR 010](../decisions/010-observability.md).

---

## Dependencies

No non-AGNOS dependencies. The gateway is built from the cyrius stdlib plus AGNOS
distlibs:

| Dependency | Role | Consumed as |
|-----------|------|-------------|
| **ai-hwaccel** | Hardware detection for model placement / cost | `[deps.ai-hwaccel]` → `lib/ai-hwaccel.cyr` |
| **tentib** | In-process ternary (1.58-bit) inference: the packed-model loader and kernel (2.11.0), its incremental decoder (1.4.0, since 2.11.1), and its serving-only load (1.5.0, since 2.11.2) | `[deps.tentib]` → `lib/tentib.cyr`, with its `lib/rosnet.cyr` + `lib/tyche.cyr` |
| **shabda**, **shabdakosh**, **varna** | Speech (2.12.2): text to svara phoneme events, through the dictionaries and the phoneme inventories | `[deps.shabda]`, `[deps.shabdakosh]`, `[deps.varna]` → `lib/*.cyr` |
| **svara**, **hisab**, **naad**, **goonj** | Speech (2.12.2): formant synthesis, and its maths and DSP | `[deps.svara]`, `[deps.hisab]`, `[deps.naad]`, `[deps.goonj]` → `lib/*.cyr` |
| **cmdit** | The command line (2.12.0): getopt-long flags, verb dispatch, generated per-verb help, cmdit's exit codes | `[deps.cmdit]` → `lib/cmdit.cyr` |
| **bote** | MCP JSON-RPC 2.0 core (registry/dispatcher/codec) | vendored `src/vendor/bote-core.cyr` |
| **szál** | MCP tools (2.12.1): 54 tools in 15 groups, registered into hoosh's own dispatcher | vendored `src/vendor/szal-mcp.cyr` (`scripts/sync-szal.sh`); szál 2.3.3 since 2.12.3, on AGNOS too |
| **majra** | Provider event bus (pub/sub) | vendored `src/vendor/majra.cyr` |
| cyrius **sandhi** | HTTPS client (URL parse + DNS + TLS) for remote providers | stdlib |
| cyrius **sigil** | sha256 / HMAC / native TLS | stdlib |
| cyrius **patra** | Embedded persistence (opt-in) | stdlib |
| cyrius **sakshi** | Structured logging | stdlib |
| cyrius **bayan** | JSON / TOML parsing | stdlib |

---

## API Endpoints (server mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Inference (streaming + non-streaming) |
| `/v1/batch` | POST | Batch inference (sync; `"async":true` → job id) |
| `/v1/batch/{id}` | GET | Async batch progress |
| `/v1/batch/{id}/cancel` | POST | Cancel an async batch |
| `/v1/events/recent` | GET | Recent provider events (majra pub/sub bus) |
| `/v1/models` | GET | Model ids across configured providers (the live catalog; local backends asked live) |
| `/v1/models/catalog` | GET | The model catalog: ids per provider with routes, limits, capabilities, known prices ([ADR 012](../decisions/012-live-model-catalog.md)) |
| `/v1/models/refresh` | POST | Ask every live route for its models now |
| `/api/tags` | GET | List models (native Ollama-compatible shape) |
| `/v1/models/pull` / `/v1/models/delete` | POST | Pull / delete a model (Ollama) |
| `/v1/training/status` / `/v1/catalog/sync` | POST | Synapse training status / catalog sync |
| `/v1/health` / `/v1/health/providers` | GET | Gateway / per-provider health, failure counts |
| `/v1/tokens/check` / `reserve` / `report` | POST | Token budget lifecycle |
| `/v1/tokens/pools` | GET | List token pools |
| `/v1/cost/estimate` / `/v1/cost/recommend` | POST | Per-token cost / cheapest capable model |
| `/v1/cache/stats` | GET | Cache hit/miss/eviction stats |
| `/v1/tools/list` / `/v1/tools/call` | GET / POST | MCP tool server (bote) |
| `/v1/costs` / `/v1/costs/reset` | GET / POST | Per-provider spend; audited reset |
| `/v1/audit` | GET | Audit chain + verification status (opt-in `[audit]`) |
| `/v1/hardware*` | GET / POST | Detection, placement, simulate, telemetry, cost, model-format |
| `/metrics` | GET | Prometheus metrics (+ per-provider latency histograms) |
| `/v1/admin/reload` | POST | Hot-reload config (also on `SIGHUP`) |

---

## Consumers

| Project | Usage |
|---------|-------|
| **[AGNOS](https://github.com/MacCracken/agnosticos)** | System-wide inference gateway |
| **[daimon](https://github.com/MacCracken/daimon)** | Inference-routing runtime |
| All AGNOS consumer apps | Via daimon or direct OpenAI-compatible HTTP |
