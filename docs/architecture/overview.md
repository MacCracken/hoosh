# Hoosh Architecture

> AI inference gateway — multi-provider LLM routing and token-budget management,
> written in **Cyrius** (single static binary, port 8088).
>
> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI. Part of the
> [AGNOS](https://github.com/MacCracken/agnosticos) ecosystem; ported from Rust
> (the Rust tree, `rust-old/`, was removed in 2.9.2; it stays in git history at tag 2.7.1).

---

## Design Principles

1. **Provider-agnostic** — uniform OpenAI-compatible API across 17 backends
   (local + remote); model-pattern routing picks the backend.
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
    ├─ Trace context (traceparent, thread-local) ── trace.cyr
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

(Whisper STT is present as a provider but its audio pipeline is migrating to
**svara**; see the roadmap's Deferred section.)

---

## Module Structure (`src/`)

`main.cyr` — CLI (`serve`/`models`/`health`/`infer`/`info`/`version`, with flags and
`--server` remote mode), HTTP route dispatch, signal handling (SIGINT/SIGTERM
shutdown, SIGHUP reload), and the `cmd_serve` accept loop + startup init.

| Module | Role |
|--------|------|
| `types.cyr` | Provider enum (`PROV_*`), `provider_name`, error codes, `HOOSH_VERSION` |
| `router.cyr` / `route.cyr` | Provider selection (4 strategies), the first enabled route of a provider type (2.9.21) + route records |
| `jsonlite.cyr` | Byte-level JSON field readers (2.6.5): value position, string/int extraction, object and span ends, raw values, JSON escaping (2.9.22 gathered the last three here) |
| `provider.cyr` | Forward to backends (local raw socket / remote sandhi), request shaping (Anthropic/Gemini) + response extraction + tool-call surfacing; Ollama's embeddings answer as OpenAI's list (2.9.22) |
| `http_client.cyr` / `http_server.cyr` | Raw HTTP request/response, header building; a local stream's keep-alive step (`sse_idle_next`, 2.9.22) |
| `handlers.cyr` | Request handlers — chat (`_chat_prep`/`_chat_forward`/`_chat_assemble`), streaming, health, models, tokens, cost, metrics |
| `models.cyr` | The model listings (2.9.22 split from `handlers.cyr`): `/v1/models`' body (`models_list_json`), one entry per id, a local backend's own list, the catalog and table fallbacks, `_cat_literal` |
| `reqparse.cyr` | Pure request helpers (2.9.16/17 splits): body fields, the cache key and warming body, the hardware inputs, `/v1/batch`'s body, the POST-only path list (2.9.21, from `main.cyr`), `add_devices`' sizes (2.9.22) |
| `wq.cyr` | The work queue (2.9.20 split from `pool.cyr`): a bounded ring of job pointers under one mutex, the in-flight count the shutdown drain waits on, the job constructors, and `/v1/queue/status`'s body (2.9.26) |
| `pool.cyr` | Unified worker pool (v2.4.0) — 7 workers draining `wq.cyr`'s queue (crypto lanes come from sigil's pool of 64 since 2.9.1); the accept loop enqueues, workers run every request concurrently |
| `batch.cyr` | `/v1/batch` sync + async — items enqueued as pool jobs; batch registry |
| `mcp.cyr` | `/v1/tools/*` via the vendored bote MCP core |
| `events.cyr` | Provider event bus (majra pubsub) + recent-events ring |
| `otlp.cyr` | OpenTelemetry OTLP/JSON spans (opt-in), the pure half: settings, the span ring, span and batch JSON |
| `otlp_export.cyr` | The I/O half (2.9.14 split): the exporter thread and the POST — local `http` + remote `https` (worker-routed TLS) |
| `trace.cyr` | W3C `traceparent` propagation (thread-local carrier) |
| `metrics.cyr` | Prometheus counters + per-provider latency histograms; the `/metrics` body (`metrics_render`, 2.9.21) |
| `cache.cyr` / `semantic.cyr` | Exact-key LRU cache with TTL expiry + semantic (embedding cosine) cache |
| `budget.cyr` | Named token pools (reserve → commit/release) |
| `ratelimit.cyr` | Per-provider RPM token bucket (`hoosh_ratelimit_*` — majra exports colliding names) |
| `audit.cyr` | HMAC-SHA256 tamper-proof audit chain |
| `dlp.cyr` | PII/secret scanner + privacy-aware routing |
| `compact.cyr` / `compression.cyr` | Context compaction + whitespace/tool-pair compression |
| `catalog.cyr` / `catalog_fetch.cyr` | The live model catalog (2.8.0, [ADR 012](../decisions/012-live-model-catalog.md)): per-provider list parsers, the published per-route snapshot, `[[models]]` operator entries, the per-model thinking plan and output ceiling, and since 2.9.25 an index over each published list (exact id and dated alias) so lookups do not scan (pure; tested for real) — and the refresh that asks each live route on a banked pool worker |
| `urlparse.cyr` | Scheme-aware host / port of a base url (2.8.0 — the remote health probe had been reading `https://` urls as `localhost:80`) |
| `provider_err.cyr` | The provider's own error kept and relayed (2.6.5), and the OpenAI envelope: completion id and head, `chat_rehead`, the `chat.completion` body (`_chat_completion_body`, 2.9.21 from `handlers.cyr`) |
| `outcome.cyr` | Finish reasons mapped from each provider's words, stream usage (Anthropic `message_*`, OpenAI `usage`, Gemini `usageMetadata`, Ollama's final line), the Anthropic tool-fragment guard |
| `keyindex.cyr` | An index over a compiled table of key-first rows (2.9.25): the exact key through a map, then the longest prefix by precomputed key lengths; pricing and metadata look up through it |
| `pricing.cyr` / `metadata.cyr` | Pricing table (with long-context tiers), per-(provider, base_url) cost accumulation, cost optimizer (cheapest capable model), the compiled model table — the catalog's fallback, refreshed 2026-10-09 |
| `hardware.cyr` | ai-hwaccel planning endpoints (placement, cost, training estimate, model-format, requirement-match, simulate, telemetry); available-VRAM accounting + periodic re-detection |
| `config.cyr` | `hoosh.cyml` (TOML) loading: `load_config` and the per-section readers |
| `cfgparse.cyr` | config.cyr's self-contained helpers (2.9.18 split): strategy names, `$ENV` expansion, TOML string lists (read with bayan's array parser since 2.9.23), a provider's TLS policy and rate limit |
| `cli.cyr` | The command line's flag parser (2.9.19 split from `main.cyr`): `--name value`, `-s value`, `--name=value`, positionals, `infer`'s model and prompt, over an argument vec |
| `storage.cyr` | Optional patra persistence (audit chain + budgets) |
| `retry.cyr` | Jittered exponential backoff, gated on retryability (permanent 4xx are not retried) |
| `health.cyr` | Background provider probing + per-route circuit breaker; `router_select` routes around unhealthy backends; `/v1/health`'s verdict and body (2.9.21) |
| `auth.cyr` | Bearer-token auth (constant-time) |
| `logging.cyr` | Structured operational logging (sakshi) + the coarse clock ticker (`hoosh_now_ms`) |

`src/vendor/` — committed single-file distlib bundles: `bote-core.cyr` (MCP
JSON-RPC core) and `majra.cyr` (pub/sub). Vendored rather than `[deps]` because
their manifests pull colliding transitive git deps; see
[ADR 005](../decisions/005-mcp-via-bote.md) / [ADR 010](../decisions/010-observability.md).

---

## Dependencies

No non-AGNOS dependencies. The gateway is built from the cyrius stdlib plus AGNOS
distlibs:

| Dependency | Role | Consumed as |
|-----------|------|-------------|
| **ai-hwaccel** | Hardware detection for model placement / cost | `[deps.ai-hwaccel]` → `lib/ai-hwaccel.cyr` |
| **bote** | MCP JSON-RPC 2.0 core (registry/dispatcher/codec) | vendored `src/vendor/bote-core.cyr` |
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
