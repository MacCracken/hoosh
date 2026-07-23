# Hoosh Architecture

> AI inference gateway — multi-provider LLM routing and token-budget management,
> written in **Cyrius** (single static binary, port 8088).
>
> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI. Part of the
> [AGNOS](https://github.com/MacCracken/agnosticos) ecosystem; ported from Rust
> (`rust-old/` kept for reference).

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
    ── audit.cyr            ── metrics.cyr          ── events.cyr   ── otlp.cyr
Cost accumulation per (provider, base_url) ── pricing.cyr

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
| `router.cyr` / `route.cyr` | Provider selection (4 strategies) + route records |
| `provider.cyr` | Forward to backends (local raw socket / remote sandhi), request shaping (Anthropic/Gemini) + response extraction + tool-call surfacing |
| `http_client.cyr` / `http_server.cyr` | Raw HTTP request/response, header building |
| `handlers.cyr` | Request handlers — chat (`_chat_prep`/`_chat_forward`/`_chat_assemble`), streaming, health, models, tokens, cost, metrics |
| `pool.cyr` | Unified worker pool (v2.4.0) — work-queue ring + 7 banked workers; the accept loop enqueues, workers run every request concurrently |
| `batch.cyr` | `/v1/batch` sync + async — items enqueued as pool jobs; batch registry |
| `mcp.cyr` | `/v1/tools/*` via the vendored bote MCP core |
| `events.cyr` | Provider event bus (majra pubsub) + recent-events ring |
| `otlp.cyr` | OpenTelemetry OTLP/JSON span export (opt-in) — local `http` + remote `https` (worker-routed TLS) |
| `trace.cyr` | W3C `traceparent` propagation (thread-local carrier) |
| `metrics.cyr` | Prometheus counters + per-provider latency histograms |
| `cache.cyr` / `semantic.cyr` | Exact-key LRU cache with TTL expiry + semantic (embedding cosine) cache |
| `budget.cyr` | Named token pools (reserve → commit/release) |
| `ratelimit.cyr` | Per-provider RPM token bucket (`hoosh_ratelimit_*` — majra exports colliding names) |
| `audit.cyr` | HMAC-SHA256 tamper-proof audit chain |
| `dlp.cyr` | PII/secret scanner + privacy-aware routing |
| `compact.cyr` / `compression.cyr` | Context compaction + whitespace/tool-pair compression |
| `pricing.cyr` / `metadata.cyr` | Pricing table, per-(provider, base_url) cost accumulation, cost optimizer (cheapest capable model), 34-entry model catalog |
| `hardware.cyr` | ai-hwaccel planning endpoints (placement, cost, training estimate, model-format, requirement-match, simulate, telemetry); available-VRAM accounting + periodic re-detection |
| `config.cyr` | `hoosh.cyml` (TOML) parsing + `$ENV` key expansion |
| `storage.cyr` | Optional patra persistence (audit chain + budgets) |
| `retry.cyr` | Jittered exponential backoff, gated on retryability (permanent 4xx are not retried) |
| `health.cyr` | Background provider probing + per-route circuit breaker; `router_select` routes around unhealthy backends |
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
| `/v1/models` | GET | List configured providers |
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
