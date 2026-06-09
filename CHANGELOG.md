# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [2.1.4] — 2026-06-09

Toolchain and dependency refresh. No API or behavior changes.

### Changed
- **Cyrius pin 6.0.57 → 6.1.18** (`cyrius.cyml`). Stdlib re-synced (`cyrius lib
  sync`); `cyrius.lock` refreshed. `cyrius fmt`/`lint`/`vet`/`deny` clean, 242
  tests pass, benchmark suite green under 6.1.18.
- **ai-hwaccel pin 2.3.7 → 2.3.9** — `dist/ai-hwaccel.cyr` bundle re-vendored
  (`cyrius deps`). The vendored `data/cloud_pricing.json` + `data/models.json`
  were re-checked against 2.3.9 and are content-unchanged (`models.json` stays a
  top-level array per the `hardware_data_files` guard).

## [2.1.3] — 2026-06-04

Optional durable persistence via the `patra` embedded SQL DB (stdlib). Opt-in and
fully backward compatible — without `[[storage]]`, hoosh runs in-memory exactly as
before.

### Added
- **`src/lib/storage.cyr`** (new) — patra-backed persistence for the HMAC audit
  chain and token-budget usage. Enabled by `[[storage]] path = "..."` in
  hoosh.toml; tables `audit` + `budgets` created on open.
- **Audit chain durability** — `audit_record` writes each entry through to disk
  (typed `patra_insert_row`, so messages with quotes/commas can't break or
  inject SQL); on startup the chain is rebuilt in id-order with `last_hash` +
  `next_id` restored so new entries continue the existing chain.
- **Token-budget durability** — `pool_commit` persists each pool's `used`;
  restored on startup. Verified end-to-end (`/v1/tokens/report` → restart →
  `used` restored).
- ADR [008-persistence-via-patra](docs/decisions/008-persistence-via-patra.md).
- `*.patra` added to `.gitignore`; commented `[[storage]]` example in hoosh.toml.

### Notes
- patra requires `fl_init()` + `patra_init()` before use — called in `main()`
  before opening storage.
- patra is single-threaded; storage access will need serialization when the
  threaded accept loop lands (next milestone).

## [2.1.2] — 2026-06-04

Structured operational logging via the `sakshi` stdlib module. Internal — no API
or response changes; the CLI surface is untouched.

### Added
- **Structured logging** (`src/lib/logging.cyr`, new) — leveled operational logs
  to **stderr** with timestamps, via sakshi. `hlog_info/warn/error/debug` cstr
  wrappers + `hlog_request(method, path)`. Log points: server startup, per
  request (`http_route`), auth rejections, config reload, chat "no provider"
  (warn) and "backend unreachable" (error), embeddings backend failure.
- **`[[logging]] level = ...`** in hoosh.toml (fatal/error/warn/info/debug/trace;
  default info) → `sakshi_set_level`. Parsed in `config.cyr`.
- Test group `logging_levels` (level-string mapping + set/get round-trip).

### Notes
- The CLI banner / `info` / `help` / `version` output stays on **stdout** as
  plain presentation; operational logs go to **stderr**, so piping stdout stays
  clean.
- `[[logging]]` uses the double-bracket table form because the TOML parser only
  honors `[[table]]` sections today (single-bracket support is a queued
  improvement) — consistent with `[[budgets]]`/`[[providers]]`.

## [2.1.1] — 2026-06-04

Surfaces ai-hwaccel 2.3.7 planning capabilities that the 2.1.0 dep upgrade pulled
in but didn't yet expose. Additive — existing endpoints unchanged.

### Added
- **`POST /v1/hardware/cost`** — cloud instance $/inference recommendations for a
  model size + quantization (ai-hwaccel `cost.cyr`; AWS/GCP/Azure).
- **`POST /v1/hardware/training-estimate`** — training-memory breakdown
  (model/optimizer/activation/total) for a model size + method
  (full/lora/qlora/dpo/…) + target (gpu/tpu/gaudi) (ai-hwaccel `training.cyr`).
- **`GET /v1/hardware/compatible-models`** — catalogue models that fit the
  detected accelerator memory at int8, with headroom % (ai-hwaccel `model.cyr`).
- **`data/cloud_pricing.json` + `data/models.json`** vendored from ai-hwaccel
  (read cwd-relative at runtime; cost/compatible-models degrade to empty if
  absent). `models.json` ships as a **top-level JSON array** — `load_models`
  scans for bare `{…}` objects, so the `{"models":[…]}` wrapper would yield only
  the first model. A test (`hardware_data_files`) guards this shape.

### Changed
- `src/lib/hardware.cyr` header refreshed for the 2.3.7 module set.

### Notes
- ai-hwaccel's threaded detector (`registry_detect_threaded`) was evaluated for
  faster startup but segfaults under hoosh's single-threaded runtime — deferred
  to the concurrency milestone. Startup still uses serial `registry_detect`.
- Still TODO on the 2.1.x line: `/v1/hardware/model-format` and
  `/v1/hardware/requirement-match` (ai-hwaccel `model_format.cyr` /
  `requirement.cyr`).

## [2.1.0] — 2026-06-04

Toolchain & scaffolding modernization to current Cyrius (6.0.x) conventions. No
gateway behavior changes; the binary builds, tests (231/231), and benchmarks
clean under the new pin. Two latent correctness fixes shipped along the way
(audit HMAC + config parsing — see Fixed).

### Changed
- **Cyrius toolchain pin 4.5.0 → 6.0.57.**
- **ai-hwaccel dependency 2.0.0 → 2.3.7**, now consumed as the single-file
  distlib bundle (`[deps.ai-hwaccel] modules = ["dist/ai-hwaccel.cyr"]`,
  vendored to `lib/ai-hwaccel.cyr` and `include`d from `src/main.cyr`) instead
  of the old per-source-module list.
- **Manifest `cyrius.toml` → `cyrius.cyml`** with `version = "${file:VERSION}"`
  interpolation (VERSION is the single source of truth) and a `repository` field.
- **Retired `.cyrius-toolchain`** — the pin now lives only in `cyrius.cyml`.
- **Syscalls go through the `sys_*` stdlib wrappers** (`sys_write`, `sys_read`,
  `sys_close`, `sys_socket`, `sys_connect`, `sys_exit`) instead of raw
  `syscall(N, …)` / bare `SYS_*` enum members (no longer global in 6.x).
- **stdlib deps** now list `ct`, `keccak`, `thread`, `thread_local` explicitly
  (split out of `sigil` / required by the ai-hwaccel bundle; Cyrius does not
  resolve transitive deps).
- **CI/release workflows modernized** — canonical installer reading the pin from
  `cyrius.cyml`, `cyrius lib sync` + `cyrius deps`, and hard `fmt`/`lint`/`vet`
  gates; release verifies tag == VERSION == `${file:VERSION}` and that the
  version is in this changelog.
- **Scripts de-Rusted** — `bench-history.sh` parses `cyrius bench` output (was
  `cargo bench`/criterion); `version-bump.sh` drives VERSION + CLAUDE.md +
  CHANGELOG (was `Cargo.toml`/`cargo generate-lockfile`).
- Whole tree formatted with `cyrius fmt`.

### Added
- **Benchmarks are now a hard, CI-enforced release gate** — CI runs
  `./scripts/bench-history.sh` and fails if the suite does not run or records no
  data (maintainer waiver via `CYRIUS_SKIP_BENCH=1`). Documented in CLAUDE.md.
- ADR [007-cyrius-6-modernization](docs/decisions/007-cyrius-6-modernization.md).

### Fixed
- **Audit chain HMAC** — replaced the removed `hmac_sign` with
  `hmac_sha256(...)` + `hex_encode` (new `_hmac_hex` helper in `audit.cyr`).
- **Config parsing under 6.x** — `toml_get_sections`/`toml_get` now take a
  **cstr** name; `config.cyr` was wrapping every lookup in `str_from(...)`,
  which silently parsed no sections. Stripped the wrappers (21 sites). Matching
  test drift (`vec_new(8)` arity, `ct_eq` → `ct_eq_bytes_lens`) fixed too.
- Stale hardcoded `"version":"2.0.0"` in the `/` response now tracks
  `HOOSH_VERSION`.

### Removed
- Rust-era cruft: `cyrius.toml`, `.cyrius-toolchain`, `tarpaulin-report.json`,
  `tarpaulin.toml`, and Rust/criterion entries in `.gitignore`.

## [2.0.0] — 2026-04-13

Complete rewrite from Rust to Cyrius. Binary drops from multi-MB to 636KB. All core gateway functionality preserved and ported.

### Added — Core Gateway
- **18 Cyrius modules** — types, ratelimit, route, router, budget, cache, metrics, auth, http_server, http_client, provider, compact, audit, retry, hardware, handlers, config, main
- **13 provider backends** — Ollama (native `/api/chat`), LlamaCPP, Synapse, LMStudio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter — all via OpenAI-compatible forwarding (Ollama uses native API)
- **SSE streaming** — `stream:true` in `/v1/chat/completions` proxies NDJSON (Ollama) or SSE (OpenAI-compat) from backend to client as OpenAI-format `chat.completion.chunk` events
- **Provider routing** — Priority, RoundRobin, LowestLatency strategies; model pattern matching with glob (`llama*`, `gpt-*`)
- **Token budget system** — named pools with capacity, reserve/commit lifecycle; `/v1/tokens/check`, `/v1/tokens/reserve`, `/v1/tokens/report`, `/v1/tokens/pools`
- **HMAC-SHA256 audit chain** — cryptographically linked log entries with tamper detection and verification; `/v1/audit` endpoint with chain validation
- **Retry with exponential backoff** — jittered delays (nanosecond clock bits for jitter), configurable max_retries/base_delay_ms/max_delay_ms via `[[retry]]` config section
- **Per-provider rate limiting** — RPM token bucket with continuous refill; `rate_limit` field in `[[providers]]` config
- **Response cache with LRU eviction** — timestamp-based access tracking, evict-oldest-on-full; hit/miss/eviction counters at `/v1/cache/stats`
- **Context compaction** — preserves system message, keeps recent N messages within token budget; runs before inference to prevent oversized requests
- **Bearer token auth** — constant-time comparison via sigil; skips `/v1/health` and `/metrics`
- **CORS** — full preflight handling on all endpoints

### Added — Hardware
- **ai-hwaccel 2.0.0 integration** — git tag dep (kybernet-style), 27 modules for hardware detection across 18 accelerator types (CUDA, ROCm, Metal, Vulkan, TPU, Gaudi, Neuron, Intel NPU, AMD XDNA, etc.)
- **`/v1/hardware`** — device summary JSON (count, memory, best device, all profiles)
- **`/v1/hardware/placement`** — model placement recommendation given model_params and quantization
- **`/v1/hardware/models`** — compatibility matrix for common model sizes (1B–405B) against detected hardware
- **Hardware on startup** — device count and best device shown in server banner and `hoosh info`

### Added — API Endpoints
- `POST /v1/chat/completions` — streaming + non-streaming inference
- `GET /v1/models` — list configured providers
- `GET /v1/health` — first provider connectivity check
- `GET /v1/health/providers` — per-provider health with TCP probe
- `GET /v1/health/heartbeat` — node status
- `POST /v1/embeddings` — routed through provider system (not hardcoded)
- `GET /v1/costs` — request/token counters per provider
- `POST /v1/costs/reset` — reset counters
- `GET /v1/cache/stats` — hit/miss/eviction stats
- `GET /v1/tokens/pools` — pool capacity/usage
- `GET /v1/queue/status` — queue depth
- `GET /v1/audit` — audit chain with verification
- `POST /v1/admin/reload` — hot-reload config
- `GET /v1/hardware`, `POST /v1/hardware/placement`, `GET /v1/hardware/models`
- `GET /metrics` — Prometheus format
- `GET /` — server info

### Added — CLI
- `hoosh serve [port]` — start gateway (default: 8088)
- `hoosh models` — list configured providers with URLs
- `hoosh health` — check provider connectivity
- `hoosh infer <model> <prompt>` — one-shot inference from CLI
- `hoosh info` — system info with hardware summary
- `hoosh help` / `hoosh version`

### Added — Configuration
- `hoosh.toml` with sections: `[[server]]`, `[[providers]]` (type, base_url, priority, models, api_key, rate_limit), `[[budgets]]`, `[[auth]]`, `[[retry]]`, `[[cache]]`
- `cyrius.toml` with `[package]`, `[build]`, `[deps]` (stdlib + ai-hwaccel git tag dep)

### Changed
- **Language**: Rust → Cyrius (cyrius 3.10.0)
- **Binary size**: multi-MB → 636KB
- **Dependencies**: 200+ crates → 29 Cyrius deps (stdlib + ai-hwaccel)
- **HTTP server**: axum/tokio → raw TCP sockets with syscalls
- **Build system**: cargo → `cyrius build`
- **Dep management**: Cargo.toml → cyrius.toml with git tag deps (kybernet-style)

### Removed
- Rust codebase (preserved in `rust-old/` for reference)
- axum, tokio, reqwest, serde, and all Rust dependencies
- Feature flags (all features compiled in)
- OpenTelemetry integration (deferred to v2.1)
- DLP content filtering (deferred to v2.1)
- TLS/mTLS support (blocked on Cyrius TLS lib)
- Audio endpoints (deferred to svara migration)
- Tool calling / MCP bridge (deferred to v2.1)
- Multi-threaded concurrency (single-threaded accept loop)

---

## Rust-era releases (pre-Cyrius port)

See `rust-old/` for source. These versions used Rust + axum + tokio.

- **1.2.0** (2026-04-03) — License change to GPL-3.0, binary size optimization, TLS provider decoupling
- **1.1.0** (2026-03-29) — GPU telemetry heartbeats, heartbeat eviction, majra ConcurrentPriorityQueue
- **1.0.0** (2026-03-27) — Context management, model metadata (63 models), semantic cache, retry manager, batch inference, cost optimizer, DLP scanner, multi-modal support, ai-hwaccel 1.0.0, 613 tests
- **0.23.4** (2026-03-23) — Tool use & MCP via bote/szál, model metadata registry, hot_path benchmarks
- **0.23.3** (2026-03-23) — Sentiment analysis via bhava
- **0.21.5** (2026-03-21) — Auth, rate limiting, TLS pinning, Prometheus, OpenTelemetry, audit chain, health checks, heartbeat, event bus, queue
- **0.21.3** (2026-03-21) — E2E benchmarks, connection tuning, HTTP/2, documentation
- **0.20.4** (2026-03-21) — Benchmark suite, CI/CD pipelines, version management
- **0.20.3** (2026-03-20) — Initial release: 14 backends, routing, caching, budgets, streaming, hardware placement, CLI, 185 tests
