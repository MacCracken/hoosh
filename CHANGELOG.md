# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.2.1] — 2026-06-09

Provider correctness & completeness — the **v2.2.1 parity items**. Restores the
per-provider token estimation lost in the port, and adds the provider lifecycle
and native-Ollama surface the Rust gateway exposed.

### Added
- **Provider lifecycle endpoints** (`src/lib/handlers.cyr`) — restore the Rust-era
  `ollama.pull_model`/`delete_model` and `synapse.training_status`/`sync_catalog`:
  - `POST /v1/models/pull` `{"model":"..."}` → forwards `POST /api/pull`
    `{"name":"..."}` to the configured Ollama backend.
  - `POST /v1/models/delete` `{"model":"..."}` → forwards `DELETE /api/delete`
    `{"name":"..."}` to Ollama.
  - `POST /v1/training/status` `{"job_id":"..."}` → forwards `GET
    /v1/training/<job>` to the Synapse backend.
  - `POST /v1/catalog/sync` → forwards `POST /v1/catalog/sync` to Synapse.

  Each resolves the target via `_router_find_provider` (first enabled route of
  the provider type), returns `404` when that provider is not configured, and
  `502` when the backend is unreachable or errors. **Live-verified** against a
  mock backend: pull/delete forward the correct method, path, and body.
- **Native Ollama `/api/tags` inbound route** (`handle_ollama_tags`) — lists each
  enabled route's model patterns in Ollama's tags shape
  (`{"models":[{"name","model","modified_at","size","digest","details"}]}`), so
  Ollama clients pointed at the gateway can enumerate available models.
- **Generic local HTTP client** (`http_req_local`, `_build_req_header` in
  `src/lib/http_client.cyr`) — arbitrary method (GET/POST/DELETE) over the
  loopback fast path, with bodyless requests supported. `http_post_local` is now
  a thin `POST` wrapper (no behaviour change for existing callers).

### Changed
- **Per-provider token estimation** (`src/lib/compact.cyr`) — context compaction
  no longer hardcodes 4 chars/token for every provider. Restores
  `ProviderTokenCounter::for_provider` from the Rust `context/tokens.rs`: ratios
  are carried scaled by 10 (no floats) — OpenAI-family 3.8, Anthropic 3.5 (denser
  tokenizer), local LLaMA-family 3.7, others a conservative 4.0. `compact_messages`
  takes the target provider's ratio (via `chars_per_token_x10(route_provider)`),
  so the budget math matches how the destination actually tokenizes. Unknown
  providers keep the prior 4.0 default.

### Tests
- `+20` assertions (**298 pass**): per-provider ratio table + `estimate_tokens`
  (incl. denser-provider-estimates-higher and zero-ratio fallback), lifecycle
  route lookup (found/missing/disabled), and generic GET/DELETE request headers.

## [2.2.0] — 2026-06-09

Remote provider transport — the **v2.2.0 criticals**. Cloud providers were enum +
default-URL entries the loopback-only client could never reach; they now forward
over TLS. All families work end-to-end: OpenAI-compatible (OpenAI, DeepSeek,
Mistral, Groq, Grok, OpenRouter), Anthropic, and Google/Gemini. The sandhi P1
that blocked production is fixed upstream; streaming is now incremental.

### Added
- **Remote forward path** (`src/lib/provider.cyr`) — `https://` routes forward
  through **sandhi**'s high-level client (`sandhi_http_post`: URL parse + DNS +
  TLS + connect), scheme-dispatched by `route_is_remote`. Local `http://`
  backends keep the raw-socket fast path unchanged.
- **Provider auth headers** (`_provider_headers`) — `Authorization: Bearer` for
  OpenAI-compat; `x-api-key` + `anthropic-version` for Anthropic — built from the
  route's `api_key` (loaded from `[providers].api_key` in hoosh.toml).
- **Anthropic `/v1/messages` shaping** (`_build_anthropic_body`,
  `anthropic_extract_text`, `extract_anthropic_tokens`) — `max_tokens` body,
  `content[].text` + `usage.{input,output}_tokens` extraction. **Live-verified
  against the real Anthropic API** (single request returns correctly).
- **Token extraction** for OpenAI `usage` (`extract_openai_tokens`) and Anthropic
  usage; Ollama's `*_eval_count` fields fall back.
- **`$ENV` key expansion** in config (`_config_expand_env`, `src/lib/config.cyr`)
  — `api_key = "$ANTHROPIC_API_KEY"` resolves from the environment, so the secret
  never lives in hoosh.toml. The Anthropic provider block in `hoosh.toml` is
  enabled (key from env).
- **Anthropic system-message hoist** (`_build_anthropic_body`, with byte-level
  JSON helpers `_json_obj_str_field` / `_obj_is_system` / `_json_obj_end`) —
  `role:"system"` turns lift to the top-level `system` field (multiple joined
  with `\n`); remaining `{role,content}` turns pass through unchanged. Anthropic
  rejects a system role inside `messages`, so this is required for system prompts.
- **Google/Gemini shaping** (`_build_gemini_body`, `_gemini_url`,
  `gemini_extract_text`, `extract_gemini_tokens`) — maps OpenAI-style messages to
  Gemini `contents` (assistant→`model`, system→`systemInstruction`), builds the
  model-scoped `:generateContent?key=` URL (key as a **query param**, no auth
  header — `_provider_headers` skips `Authorization` for Google), and extracts
  `candidates[].content.parts[].text` + `usageMetadata.{prompt,candidates}TokenCount`.
- **Incremental remote streaming** — `handle_chat_stream`'s remote branch drives
  `sandhi_http_stream` with a per-event SSE callback (`_remote_stream_cb`),
  decoding each provider's delta (OpenAI `choices[].delta.content`, Anthropic
  `content_block_delta.delta.text`, Gemini `:streamGenerateContent?alt=sse`) and
  re-emitting them as OpenAI SSE chunks. Replaces the buffered one-shot fallback,
  which is retained only for the error-before-first-byte case (so a stream that
  fails early degrades to buffered without duplicating output).
- `tls`, `sandhi`, `mmap`, `dynlib`, `fdlopen` added to `cyrius.cyml` `[deps]`
  (in include order). No `main()` init needed — sandhi/tls self-initialize.
- Tests: `remote_transport` group (scheme dispatch, path/URL/bearer building,
  OpenAI + Anthropic + Gemini token/text extraction, system-message hoist, stream
  request shaping, SSE delta extraction) — **269 tests pass**.

### Changed
- **Cyrius pin 6.1.18 → 6.1.20** (`cyrius.cyml`); stdlib re-synced, `cyrius.lock`
  refreshed (sandhi 1.4.4 → **1.4.5**). This ships the fix for the P1 below.
- **Build the gateway with `-D CYRIUS_TLS_NATIVE`** (CI + release workflows;
  flag precedes the source). Compiles in sandhi's native TLS stack so the binary
  never fdlopen-loads libssl/libcrypto/glibc. `main()` now calls
  `sandhi_tls_use_native()` at startup and prints a loud stderr WARNING if the
  native backend is not active (so a dropped flag can't silently regress to the
  crash-prone libssl path). See CLAUDE.md Key Principles + sandhi architecture/004.

### Fixed
- ✅ **Whitespace-tolerant response extraction — OpenAI & Gemini non-streaming
  responses were silently dropped.** Live verification revealed the text/token
  extractors (`ollama_extract_text`, `anthropic_extract_text`, and the
  `extract_*_tokens` scanners) matched compact-only needles (`"content":"`,
  `"prompt_tokens":`). OpenAI and Gemini **pretty-print** their REST responses
  (`"content": "ok"`, `"prompt_tokens": 14`), and `atoi` doesn't skip leading
  spaces — so both **text and token extraction returned empty/0** for those
  providers (Anthropic returns compact JSON, which is why it passed earlier).
  Replaced all six extractors with three shared, whitespace-tolerant,
  quote-anchored scanners (`_json_value_pos` / `_json_extract_str` /
  `_json_extract_int`) that skip whitespace around the `:` and require a colon to
  confirm a key — the latter also cleanly skips a string *value* equal to a key
  name (Anthropic's `"type":"text"` vs the real `"text":` field) and subsumes the
  hand-rolled `prompt_eval_count`/`eval_count` disambiguation. Added a
  pretty-printed-JSON regression group (OpenAI/Anthropic/Gemini text + tokens).
  **Live-verified**: `gpt-4o-mini`, `claude-haiku-4-5`, and `gemini-2.5-flash`
  all return through the gateway.
- ✅ **Remote-transport repeated-request SIGSEGV — fixed by switching hoosh to the
  native TLS backend.** Live smoke testing revealed the gateway crashed (SIGSEGV)
  on the *2nd–4th* remote request (stream *and* non-stream, intermittent). Root
  cause: hoosh was building **without** `-D CYRIUS_TLS_NATIVE`, so it ran on the
  deprecated libssl fdlopen bridge; the fault was inside the loaded libssl/glibc
  TLS layer (`cmp …,%fs:…` — the brk-malloc/TLS-arena family of the upstream P1).
  Building native (flag + `sandhi_tls_use_native()`) means **no libssl is ever
  loaded** (verified: 0 libssl maps), and the crash is gone — 10/10 non-stream and
  8/8 streaming requests to Anthropic succeed with the server staying up.
- ✅ **sandhi P1 repeated-HTTPS SIGSEGV resolved upstream** (cyrius 6.1.20 /
  sandhi 1.4.5). cyrius `alloc.cyr`'s `brk` bump heap collided with glibc malloc's
  `brk` arena (pulled in by `fdlopen` loading libssl); fixed by moving the alloc
  heap onto an anonymous-`mmap` chunk-bump allocator and default-switching sandhi
  to the native TLS backend. (hoosh additionally had to *opt into* native — see
  above — to stop loading libssl at all.)

### Verified (live)
- **All three cloud families live end-to-end through the gateway** (`hoosh infer`,
  native TLS backend): **OpenAI** (`gpt-4o-mini`), **Anthropic**
  (`claude-haiku-4-5`), and **Google/Gemini** (`gemini-2.5-flash`,
  `gemini-flash-latest`) each return correct text. OpenAI and Gemini surfaced the
  pretty-printed-JSON extraction bug above; verified fixed.
- **Anthropic** also verified for the system-message hoist (a 3-word system
  instruction is obeyed, which only works if the system turn is hoisted out of
  `messages`), incremental streaming, and repeated requests — no crash.
- **Config**: `$ENV` key expansion verified live — provider blocks resolve
  `api_key = "$GEMINI_KEY"` / `"$ANTHROPIC_AGNOS_KEY"` / `"$OPENAI_KEY"` from the
  environment; secrets stay out of `hoosh.toml`.
- Gemini auth confirmed as the `?key=` query param (Bearer returns 401); the
  gateway degrades gracefully on a `404`/`429` upstream (unknown model / quota)
  without crashing.

### Notes
- **Deferred — blocked on a sandhi P1:** certificate pinning + optional mTLS for
  local providers. sandhi already has live pinning/mTLS
  (`sandhi_tls_policy_new_pinned` → `sandhi_conn_open_with_policy`), but the
  high-level `sandhi_http_post`/`_stream` client never threads a policy
  (`sandhi_http_options` has no policy field), so it's unreachable without
  hand-rolling HTTP+chunked+SSE over `sandhi_conn`. Filed P1 on sandhi to thread a
  policy through `sandhi_http_options`
  (`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`).
- Binary grows (~sandhi/tls/libssl); local-only deployments are unaffected at
  runtime — the TLS path is reached only for `https://` routes.

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
