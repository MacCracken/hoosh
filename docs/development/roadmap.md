# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work. Shipped releases are
summarized below with per-release detail in [CHANGELOG.md](../../CHANGELOG.md).
Design decisions live in [ADRs](../decisions/).

---

## Shipped

The Cyrius port matched the `rust-old` reference's **surface area** and then extended
well past it. A 2026-07-22 review found the **request path** is not at full parity —
see the [v2.5.x arc](#v25x--rust-old-parity-closeout) and the
[parity review](rust-old-parity-review.md). One line per release; see CHANGELOG for detail.

| Release | Theme |
|---------|-------|
| **2.0.0** | Cyrius port — gateway control plane (routing, budgets, cache, auth, audit, retries, rate-limit, metrics, CLI) |
| **2.1.0 / 2.1.x** | Cyrius 6.0 toolchain modernization; hardware planning endpoints, patra persistence, sakshi logging |
| **2.2.0** | Remote provider transport — sandhi HTTPS, auth headers, Anthropic/Gemini shaping, incremental remote streaming |
| **2.2.1–2.2.4** | Provider correctness (token ratios, lifecycle); DLP; cost optimizer + semantic cache + compression; tool calling (3 families, streaming) |
| **2.3.0** | MCP tool server (`/v1/tools/list` + `/v1/tools/call`, bote) |
| **2.3.1 / 2.3.2 / 2.3.3** | Batch inference — concurrent (`/v1/batch`), then async (job-id/progress/cancel), then concurrent multi-batch + registry eviction ([ADR 009](../decisions/009-concurrent-batch-inference.md)) |
| **2.3.4 / 2.3.5** | Observability — latency histograms, majra event bus + `/v1/events/recent`, traceparent propagation, then OTLP/JSON span export ([ADR 010](../decisions/010-observability.md)) |
| **2.4.0** | Multi-threaded accept loop — unified 7-worker pool serves all traffic concurrently; batch unified onto it ([ADR 011](../decisions/011-multithreaded-accept-loop.md)) |
| **2.4.1** | Hardware planning endpoints — `/v1/hardware/model-format` + `/v1/hardware/requirement-match` |
| **2.4.2** | Threaded hardware detection — parallel CLI probes (`registry_detect_threaded`), unblocked by the 2.4.0 thread-safe foundation |
| **2.4.3** | OTLP remote/`https` export (worker-routed TLS) + scaffolding (state.md, fuzz harnesses, CI security scan) |
| **2.4.4** | New backends — vLLM, TensorRT-LLM, ONNX Runtime (OpenAI-compatible local provider types) |
| **2.4.5** | Hardening review — concurrency sync-pass fixes (cache_stats/tokens_pools), configurable routing strategy + working lowest-latency, dead-code removal ([ADR 011](../decisions/011-multithreaded-accept-loop.md) §2.4.5) |
| **2.4.6** | Toolchain + dependency refresh — Cyrius 6.2.11, ai-hwaccel 2.3.12, bote 2.7.6 (`registry_new` → `tool_registry_new`), majra 2.4.7 |
| **2.4.7** | Toolchain refresh — Cyrius 6.2.37 (single-pass include order) |
| **2.4.8** | Local providers reachable off-localhost + toolchain refresh |
| **2.4.9** | Concrete per-provider model catalog (`/v1/models` lists provider model names) |
| **2.4.10** | Tier-4 consumer step of the coordinated base-security-stack migration |
| **2.4.11** | AGNOS cross-build readiness — networking ported to the portable `net.cyr` socket API |
| **2.4.12** | Tool-continuation fix — OpenAI `tool_calls`/`role:tool` messages now translate to Anthropic `tool_use`/`tool_result` blocks; agentic loops complete (was a misclassified 502) |
| **2.4.13** | SIGPIPE guard — a client aborting mid-SSE-stream no longer kills the gateway |
| **2.5.0** | Extended thinking — `reasoning_effort` → Anthropic adaptive thinking, separate `reasoning_content` stream delta; Cyrius 6.4.62 |

**Toolchain**: Cyrius pin bumped per release; clean `lib/` re-sync each time —
see [the bump note](#toolchain).

---

<a id="v25x--rust-old-parity-closeout"></a>

## v2.5.x — rust-old parity closeout arc

A full behavioral diff of `rust-old/` against the port (2026-07-22, 1,007 behaviors
catalogued) found the surface area is a superset — but a band of **request-path
fidelity, resilience, and config-reader** work did not survive the rewrite. Findings
and evidence: [rust-old-parity-review.md](rust-old-parity-review.md).

Bands below are ordered by severity-over-effort, not by dependency; each is
independently shippable. Items marked ⚠ are **regressions against rust-old**, not
merely unported features.

### v2.5.1 — Security & hard limits  *(arc foundation)* — ✅ SHIPPED 2026-07-22
- [x] ⚠ **`[server] bind` is never read** — `hoosh.cyml` shipped `bind = "127.0.0.1"`
      while `config.cyr` had no reader and `main.cyr` bound `INADDR_ANY()` (0.0.0.0),
      so an auth-optional gateway listened on every interface regardless of config.
      Now parsed via `host_addr` (dotted-quad, hostname, or `0.0.0.0`), and
      **the default is loopback** — matching rust-old's `ServerConfig.bind`
      (`config.rs:496`). The startup banner prints the real address.
      **BREAKING**: a config with no `bind` key now serves loopback only; set
      `bind = "0.0.0.0"` to keep off-host access.
- [x] ⚠ **No request body size limit** — added `MAX_REQUEST_BODY` (1 MiB, matching
      rust-old's `DefaultBodyLimit`) enforced from `Content-Length` **before** any
      large allocation, answering 413. Also fixed the read path itself: `_handle_conn`
      did a single 64 KiB `recv`, silently **truncating** anything larger or split
      across TCP segments — so a legitimate 100 KiB request surfaced as a
      malformed-JSON 400 and could not be served at all. Bodies between 64 KiB and
      the cap now read into a right-sized buffer.
- [x] ⚠ **`[auth] tokens` parses one token, not a list** — `tokens = ["a","b","c"]`
      now yields three tokens (`auth_check` already looped the vec). Extracted a
      shared `_config_str_array` helper, which also **fixes a latent overflow** in the
      `models` pattern parser: it allocated 8 slots and never bounds-checked, so a 9th
      pattern wrote past the allocation. Both arrays are now capped at 32.

### v2.5.2 — Request-path fidelity
- [ ] ⚠ **`temperature` / `top_p` are silently dropped** — neither string exists in
      `src/lib/` or `src/main.cyr`. Rust forwarded and range-validated both
      (`server/handlers.rs:245-267`). A client asking for `temperature: 0` gets
      nondeterministic output with no error — the loudest API-conformance break.
- [ ] ⚠ **`max_tokens` never reaches the provider** — read once
      (`handlers.cyr:1778`) only to size the budget reservation. Anthropic bodies
      hardcode 4096 / 16384 (`provider.cyr:551,558`).
- [ ] **Request validation** — empty `messages` → 400; `messages.len() > 256` → 400;
      temperature ∈ [0,2] and top_p ∈ [0,1] range errors carrying the offending value;
      per-route method enforcement → 405 (`handlers.rs` `validate_chat_request`).
- [ ] **`[[providers]] max_tokens_limit`** — per-provider request clamp with a warn log.

### v2.5.3 — Provider correctness
- [ ] ⚠ **Groq default base URL is wrong** — `types.cyr:150` returns
      `https://api.groq.com`; `_provider_url` concatenates, yielding
      `…/v1/chat/completions`. Groq's OpenAI-compatible endpoint is under `/openai`
      (Rust: `provider/groq.rs:21`). **The default Groq route 404s.**
- [ ] ⚠ **Retry has no retryability gate** — `retry.cyr` carries no status inspection.
      Rust gated on `HooshError::is_retryable()` (`provider/retry.rs:64`), failing fast
      on 400/401/404. The port burns the full backoff schedule on permanent errors.
- [ ] **Connect / total timeouts** — `connect_timeout(10s)`, 300 s provider total
      (`client.rs:127`, `provider/mod.rs:26`).
- [ ] **Ollama passthrough** — `options.temperature`, `options.num_predict` from
      `max_tokens`; embeddings normalized to the OpenAI `{object:"list", data:[…]}`
      envelope (`ollama.rs:125-133,314-340`).
- [ ] **Model catalog completion** — Rust shipped 66 defaults behind a `>= 60` test
      invariant. The Mistral family (`mistral-large`, `mistral-small`, `codestral`,
      `pixtral-large`, `mistral-nemo`) and the local `llama3*` defaults are reported
      absent (`provider/metadata.rs:523,579`).
- [ ] **`ANTHROPIC_API_VERSION` env override** — hardcoded `"2023-06-01"` at
      `provider.cyr:775`.

### v2.5.4 — Cache expiry
- [ ] ⚠ **`[cache] ttl_secs` is inert** — `hoosh.cyml:33` ships `ttl_secs = 300`;
      `ttl` appears zero times in `cache.cyr`/`config.cyr`. The entry layout is
      `{value, last_access_ms}` and that stamp is an LRU recency key, never a deadline,
      so a cached response is served as a hit **indefinitely**. Rust had
      `CacheEntry.ttl`, `is_expired()`, expired-on-get removal bumping *both* evictions
      and misses, and an `evict_expired()` sweep (`cache/mod.rs:40-44,123-129,207-214`).
      Either implement it or drop the key — shipping an inert knob is worse than neither.
- [ ] **`/v1/cache/stats` fields** — `max_entries`, `hit_rate`, `enabled`.
- [ ] **`SemanticCache::remove(cache_key)`**.

### v2.5.5 — Health & failover  *(largest band)*
- [ ] ⚠ **No health probing or circuit breaker** — `health_check`, `consecutive_fail`,
      `unhealthy`, `circuit` match nothing in `src/lib/`. Rust ran a background prober
      with a `UNHEALTHY_THRESHOLD = 3` state machine (`health.rs:27,123-269`).
- [ ] ⚠ **The router never consults health** — `_health_map` is written and read only
      inside `handle_health_providers` (`handlers.cyr:558-567`); it is telemetry for the
      endpoint. Rust filtered unhealthy providers out of the candidate set
      (`router.rs:76-96`). A dead provider keeps taking traffic until an operator polls
      and intervenes.
- [ ] **Per-provider probes** — Ollama `GET /api/tags`; OpenAI-compat authenticated
      `GET /v1/models`; Anthropic `POST /v1/messages` with `content-length: 0`.
- [ ] **`[server] health_check_interval_secs`** (default 30, 0 = disabled).
- [ ] **`ProviderHealth.enabled`** — disabled routes still listed, status `"disabled"`.
- [ ] **`[[providers]] enabled = false`** — disable a route from config (`config.rs:272`).

### v2.5.6 — Cost accounting
- [ ] ⚠ **Nothing accumulates cost per provider** — `pricing.cyr` computes per-token
      cost correctly, but `/v1/costs` (`handlers.cyr:383`) emits global aggregates plus
      a route listing. Rust returned `{records: [ProviderCostRecord], total_cost_usd}`
      keyed `{provider}:{base_url}`, each with `total_input_tokens`,
      `total_output_tokens`, `total_cost_usd`, `request_count` (`cost/mod.rs:116-127`).
      As shipped, `/v1/costs` cannot answer "what has Anthropic cost me today".
- [ ] **`admin.costs_reset` audit entry** at severity `warn` on `POST /v1/costs/reset`.

### v2.5.7 — Config-reader closeout
Keys that exist in the Rust config surface with no Cyrius reader — each is either a
silent no-op today or an unavailable capability:
- [ ] **`[context]`** — `compaction_threshold` (0.8), `keep_last_messages` (10), `enabled`.
- [ ] **`[hardware]`** — `cache_ttl_secs`, `disabled_backends`, `vram_reserve_bytes`,
      `refresh_interval_secs`.
- [ ] **`[audit]`** — `enabled` (default false), `signing_key` (`$ENV`-expandable,
      random 32-byte fallback), `max_entries` (`config.rs:57-72`, `server/mod.rs:154-169`).
- [ ] **`[retry] jitter_factor`** (default 0.5).
- [ ] **Audit chain-link verification** — `entry.previous_hash == prior.hash`
      (`audit.rs:186-195`); the port signs the chain but never validates it end-to-end.
- [ ] **`inference.error` audit entry** on the failure path.

### v2.5.8 — CLI & process lifecycle
- [ ] **`serve --bind <addr>` / `-c|--config <path>`** — `main.cyr:623` takes a bare
      positional port only.
- [ ] **`--server <url>` on `models` / `infer` / `health`**; **`infer --stream`**
      (token-by-token stdout with flush).
- [ ] **Graceful shutdown on SIGINT** (`with_graceful_shutdown`) and **SIGHUP config
      reload** (`server/mod.rs:390-401`).
- [ ] **Log level from the environment** — the `RUST_LOG`/EnvFilter equivalent.
- [ ] **Fatal config-load failure** — Rust exited 1 when a present config failed to parse.
- [ ] **Unknown token pool → 400** `"Token pool 'X' does not exist"` — the port reportedly
      falls through silently (`server/handlers.rs:167-175`); budget must not be skippable
      by naming a nonexistent pool.

### v2.5.9 — Hardware planning closeout
- [ ] **`POST /v1/hardware/simulate`** — what-if add/remove/replace devices, re-run
      sharding, return `{original, simulated}`; validated (`model_params > 0`, ≤ 64
      devices, non-zero `memory_bytes`). Never ported and never deferred
      (`handlers.rs:1099-1183`, `hardware.rs:389-415`).
- [ ] ⚠ **`available_vram(reserved)` / `fits_model`** — the port compares against
      **total** accelerator memory; Rust subtracted in-use memory and a reserve
      (`hardware.rs:276-296`). Overcommits a busy GPU.
- [ ] **`Router::select_with_hardware`** — deprioritize local providers when the model
      won't fit available VRAM (`router.rs:218-295`).
- [ ] **Topology / telemetry accessors** — `system_io()`, `has_fast_interconnect()`,
      `gpu_telemetry()`, `runtime_environment()`, `detect_selective(&disabled)`, and
      periodic re-detection at `refresh_interval_secs`.
- [ ] **Document the `/v1/hardware/format` → `/v1/hardware/model-format` rename** — the
      2.4.1 redesign (path arg → raw bytes) is the safer design, but it is an
      unannounced breaking URL change for rust-era consumers.

### v2.5.10 — Scaffolding parity
- [ ] **`cyrius deny` in CI** — CLAUDE.md lists it in the cleanliness check but
      `.github/workflows/ci.yml` never runs it (Rust: `make deny` + `deny.toml`).
- [ ] **Coverage gate** — no equivalent to rust-old's `codecov.yml` (project 85% /
      patch 80%).
- [ ] **Fuzz targets** — `fuzz_inference_request` and `fuzz_message_content` have no
      `.fcyr` counterpart (current harnesses cover `batch_split` / `trace_extract`).
- [ ] **Bench suites** — `e2e.rs` (client → gateway → Ollama round trip) and
      `live_providers.rs` (live Ollama + hwaccel) unported; plus per-case gaps in
      auth / rate-limit / cost / audit-verify / event-publish / tool-convert.
- [ ] **DLP `custom_patterns`** + `PatternMatch` records (name, level, byte offset,
      length). The port is presence-only, returning just the highest level; all 8
      built-in patterns **are** present.

> **Verification note**: bands 2.5.1–2.5.6 and 2.5.9's simulate item were hand-verified
> against source. The remainder are agent-reported leads from the same sweep — confirm
> before scheduling. See the review doc's split between *Verified findings* and
> *Unverified*.

---

## v2.4.x — Concurrency & completeness arc

Mirroring the 2.2.x parity arc: **v2.4.0 landed the foundation** (the
multi-threaded accept loop, ✅ shipped); the point releases below work the
remaining completeness + hardening items. Order within the arc is a guide, not a
commitment; items tagged *upstream-gated* wait on a sibling repo.

### v2.4.0 — Multi-threaded accept loop  *(arc foundation)* — ✅ SHIPPED 2026-06-10
A unified pool of 7 banked worker threads serves all traffic concurrently; the
accept loop only accepts + enqueues. Batch items share the same pool (the separate
crypto-lane pool is gone); sync batch work-steals the queue. Synchronization pass
landed (batch-registry lock, `_chat_lock` extended to token/health handlers,
lock-free atomic `_router` swap on reload). Live-verified: 14 concurrent chats in
115 ms (≈6× the serialized path), 200 mixed + reload-under-load survived.
See [ADR 011](../decisions/011-multithreaded-accept-loop.md).
**Follow-on:** the concurrency ceiling is 7 (the sigil bank budget) — a
per-thread-arena allocator + more banks would lift it (deferred). Also now
unblocks threaded hardware detection (below).

### v2.4.x candidates (point releases)

**Hardware planning** — remaining ai-hwaccel surface:
- [x] `POST /v1/hardware/model-format` — detect SafeTensors/GGUF/ONNX/PyTorch
      from raw model bytes (**2.4.1**).
- [x] `POST /v1/hardware/requirement-match` — scheduler requirement matching
      against detected hardware (**2.4.1**).
- [x] Threaded detection at startup (`registry_detect_threaded`) — parallel CLI
      probes; was blocked on the pre-thread-safe allocator, wired in **2.4.2**
      (verified byte-identical to serial).

**New backends** — [x] vLLM (PagedAttention), TensorRT-LLM (NVIDIA), ONNX Runtime
(**2.4.4**) — added as OpenAI-compatible local provider types.

**OTLP follow-ups** (extends 2.3.5):
- [x] **Remote / `https://` collector** — DNS + TLS via sandhi; https POSTs
      worker-routed so a banked worker does the TLS (**2.4.3**).
- [ ] **Nested spans** — provider-forward / cache / retry child spans under the
      inference span.
- [ ] **OTLP/protobuf** — the standard wire format; *upstream-gated* on a cyrius
      protobuf lib (proposed: `cyrius/docs/development/proposals/2026-06-10-protobuf-lib.md`).

**Scaffolding modernization** (sibling-repo conventions):
- [x] `docs/doc-health.md` — doc currency tracker (2026-06-10 doc sweep).
- [x] `docs/development/state.md` — volatile state snapshot, per release (**2.4.3**).
- [x] Fuzz harnesses (`fuzz/*.fcyr`) + a CI fuzz step (**2.4.3**).
- [x] Security-pattern scan in CI (`scripts/security-scan.sh`) (**2.4.3**).
- [ ] Split `tests/hoosh.tcyr`/`hoosh.bcyr` into per-topic units — only if the
      suite keeps growing (currently fine as single files).

**MCP tools (szál)** — *upstream-gated*. `/v1/tools/list` + `/v1/tools/call` are
live, but the registry holds only a `bote_echo` smoke tool until **szál** (58
built-in MCP tools) ships as a Cyrius distlib. Register them in `mcp_init`
alongside `bote_echo` — no transport changes. ([ADR 005](../decisions/005-mcp-via-bote.md).)

### Upstream-gated (sandhi)
- **Connection pooling** — high-value case is remote TLS-handshake reuse; gated on
  sandhi keep-alive/pooling. (Local loopback connect ≪ inference latency, so the
  local path has low ROI.)
- **Certificate pinning + mTLS** — pinning/mTLS exist
  (`sandhi_tls_policy_new_pinned`) but the high-level `sandhi_http_post`/`_stream`
  client doesn't thread a TLS policy. Filed upstream
  (`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`).

<a id="toolchain"></a>
> **Toolchain bumps** (process, not arc work): on each pin bump, wipe `lib/` and
> run a clean `cyrius lib sync` + `cyrius deps`, then the full CI step order,
> before trusting a local build — stale `lib/` masks stdlib module renames (e.g.
> 6.1.27 merged `bigint`/`toml`/`json` → `bayan`). Compiler strictness can also
> tighten: 6.2.11 turned duplicate same-scope `var` declarations into a hard
> error (the 2.4.6 bump renamed three such test vars; block-scoped shadowing
> still compiles).

---

## Deferred (external)

### svara — Speech/Audio (migration pending)
STT (Whisper) and TTS (Piper) are migrating from hoosh to **svara**. Hoosh keeps
the provider interface; svara owns the audio pipeline. `/v1/audio/transcriptions`
and `/v1/audio/speech` will not be ported here.

---

## Non-goals

- **Model training** — hoosh is for inference.
- **Model storage** — hoosh doesn't manage model files.
- **Direct GPU compute** — delegated to backends; ai-hwaccel handles detection.
- **Web UI** — hoosh is an API gateway; a dashboard is separate.
- **Audio pipeline** — speech processing belongs to svara.
- **WASM target** — Cyrius doesn't target WASM (was backlog D3).

---

## Backlog

Absorbed from the former root `BACKLOG.md` (deleted 2026-07-22 — every live item
had a roadmap home, and keeping two lists let them drift: it still carried P4 as
open six weeks after 2.4.0 shipped it). This section is the audit trail of the
2026-04-13 Cyrius-port audit; **current** open work lives in the arcs above.

### Open

| ID | Issue | Where it lives now |
|----|-------|--------------------|
| P5 | Connection pooling | [Upstream-gated (sandhi)](#upstream-gated-sandhi) — deferred; local loopback connect is cheap, remote TLS-reuse waits on sandhi keep-alive |

### Cleared during the parity arc (v2.2.x–v2.3.x)

| ID | Issue | Shipped |
|----|-------|---------|
| P1 | Tool calling / MCP bridge | Tool calling **2.2.4**; MCP server (bote) **2.3.0** |
| P2 | DLP content filtering | **2.2.2** — PII scanner + privacy-aware routing |
| P3 | TLS for remote providers | **2.2.0** — sandhi native TLS |
| P4 | Multi-threaded accept loop | **2.4.0** — unified 7-worker pool ([ADR 011](../decisions/011-multithreaded-accept-loop.md)) |
| P6 | OpenTelemetry traces | traceparent **2.3.4**; OTLP/JSON export **2.3.5** |
| P7 | Semantic cache | **2.2.3** — embedding cosine |
| P8 | Batch inference manager | **2.3.1** sync → **2.3.2** async → **2.3.3** concurrent |
| P9 | Cost optimizer | **2.2.3** — cheapest capable model |

### Deferred (external)

| ID | Issue | Reason |
|----|-------|--------|
| D1 | Audio endpoints (STT/TTS) | Migrating to **svara** — see [Deferred (external)](#deferred-external) |
| D3 | WASM target | Cyrius doesn't target WASM — now a [non-goal](#non-goals) |

D2 ("DNS resolution") is **resolved** — the cyrius `sandhi`/`net` stack ships DNS;
remote provider transport landed in 2.2.0.

Rust-era items (M1–M10, L1–L13) were all cleared before the Cyrius port; the
historical list lives in `rust-old/`.
