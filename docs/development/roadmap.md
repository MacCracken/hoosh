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
independently shippable. The one exception is **2.5.11**, the P(-1) hardening sweep —
it audits what the other bands leave behind, so it closes the arc. Items marked ⚠ are
**regressions against rust-old**, not merely unported features.

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

### v2.5.2 — Request-path fidelity — ✅ SHIPPED 2026-07-22
- [x] ⚠ **`temperature` / `top_p` are silently dropped** — both are now parsed,
      range-validated, and forwarded on the blocking *and* streaming paths.
      Carried with `max_tokens` and the 2.5.0 thinking effort in one per-request
      `genopts` record (`types.cyr`) threaded through
      `retry_forward` → `provider_forward` → the body builders, replacing the bare
      effort i64. Fractions are scaled ints (no floats in this codebase) rendered
      back to wire form by `_sb_add_milli`.
- [x] ⚠ **`max_tokens` never reaches the provider** — now forwarded. It also fixes a
      second bug: the old `json_get("max_tokens")` could not see a key placed *after*
      the nested `messages` array (where OpenAI clients put it), so the budget
      reservation silently fell back to 2048. All three params use the raw byte scan
      `_req_num_milli`, the same technique 2.5.0 needed for `reasoning_effort`.
      Anthropic keeps its 4096 / 16384 defaults only when the request omits the field.
- [x] **Request validation** — empty `messages` → 400; `> 256` messages → 400 with the
      count; `temperature` ∈ [0,2] and `top_p` ∈ [0,1] → 400 echoing the offending
      value; `max_tokens` ≤ 0 → 400.
- [x] **Method enforcement → 405** — body-taking routes reject non-POST. Deliberately
      one-sided: the GET routes stay verb-agnostic, since gating them would turn a
      POST that works today into a 405 for no correctness win.
- [x] **`[[providers]] max_tokens_limit`** — per-provider output clamp with a warn log;
      an over-limit request is trimmed, not refused.

> **Anthropic note**: `temperature`/`top_p` are forwarded only with thinking *off*.
> Anthropic pins temperature to 1 under extended thinking and rejects other values, so
> sending them alongside `reasoning_effort` would turn a working request into a 400.
> Effort wins; sampling is dropped for that request.

### v2.5.3 — Provider correctness — ✅ SHIPPED 2026-07-22
- [x] ⚠ **Groq default base URL is wrong** — now `https://api.groq.com/openai`, so the
      default route builds Groq's real endpoint instead of 404ing.
- [x] ⚠ **Retry has no retryability gate** — `provider_forward` now classifies the
      response: a 4xx other than 408/429 returns `FWD_PERMANENT` and `retry_forward`
      stops immediately. Measured: a backend 400 costs 1 attempt / ~7 ms (was 4
      attempts over the full backoff); 500/429/503 still get all 4.
- [x] ⚠ **No socket timeouts at all** — the local provider path had neither a connect
      nor an I/O deadline, so a wedged backend pinned a worker thread forever; with
      7 workers, seven such calls hang the gateway. Now `connect_timeout` 10 s
      (`net_connect_nb`) + 300 s recv/send (`SO_RCVTIMEO`/`SO_SNDTIMEO`), matching
      rust-old. Measured: a blackholed IP now fails at ~10 s, not the ~130 s OS default.
- [x] ⚠ **Ollama never saw any sampling param** — Ollama's native `/api/chat` reads
      `options.{temperature,top_p,num_predict}` and ignores the OpenAI-style top-level
      fields, so 2.5.2's forwarding was inert for the flagship local provider. The body
      builders now emit the native shape for ollama routes.
- [x] **Model catalog completion** — 16 → 34 entries: the Mistral family
      (`mistral-large`, `mistral-small`, `codestral`, `pixtral-large`,
      `open-mistral-nemo`) and the local defaults (`llama3`/`3.1`/`3.2`/`3.3`,
      `codellama`, `mistral`, `mixtral`, `qwen2.5`/`qwen3`, `gemma2`/`gemma3`, `phi3`,
      `deepseek-r1`). Local models previously fell through the unknown-model path,
      which skips context compaction and cost/tier reasoning entirely.
- [x] **`ANTHROPIC_API_VERSION` env override** — was hardcoded `"2023-06-01"`.

> **Deferred to 2.5.7**: Ollama's embeddings response is still passed through rather
> than normalized to the OpenAI `{object:"list", data:[…]}` envelope
> (`ollama.rs:314-340`). `/v1/embeddings` needs its own pass — see
> [the review](rust-old-parity-review.md)'s note that the handler also passes a port
> where a base-url cstr is expected.

### v2.5.4 — Cache expiry — ✅ SHIPPED 2026-07-22
- [x] ⚠ **`[cache] ttl_secs` is inert** — now read, and enforced. The entry grew a
      `created_ms` stamp (the existing one is the LRU recency key, rewritten on every
      read, so nothing could measure age against it). Expiry is checked on read —
      removing the entry and counting it **both** as an eviction and a miss, rust-old's
      accounting — plus a `cache_evict_expired` sweep that runs on capacity pressure
      and before `/v1/cache/stats` reports. `ttl_secs = 0` disables expiry explicitly.
      Verified live: with `ttl_secs = 3`, a repeat query hits, then re-fetches after
      the deadline; the backend saw exactly 2 calls for 4 requests.
- [x] **`/v1/cache/stats` fields** — `max_entries`, `hit_rate`, `enabled`, `ttl_secs`.
- [x] **`semantic_remove(key)`** — an expired or evicted response now takes its
      embedding with it. Without it the index outlives the response it points at, so
      `semantic_find` keeps returning a key that resolves to nothing and the dead
      embedding still competes for `max_search`.

> **Benchmark step change (not a regression).** `cache_get_hit` 108 ns → 1487 ns and
> `cache_insert` 71 ns → 1447 ns in `bench-history.csv`. The *code* did not get
> ~14× slower — the **benchmark got honest**. The bench mirror was a bare map lookup;
> it now models the real entry struct and the `clock_now_ms()` call that `cache_get`
> has always made for the LRU touch. Measured in isolation, `clock_now_ms()` alone is
> **1.351 µs** (a real syscall, not vDSO) and the entry alloc is 19 ns — so the clock
> is essentially the whole number, and it was being paid before 2.5.4 too. A cheaper
> monotonic clock is a genuine win worth taking; filed under [2.5.11](#v2511--security--hardening-audit-sweep).

### v2.5.5 — Health & failover  *(largest band)* — ✅ SHIPPED 2026-07-22
New module [`health.cyr`](../../src/lib/health.cyr): one bankless background thread
probes every enabled route and maintains a per-route health record that
`router_select` consults.
- [x] ⚠ **No health probing or circuit breaker** — now a `UNHEALTHY_THRESHOLD = 3`
      consecutive-failure state machine per route, matching rust-old (`health.rs:27`).
      One blip does not depool a backend; a single success recovers it and resets the
      counter. A `health_changed` event fires on each transition, never on a repeat.
- [x] ⚠ **The router never consults health** — `router_select` now filters unhealthy
      routes out of the candidate set. Verified live: killing the priority-1 backend
      moved traffic to priority-2 after 3 failed probes, and restarting it moved
      traffic back.
- [x] **Per-provider probes** — Ollama `GET /api/tags`, OpenAI-compatible
      `GET /v1/models`.
- [x] **`[server] health_check_interval_secs`** (default 30, 0 = disabled).
- [x] **`ProviderHealth.enabled`** — disabled routes are listed with status
      `"disabled"`; the response also carries `consecutive_failures`, `last_check_ms`
      and the configured `check_interval_secs`.
- [x] **`[[providers]] enabled = false`** — `route_enabled` was already honored by the
      router; nothing ever set it from config.

Three deliberate departures from rust-old, all verified:
- **Probe depth is asymmetric.** Local (`http://`) routes get a real HTTP GET; remote
  (`https://`) routes get a TCP-connect probe. rust-old issued the authenticated GET
  remotely too, but that bills an API call per provider per interval and adds
  rate-limit pressure for a liveness signal a connect already gives — and it would
  need a sigil crypto bank the bankless prober thread does not own.
- **All-candidates-unhealthy falls back to the unfiltered set** rather than returning
  404. A backend can recover between probes, so refusing outright turns a blip into a
  guaranteed outage; trying and failing (502) is strictly more useful. Logged when it
  happens.
- **`/v1/health/providers` no longer probes.** It used to run N blocking connects
  inline on a pool worker for every poll — so a monitoring scrape tied up a worker for
  as long as the slowest backend took — and threw the result away, since nothing but
  that handler read it. It now reports the state the prober maintains.

### v2.5.6 — Cost accounting — ✅ SHIPPED 2026-07-23
- [x] ⚠ **Nothing accumulates cost per provider** — a `CostTracker` now accumulates
      per `(provider, base_url)` on the successful-inference path, and `/v1/costs`
      returns `records` with `total_input_tokens`, `total_output_tokens`,
      `request_count`, `total_cost_micro_usd`/`total_cost_usd` plus a grand total —
      rust-old's `ProviderCostRecord` (`cost/mod.rs:116-127`). It replaces the old
      `providers` array, which listed routes with no cost attached.
- [x] **`admin.costs_reset` audit entry** at `warn` on `POST /v1/costs/reset`, plus a
      warn log. Clearing spend history destroys billing evidence, so it belongs in the
      chain.

> **Bug found while verifying**: `pricing_lookup` checked `provider_is_local` only in
> its *fallback*, after the model-name table. The table is keyed by model NAME and
> hosted-model names are routinely served locally, so `llama-3.3-70b` on Ollama matched
> the Groq row and was billed at 590/790 per million — real money for tokens generated
> on your own GPU. Exactly the realistic cases were mispriced. The local check now runs
> first. (The test file's mirror had it right all along; production had diverged.)

### v2.5.7 — Config-reader closeout — ✅ SHIPPED 2026-07-23
- [x] **`[context]`** — `enabled`, `compaction_threshold` (0.8), `keep_last_messages`
      (10). The compaction budget now derives from the **model's context window** ×
      threshold; it was `est_tokens * 2`, keyed off the requested *output* size, which
      says nothing about how much input a model accepts — a 200k-context model was
      compacted as hard as an 8k one. `keep_last_messages` floors the retained tail so
      compaction cannot strip a conversation to nothing.
- [x] **`[audit]`** — `enabled` (default false), `signing_key` (`$ENV`-expandable,
      random 32-byte fallback), `max_entries`.
- [x] **`[retry] jitter_factor`** (default 0.5) — was hardcoded at 25%.
- [x] ⚠ **Audit chain-link verification** — `audit_verify` now checks that each entry's
      `previous_hash` equals the prior entry's hash, and that the first is GENESIS.
- [x] **`inference.error` audit entry** on the failure path.

> ⚠ **The audit signing key was compiled into the binary** (`"hoosh-audit-key"`), so
> anyone holding the binary could forge or re-sign entries — the signature proved
> nothing. Now sourced from `[audit] signing_key`, with a random per-process key and a
> startup warning when unset.
>
> ⚠ **Two bugs found while verifying.** (1) `audit_verify` checked each entry's own
> hash and HMAC but never the *linkage*, so deleting a record from the middle left a
> chain that verified clean — the surviving entries are individually untouched.
> (2) The compaction system-message detector used the compact literal
> `"role":"system"`, but virtually every JSON encoder emits `"role": "system"` with a
> space (Python's `json.dumps` does by default), so **ordinary client traffic had its
> system prompt dropped during compaction**. Both are fixed and covered.

- [x] **`[hardware]`** — all four keys, each landed **with** the code that acts on it:
      - `disabled_backends` → an ai-hwaccel builder mask; named backends are not
        probed. An unrecognized name warns rather than silently disabling nothing.
      - `vram_reserve_bytes` → `hw_available_vram()` = total accelerator memory −
        in-use − reserve. Placement previously compared against **total**, which
        assumes an idle, exclusively-owned card; on a GPU already running a display
        server or another model that overcommits and the load fails at the backend
        instead of being planned around. `/v1/hardware/placement` now reports
        `available_vram_bytes`, `vram_reserve_bytes` and `fits_available` alongside
        the existing `fits_single_device`.
      - `refresh_interval_secs` → a background re-detection thread (0 = off, the
        previous detect-once-at-startup behavior).
      - `cache_ttl_secs` → the minimum snapshot age before a wake actually re-probes.
        Detection shells out to `nvidia-smi` and friends, so this is what stops a
        short refresh interval from spawning probes constantly.

      Config load also moved **before** `hw_init()` — `disabled_backends` has to be
      known before detection runs, or the backends it names get probed once anyway.

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

<a id="v259--hardware-planning-closeout"></a>

### v2.5.9 — Hardware planning closeout
> `available_vram(reserved)`, `detect_selective` (as a builder mask) and periodic
> re-detection landed early in **2.5.7** with the `[hardware]` config keys that drive
> them. What remains below is the placement/simulation surface.
- [ ] **`POST /v1/hardware/simulate`** — what-if add/remove/replace devices, re-run
      sharding, return `{original, simulated}`; validated (`model_params > 0`, ≤ 64
      devices, non-zero `memory_bytes`). Never ported and never deferred
      (`handlers.rs:1099-1183`, `hardware.rs:389-415`).
- [x] ⚠ **`available_vram(reserved)` / `fits_model`** — landed in **2.5.7**.
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

### v2.5.11 — Security & hardening audit sweep  *(arc closeout — do last)*
A full **[P(-1) scaffold-hardening pass](../../CLAUDE.md)** over the whole gateway once
the parity bands have landed. Deliberately last: 2.5.2–2.5.9 change the request path,
the resilience model, and the config surface, so auditing before they settle would
audit code that is about to move. Run the P(-1) steps in order — baseline benches →
audit (performance, memory, security, edge cases) → cleanliness → tests/benches from
the observations → post-audit benches to prove the wins → repeat if heavy → doc audit.

Carry-forward items observed during the 2.5.1 work, to seed the audit rather than
bound it:
- [ ] **Per-connection allocation is never reclaimed.** `_handle_conn` calls `alloc`
      per request against a bump arena that [never frees](../../CLAUDE.md) — 64 KiB
      minimum, and after 2.5.1's cap up to 1 MiB for a large body. A long-lived
      gateway's RSS therefore grows with total requests served, not concurrency.
      A per-worker reusable buffer (7 workers ⇒ 7 buffers) would bound it. Known and
      deliberately deferred in 2.5.1 — it is a pool refactor, not a hot-fix.
- [ ] **Auth fails open.** `auth.cyr` allows every request when no tokens are
      configured. That is rust-old parity and fine on loopback, but it should be a
      *conscious* posture: at minimum a startup WARNING when hoosh binds a non-loopback
      address with an empty token set. Pairs with 2.5.1's `bind` default.
- [ ] **SSE writers ignore the `sys_write` return.** Flagged as a follow-up in
      [2.4.13](../../CHANGELOG.md): after a client disconnects, hoosh keeps pulling the
      rest of the response from the provider and writing `EPIPE`s until the upstream
      ends. Harmless since the SIGPIPE guard, but it wastes a worker and provider
      tokens on an abandoned request — an unauthenticated client could abandon streams
      in a loop.
- [ ] **Duplicate `ratelimit_new` / `ratelimit_check`** — the build warns
      "last definition wins". Silent shadowing in the rate-limit path is exactly the
      kind of thing an audit exists to catch; resolve or rename.
- [ ] **`clock_now_ms()` costs 1.351 µs — it is a syscall, not vDSO.** Measured during
      2.5.4. It is called on every cache read and twice per insert, and it dominates
      those paths entirely (the surrounding work is ~20 ns). Anywhere hoosh timestamps
      per-request work pays it. A vDSO/coarse-clock path, or reusing one reading per
      request, is a large and cheap win.
- [ ] **Build-surface hygiene** — 13 MB of static data and ~2,700 unreachable functions
      (`CYRIUS_DCE=1`), plus the pinned-vs-installed toolchain drift the compiler warns
      about on every build. Decide the intended posture and make the build quiet, so a
      *new* warning is visible.
- [ ] **Re-run the parity sweep's [Unverified](rust-old-parity-review.md) list** against
      the post-2.5.x source — several leads there are security-adjacent (audit chain
      link verification, `signing_key` sourcing, request validation limits).

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
