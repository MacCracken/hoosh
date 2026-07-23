# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [2.5.1] — 2026-07-22

**Security & hard limits — the first band of the rust-old parity closeout arc.** A full behavioral diff of the
archived Rust tree against the port (1,007 behaviors catalogued, see
[the parity review](docs/development/rust-old-parity-review.md)) found the surface area is a superset but the
request path is not at parity. This cut closes the three highest-severity gaps.

### Security
- **The gateway ignored `[server] bind` and listened on `0.0.0.0`.** `hoosh.cyml` has shipped
  `bind = "127.0.0.1"` since the port, but `config.cyr` never parsed the key and `cmd_serve` bound
  `INADDR_ANY()` unconditionally — so hoosh listened on **every interface** while its own config claimed
  loopback. Combined with auth being optional (`auth.cyr`: no tokens configured ⇒ allow all), a stock
  deployment was an unauthenticated LLM gateway reachable from the whole network. `bind` is now parsed
  (via `host_addr`, so a dotted quad, a hostname, or `0.0.0.0` all work) and **defaults to loopback**,
  matching rust-old's `ServerConfig.bind` default. The startup banner prints the address actually bound.
- **`[auth] tokens` accepted only one token.** The parser pushed the raw TOML scalar as a single token, so
  `tokens = ["a", "b"]` matched nothing and multi-key deployments (per-consumer keys, key rotation) were
  inexpressible — rust-old took a list (`config.rs:166`). Now parsed as an array; `auth_check` already
  looped the vec correctly, so this was purely parse-side.

### Added
- **A 1 MiB request body cap, answering `413 Payload Too Large`** — matching rust-old's axum
  `DefaultBodyLimit`. `Content-Length` is checked *before* any large allocation, so an oversized request
  costs one response rather than a megabyte of the never-freeing arena.

### Fixed
- **Requests larger than 64 KiB were silently truncated.** `_handle_conn` did a single `sock_recv` of
  65535 bytes and never drained the rest, so any body bigger than that — or merely split across TCP
  segments — was cut short and surfaced as a malformed-JSON 400. A legitimate 100 KiB chat request could
  not be served at all. The read path now honors `Content-Length` and reads the full body into a
  right-sized buffer (bounded by the new cap). Verified: 200 KB and exactly-1 MiB bodies parse and route;
  1 MiB + 1 is refused with 413.
- **Latent heap overflow in the `models` pattern parser.** `[[providers]] models = [...]` allocated 8
  pointer slots and never bounds-checked the parse loop, so a 9th model pattern wrote past the
  allocation. Both `models` and the new `tokens` array now share a bounded `_config_str_array` helper,
  capped at 32 entries each.

### Changed
- **BREAKING — default listen address is now `127.0.0.1`, was `0.0.0.0`.** A config that omits `bind`
  (or no config at all) now serves loopback only. Set `[server] bind = "0.0.0.0"` to restore off-host
  access. Deployments using the shipped `hoosh.cyml` are unaffected in effect — it already said
  `127.0.0.1`; only the actual behavior changes to match it.
- **`BACKLOG.md` removed**; its contents are now a [Backlog section](docs/development/roadmap.md) in the
  roadmap. Two lists had drifted — it still carried P4 (multi-threaded accept loop) as open six weeks
  after 2.4.0 shipped it.

### Docs
- **New: `docs/development/rust-old-parity-review.md`** — the parity diff's evidence record, splitting
  hand-verified findings from agent-reported leads.
- **Roadmap: the v2.5.x parity-closeout arc** (10 bands), plus the missing 2.4.13 / 2.5.0 shipped rows.
  The *Shipped* preamble's "reached parity" claim is qualified — surface area matched, request path did not.

Gates: 482 tests (was 464), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.0] — 2026-07-14

**Extended thinking / reasoning: `reasoning_effort` control + a `reasoning_content` stream. Toolchain + deps refresh.**

### Added
- **`reasoning_effort` (`low`/`medium`/`high`) now controls Anthropic extended thinking.** hoosh maps it to the
  model's **adaptive** thinking API — `"thinking":{"type":"adaptive"},"output_config":{"effort":"..."}` — and raises
  `max_tokens` to leave room for a reasoning-influenced answer. This is the format current models (Opus 4.8, Sonnet
  4.x) require; the legacy `"thinking":{"type":"enabled"}`+`budget_tokens` is **rejected** by them ("not supported
  for this model"). The effort is read with a **raw byte scan** (`_req_reasoning_effort`, like `_req_is_stream`) —
  NOT the flat `json_parse`, which can't reach a top-level key that follows the nested `messages` array (where
  OpenAI clients put it). Threaded per-request on the prep struct (hoosh is multi-threaded — a global would race).
  Without `reasoning_effort` the request is byte-identical to before.
- **A `reasoning_content` streaming delta** — hoosh now translates a provider's reasoning stream into the de-facto
  OpenAI-compat `delta.reasoning_content` field, kept SEPARATE from `delta.content` so a client can fold the
  reasoning apart from the answer. For Anthropic, an SSE `thinking_delta` → `reasoning_content` (new
  `anthropic_extract_thinking` + `_sse_reasoning_chunk`). Inert on a normal turn, and — usefully — inert on **Opus
  4.8**, which keeps its reasoning INTERNAL (it streams only `text_delta`, never `thinking_delta`, even at high
  effort); the translation is there for models that DO expose thinking. Verified live: `reasoning_effort` requests
  complete (were rejected/empty before this cut); unit-tested (`_req_reasoning_effort` incl. the after-`messages`
  case).

### Changed
- **Toolchain + dependency refresh**: Cyrius pin **6.3.15 → 6.4.62** (`cyrius.cyml`, vendored stdlib re-synced),
  and the `ai-hwaccel` dep **2.3.12 → 2.3.14**. Build + the 459-test suite green on the new toolchain (464 with the
  new `reasoning_effort` tests).

## [2.4.13] — 2026-07-13

**Fix: a client disconnecting mid-stream (SIGPIPE) crashed the gateway.**

### Fixed
- **The gateway process died when an SSE-streaming client disconnected mid-response** — e.g. thoth aborting a
  streaming turn with its Esc-interrupt. `cmd_serve` runs hoosh's own accept loop and writes SSE chunks with bare
  `sys_write` (no `MSG_NOSIGNAL`), so the *next* chunk write after the client went away raised **SIGPIPE**, whose
  default disposition **terminates the process**. hoosh crashed a moment after any interrupted stream, so every
  following request "failed to connect." `cmd_serve` now installs the SIGPIPE→`SIG_IGN` guard at startup (sandhi's
  proven `rt_sigaction` helper — the same one `sandhi_server_run` installs, which our hand-rolled loop bypassed), so
  a write to a dead peer yields an `EPIPE` error instead of killing the gateway. Verified: repeated mid-stream client
  aborts leave hoosh alive and serving subsequent requests.
- **Follow-up (not in this patch):** the SSE writers don't yet check the `sys_write` return, so after a client
  disconnects hoosh keeps pulling the rest of the response from the provider and writing (ignored) `EPIPE`s until the
  upstream stream ends — harmless now (no crash) but wasteful. Detecting the write error to abort the upstream
  early is a follow-up.

## [2.4.12] — 2026-07-03

**Tool-continuation fix — agentic loops now complete on Anthropic backends.**
The Anthropic request-builder copied OpenAI messages verbatim, so a follow-up
turn carrying an assistant `tool_calls` message plus a `role:"tool"` result was
sent to Anthropic unchanged — shapes the Messages API rejects — and the failure
surfaced (misclassified) as `502 provider backend unreachable`. This broke every
multi-step agentic tool loop: the model could call a tool once, but feeding the
result back to continue failed. Single-turn and repeated-single-turn calls were
unaffected, which is why the break was specific to the tool-continuation shape.

### Fixed
- **`_build_anthropic_body_x` (`src/lib/provider.cyr`) now translates OpenAI tool
  messages into Anthropic content blocks** as it re-emits the `messages` array:
  - an assistant message with `tool_calls` → `{"role":"assistant","content":[…]}`
    with a `text` block (only when `content` is a non-empty string; `content:null`
    is dropped) followed by one `tool_use` block per call — the call's `arguments`
    **string** is unescaped into the `input` **object** (empty/absent → `{}`), and
    `tool_use.id` carries the OpenAI `id` so it matches the tool result;
  - a `role:"tool"` message → `{"role":"user","content":[{"type":"tool_result",
    "tool_use_id":<tool_call_id>,"content":<result string>}]}`;
  - every other message (plain `{role, content-string}`) still passes through
    verbatim — already Anthropic-shaped.
  New helper `_json_unesc_span` reverses JSON string-escaping so an `arguments`
  string becomes a literal JSON object in `input`.

### Verified
- The exact filed repro (assistant `tool_calls` + `role:"tool"` continuation) now
  returns **200** with a correct completion; streaming and a two-`tool_call` /
  two-result continuation both return 200 (streaming also restores the closing
  summary that previously came back empty).
- Full vertical re-proven: thoth → hoosh (`claude-opus-4-8`) → tool call →
  t-ron allow → daimon → bote 3.0.0 `fs_write` (file written) → **continuation
  turn returns the final summary**; the one-shot agentic loop exits 0.
- Suite green (457 assertions).

## [2.4.11] — 2026-07-01

**AGNOS cross-build readiness.** hoosh now compiles cleanly under
`cyrius build --agnos`. Host build byte-identical; 457 assertions still pass.

### Changed
- **Networking ported to the portable `net.cyr` socket API.** The six
  outbound-TCP sites (`main.cyr`, `lib/http_client.cyr` ×2, `lib/otlp.cyr`,
  `lib/handlers.cyr` ×2) used raw Linux BSD sockets (`sys_socket` / `sys_connect`)
  — undefined on agnos's frozen syscall surface. They now use `tcp_socket()` +
  `sock_connect(fd, addr, port)`, which dispatch per target (Linux
  `socket`/`connect`; agnos `sock_connect`#47 yielding a tagged fd). The existing
  `sys_write` / `sys_read` / `sys_close` on the fd are unchanged — the agnos
  syscall peer routes tagged fds to `sock_send`/`recv`/`close` (#48/#49/#50), so
  hoosh's gateway networking **runs** on agnos, not merely compiles.

### Fixed
- **`--agnos` build**: guarded the Linux-only `THREAD_STACK_SIZE` knob in
  `lib/hardware.cyr` (HW-probe threads) and `lib/pool.cyr` (worker pool) behind
  `#ifndef CYRIUS_TARGET_AGNOS` — the agnos threading lib has no such global, so
  detection and the pool fall back to their existing paths.

## [2.4.10] — 2026-06-30

Tier-4 (consumer) step of the coordinated base-security-stack migration
to cyrius **6.3.15**. Toolchain pin + re-sync of the vendored bundles to
the migrated versions, plus the stdlib boundary reconciliation the 6.3.x
line requires. All 457 assertions pass on the new stack.

### Changed

- **Cyrius toolchain pin: 6.2.39 → 6.3.15.**
- **Vendored bundles re-synced**: `src/vendor/majra.cyr` 2.4.7 → **2.5.0**
  and `src/vendor/bote-core.cyr` 2.7.6 → **2.7.7** (the migrated tiers).
- **`[deps] stdlib`**: added `atomic` + `sync` (patra's transitive
  `lib/sync.cyr` requirement on 6.3.x), `random` (sigil's `random_bytes`
  pull-through, before `sigil`), and `async` (tls/sandhi inter-dep on
  6.3.x, before `tls`).

### Fixed

- **Vendored majra var-bomb** (`src/vendor/majra.cyr`). The stale 2.4.7
  bundle carried the undersized `var ts[2]` / `var buf[2]` / `var hdr[1]`
  locals that cyrius 6.3.13's move of function-local `var X[N]` arrays to
  the guard-paged thread stack turns from a benign adjacent-scribble into
  a hard SIGSEGV. The 2.5.0 re-sync brings the correctly-sized buffers.

## [2.4.9] — 2026-06-24

**Concrete per-provider model catalog.** `/v1/models` lists provider *names* (one
per enabled route); it does not enumerate the specific models a client can switch
to. New endpoint surfaces them.

### Added
- **`GET /v1/models/catalog`** — concrete, switchable model ids from the pricing
  catalog (metadata.cyr), each tagged with the provider that would route it as
  `owned_by`. Only models an **enabled route actually matches** are listed, so the
  catalog reflects what this gateway can serve (e.g. with no DeepSeek/Groq route
  configured, those models are omitted; `o1` is omitted when only `o1-*` is
  configured). Shape mirrors `/v1/models` so the same client parser works:
  `{"object":"list","data":[{"id","object":"model","owned_by"}]}`. Clients drill
  down per provider by filtering on `owned_by`. (`handle_models_catalog`,
  `_catalog_provider_for`; new `metadata_count` / `metadata_key` accessors.)
  `/v1/models` (provider list) is unchanged.

## [2.4.8] — 2026-06-24

**Local providers reachable off-localhost + toolchain refresh.** Fixes a bug
that made any local (plaintext `http://`) provider — Ollama, LlamaCpp —
unreachable unless it ran on the *same host* as hoosh, and bumps Cyrius to
**6.2.39**.

### Fixed
- **Local providers on a non-localhost host returned `502 provider backend
  unreachable`.** The raw-socket forwarder for `http://` backends
  (`http_client.cyr`) hardcoded the connect target to `127.0.0.1` and parsed
  only the **port** from `base_url` — the host was discarded. So an Ollama box
  at e.g. `http://192.168.1.186:11434` was contacted at `127.0.0.1:11434`,
  where nothing listened. Added `url_host` (parse the host from the URL) and
  `host_addr` (resolve it: `localhost` → loopback, dotted-quad literal parsed
  directly, otherwise sandhi DNS; loopback fallback on miss), and routed every
  local-path connect through them: blocking + streaming chat forward
  (`provider.cyr`), the HTTP `/health` + provider-status probes and the CLI
  `health` command, the Ollama/Synapse lifecycle admin (pull/delete/training/
  catalog), embeddings, and OTLP export. The `Host:` header now carries the
  real host. Remote `https://` routing (via sandhi) was always host-correct and
  is unchanged.

### Changed
- **Toolchain: Cyrius 6.2.39** (pin, was 6.2.37). Clean `lib/` re-sync; no
  stdlib module migration. All 457 tests pass.

## [2.4.7] — 2026-06-23

**Toolchain refresh.** Bumps Cyrius to **6.2.37** with the one single-pass
include-order fix the new stdlib requires.

### Changed
- **Toolchain: Cyrius 6.2.37** (pin, was 6.2.11). Clean `lib/` re-sync; no
  stdlib module migration.

### Fixed
- **Test/bench harness `undefined variable 'sys_getrandom'` under 6.2.37.**
  Since stdlib 6.2.28, `tls_native_conn` takes `&sys_getrandom`. `cyrius build`
  prepends the auto-injected `[deps]` stdlib set (incl. `tls`) ahead of a file's
  explicit includes, in array order — so an explicit `include "lib/syscalls.cyr"`
  in `tests/hoosh.tcyr` / `tests/hoosh.bcyr` landed *after* the prepended `tls`,
  leaving `sys_getrandom` undefined at that point. Dropped the explicit syscalls
  include from both harnesses (it auto-injects at its array position, before
  `tls`); `sys_*` wrappers stay in scope. All 457 tests pass, benchmarks intact.

## [Unreleased]

## [2.4.6] — 2026-06-15

**Toolchain + dependency refresh.** Bumps Cyrius to **6.2.11** and all
dependencies to their latest tags, with the one breaking-API fix the bote
update requires.

### Changed
- **Toolchain: Cyrius 6.2.11** (pin, was 6.1.31). Clean `lib/` re-sync; no
  stdlib module migration (the 6.2.x snapshot only *adds* modules —
  `tls_native_*` split, `*_agnos` variants — and removes none).
- **ai-hwaccel 2.3.9 → 2.3.12** (`cyrius.cyml` tag). `dist/ai-hwaccel.cyr`
  re-vendored; the vendored `data/cloud_pricing.json` + `data/models.json` are
  byte-/content-identical at the new tag, so they are unchanged.
- **bote 2.7.3 → 2.7.6** (`src/vendor/bote-core.cyr`). Code is identical apart
  from the rename below and whitespace.
- **majra 2.4.5 → 2.4.7** (`src/vendor/majra.cyr`). Version header only — the
  bundle body is unchanged.

### Fixed
- **bote renamed its registry constructor `registry_new` → `tool_registry_new`.**
  This matters because ai-hwaccel *also* defines `registry_new`, so leaving the
  call unchanged would have silently bound hoosh's MCP registry to the hardware
  registry ("last definition wins"). Updated the call sites in `src/lib/mcp.cyr`
  and the test/bench harnesses to call `tool_registry_new` explicitly. The other
  `registry_*` ops keep their names. All 457 tests pass, MCP benchmarks intact.
- **Cyrius 6.2.11 now hard-errors on duplicate same-scope variables.** The test
  harness redeclared three top-level `var`s in `main()` (`tp`, `ll1`, `ll2`);
  renamed the second occurrences (`tval`, `llv1`, `llv2`). Block-scoped
  shadowing (e.g. `rec`) is still allowed and was left as-is.

## [2.4.5] — 2026-06-10

**Hardening review** — closes the v2.4.x arc. A concurrency/correctness/security
audit of the 2.4.x code with concrete fixes, plus Cyrius **6.1.31**.

### Fixed (concurrency — v2.4.0 sync-pass misses)
- **`GET /v1/cache/stats`** read `map_count` over the cache entries map without a
  lock — racing the chat path's `cache_insert`/evict (`map_set`/`map_delete` →
  rehash) on the pool workers, a latent crash. Now snapshots the four values
  under `_chat_lock`.
- **`GET /v1/tokens/pools`** iterated `map_keys(_budget)` + read pool fields
  unlocked — racing reload's `budget_add_pool` and reserve/commit writes. Now
  under `_chat_lock`. Both verified with a 240-way concurrent stress (chat +
  stats + pools), no crash.

### Fixed (correctness)
- **Routing strategy is now configurable** — `[server] strategy =
  "priority"|"round-robin"|"lowest-latency"|"direct"`. It was hardcoded to
  `priority`, so three implemented strategies were unreachable.
- **Lowest-latency routing now works** — its EMA writer (`router_report_latency`)
  was never called (dead), and `router_select` treated untried backends as
  *max* latency, so they were never explored. Now the forward feeds the EMA
  (thread-safe via a dedicated `_lat_lock`, gated on the active strategy), and
  untried backends are explored before exploiting the fastest. Live-verified
  24:1 toward the fast backend across two mocks.

### Changed
- **Toolchain: Cyrius 6.1.31** (pin). Clean `lib/` re-sync; no stdlib migration.
- Removed dead `_cost_map` (declared + `map_new`'d, never used).

## [2.4.4] — 2026-06-10

**New backends** — vLLM, TensorRT-LLM, and ONNX Runtime, completing the v2.4.x
arc's new-backend item. **17 providers** now (was 14).

### Added
- **vLLM** (`type = "vLLM"`), **TensorRT-LLM** (`"TensorRT-LLM"` / `"TensorRT"` /
  `"trtllm"`), **ONNX Runtime** (`"ONNX"` / `"ONNXRuntime"`) provider types. All
  three serve an OpenAI-compatible API, so they route through the existing local
  OpenAI-compatible forward (`/v1/chat/completions`) — `types.cyr` gains the enum
  entries + name/is-local/parse/default-url; no forward-path change. Default base
  URLs `http://localhost:8000`; marked **local** (free in the cost optimizer,
  permitted for Confidential-classified traffic under DLP). Live-verified: all
  three parse + register + route + forward to a mock (`POST /v1/chat/completions`
  confirmed); vLLM end-to-end chat round-trip.

## [2.4.3] — 2026-06-10

**OTLP remote/https export + scaffolding** — an observability follow-up plus
sibling-repo scaffolding conventions.

### Added
- **OTLP remote / `https://` collector** — the exporter now posts to remote
  collectors over TLS via sandhi (DNS + TLS), not just a localhost `http://`
  sidecar. Because TLS needs a sigil crypto bank and the exporter thread is
  bankless (banks 0..7 = main + 7 workers), an `https` POST is enqueued as a
  `JOB_OTLP_EXPORT` so a **banked pool worker** does the TLS; `http` stays direct
  on the exporter. Live-verified: http path receipt-confirmed at a local
  collector; https path does real TLS POSTs to a public endpoint on a worker with
  the server stable under concurrent load.
- **`docs/development/state.md`** — volatile-state snapshot (version, sizes,
  test/bench counts), refreshed per release (patra/cyrius pattern); linked from
  the docs index.
- **Fuzz harnesses** (`fuzz/*.fcyr`) — adversarial-input harnesses for the
  hand-rolled parsers (`_batch_split` array splitter, `trace_extract` header
  scanner): unbalanced brackets, trailing escapes, unterminated strings,
  truncation, generated byte stress. No crash across ~8k inputs.
- **CI: security-pattern scan + fuzz step** — `scripts/security-scan.sh` (flags
  subprocess exec, hardcoded `/etc` paths, hardcoded provider keys in `src/`) and
  a CI step building + running every `fuzz/*.fcyr` under a timeout.

### Notes
- OTLP/protobuf stays upstream-gated (cyrius protobuf lib); nested spans deferred.

## [2.4.2] — 2026-06-10

**Threaded hardware detection** — the last hardware item of the v2.4.x arc, now
unblocked by the v2.4.0 thread-safe foundation.

### Changed
- **`hw_init` uses ai-hwaccel's threaded detector** (`registry_detect_threaded`)
  — the CLI-tool probes (nvidia-smi, vulkaninfo, hl-smi, neuron-ls, …) run in
  parallel threads while sysfs backends run inline; per-thread result vecs are
  merged after join (race-free). Previously serial: the threaded detector
  segfaulted under the pre-thread-safe allocator. It's safe since the v6.0.64 CAS
  spinlock — the only shared mutation is `alloc` (atomic), and the fork+exec child
  does only raw syscalls (no child-side alloc → fork-safe from a thread). Probe
  threads get 1 MB stacks (set before the first `thread_create`); internal serial
  fallback if `thread_create` fails. **Verified**: detection output is byte-identical
  to serial (3 profiles, ROCm GPU), no segfault across repeated runs, server
  startup clean.

## [2.4.1] — 2026-06-10

**Hardware planning endpoints** — the remaining ai-hwaccel surface from the
v2.4.x arc. Also bumps the toolchain to Cyrius **6.1.30**.

### Added
- **`POST /v1/hardware/model-format`** — detect a model's container format from
  the **raw model bytes** in the request body (send the file or its header; no
  filesystem path, so no arbitrary-read surface). Returns ai-hwaccel's
  `ModelMetadata` JSON (`format` + `param_count`/`dtype`/`tensor_count`/
  `format_version` when present). Detects SafeTensors / GGUF / ONNX / PyTorch;
  unrecognized → 400. Live-verified against crafted GGUF + SafeTensors headers.
- **`POST /v1/hardware/requirement-match`** — does the detected hardware satisfy a
  scheduler requirement? Body `{requirement: "gpu"|"tpu"|"gaudi"|"aws-neuron"|
  "gpu-or-tpu"|"any-accelerator"|"none", min_chips?: N}` → `{requirement,
  min_chips, satisfied, device?}`. Reads the immutable hw registry (detected at
  startup) — lock-free under the v2.4.0 worker pool. Live-verified (ROCm GPU
  detected on the dev machine; TPU/min_chips → false; unknown req → 400).

### Changed
- **Toolchain: Cyrius 6.1.30** (pin). Clean `lib/` re-sync; no stdlib migration;
  442 tests green.

## [2.4.0] — 2026-06-10

**Multi-threaded accept loop** — interactive traffic now runs concurrently, not
just batch. A unified pool of 7 banked worker threads serves all requests; the
accept loop only accepts + enqueues. See [ADR-011](docs/decisions/011-multithreaded-accept-loop.md).

### Added
- **Unified worker pool** (`src/lib/pool.cyr`) — `WORKER_COUNT = 7` threads, each
  permanently owning a sigil crypto bank (1..7), draining a bounded work-queue
  ring. The accept loop hands each connection to the pool as a job. Bounded by the
  8-crypto-bank limit (bank 0 = main); no thread-per-connection, no DoS surface.
- **Batch unified onto the pool** — `/v1/batch` items are now queue jobs run by
  the same workers (the separate crypto-lane pool is gone). Sync batch
  **work-steals** the queue while waiting (can't stall the pool); async batch uses
  a bankless coordinator thread. The full 7-bank budget is shared dynamically
  between interactive + batch, with no static split.

### Changed
- **Synchronization pass** — shared state guarded for concurrent handlers: a
  `_batch_reg_lock` for the batch registry (map/id-list), `_chat_lock` extended to
  the token handlers (budget) and the health map. Config hot-reload rebuilds
  `_router` as an atomic pointer swap (lock-free readers, safe via the never-frees
  allocator) under `_chat_lock`. Worker stacks bumped to 1 MB.

### Performance
- Live-verified: 14 concurrent 50 ms chats in **115 ms** (≈6× faster than the
  ~700 ms serialized path), 50 in 417 ms; 200 mixed requests + reload-under-load
  survived; sync/async/concurrent batches all correct. `work_queue_push_pop`
  dispatch overhead **8 ns**.

## [2.3.5] — 2026-06-10

**OpenTelemetry OTLP span export** — the deferred half of the 2.3.4 observability
work. One span per inference request is exported to a collector over OTLP/HTTP
with **JSON** encoding (no protobuf — a Cyrius protobuf lib is proposed upstream
for a future OTLP/protobuf path). Also bumps the toolchain to Cyrius **6.1.29**.

### Added
- **`src/lib/otlp.cyr`** — opt-in OTLP exporter, off unless `[[telemetry]]
  otlp_endpoint` is set. Each inference enqueues an OTLP span (traceId/spanId
  from the 2.3.4 traceparent → joins the caller's trace; epoch-ns start/end;
  attributes provider/model/latency_ms/tokens; status OK/ERROR) into a bounded
  ring; a **background thread** batches the ring into an OTLP `resourceSpans`
  document and POSTs it to the collector every second (non-blocking). Localhost
  http collector (the common sidecar); remote/https is a follow-up.
  **Live-verified** against a mock collector: success (status 1) + failure
  (status 2) spans, correct trace-id correlation, epoch timestamps, batching.
- **`[[telemetry]]` config** — `otlp_endpoint` (e.g. `http://localhost:4318/v1/traces`)
  + `service_name`. Absent → no exporter, no overhead.
- **CLOCK_REALTIME epoch-ns clock** for OTLP timestamps (`clock_now_ns` is
  monotonic).

### Changed
- **Toolchain: Cyrius 6.1.29** (pin). Clean `lib/` re-sync; no stdlib migration;
  427 tests green.

## [2.3.4] — 2026-06-10

**Observability** — per-provider latency histograms, a provider event bus
(majra pub/sub) with a recent-events endpoint, and W3C `traceparent` propagation.
Full OpenTelemetry OTLP export is deferred to 2.3.5. See ADR-010.

### Added
- **Per-provider latency histograms** (`/metrics`) — `hoosh_provider_latency_ms`
  Prometheus histogram (11 le-buckets + sum + count, per provider), recorded
  around each provider forward (atomic — batch workers record concurrently).
- **Provider event bus** (`src/lib/events.cyr`, **majra 2.4.5** pubsub) — the
  four `ProviderEvent` kinds from `events.rs` are published as JSON: HealthChanged
  (on a health-poll flip), InferenceCompleted (provider/model/latency/tokens),
  InferenceFailed (provider/model/error), RateLimited (provider). majra is the
  pub/sub substrate (`hoosh_events_published_total` in `/metrics`); since hoosh's
  loop is synchronous, observability is a bounded **recent-events ring** at
  **`GET /v1/events/recent`** (a never-drained subscriber channel would fill and
  block, so events append to the ring as they publish). Live-verified all four.
- **W3C `traceparent` propagation** (`src/lib/trace.cyr`) — the incoming
  `traceparent` is forwarded verbatim to backend requests (local + remote), or a
  fresh one is generated (`clock_now_ns` + atomic counter) when absent, so the
  gateway→backend hop joins the trace. Carried through the deep, threaded forward
  path via a **thread-local** slot (slot 1; slot 0 is sigil's crypto bank) rather
  than threading the header through every signature. Live-verified (incoming
  forwarded; absent → generated).

### Notes
- **majra is vendored** at `src/vendor/majra.cyr` (committed, not `[deps]` — same
  rationale as bote-core). Its `ratelimit_new`/`ratelimit_check` collide by name
  with hoosh's (different signatures); harmless — they're dead code in majra's
  core pubsub, so hoosh's win and work. One benign `last-definition-wins` build
  note, like the existing bayan ones.
- **Gotcha fixed during bring-up:** `thread_local_init()` is NOT idempotent — it
  installs a fresh zeroed TLS block. Calling it per-access wiped the slot (and
  would clobber sigil's crypto bank). It is now called exactly once per thread
  (main via `crypto_tls_main_init` at startup; workers via CLONE_SETTLS).

## [2.3.3] — 2026-06-10

**Concurrent async batches + registry eviction** — the deferred follow-ups to
2.3.2. Multiple async batches now execute *concurrently* (sharing a global
crypto-lane pool), and the batch registry is bounded. Also bumps the toolchain to
Cyrius **6.1.28**.

### Added
- **Global crypto-lane pool** — replaces the one-batch-at-a-time `_batch_exec_lock`.
  sigil's 7 worker crypto lanes (banks 1..7; bank 0 = accept thread) are handed
  out across **all** batches: a worker acquires a free lane → uses it as its
  crypto bank → releases it. Total live workers stay ≤ 7 globally regardless of
  how many batches run, with no cross-batch lane collision. Both sync and async
  batches draw from the pool. Live-verified: 4 async batches of 12 items
  progressed simultaneously (e.g. 8/2/1/3 → 11/9/6/9), all completed 12/12, with
  GET polls returning live progress (HTTP 200) and 15/15 concurrent chats served.
- **Registry eviction** — `BATCH_MAX_TRACKED` (64); on submit over the cap, the
  oldest *terminal* (completed/cancelled) batch is dropped from the registry +
  id list (evicted ids 404). Bounds the tracking map (not heap — hoosh's bump
  allocator never frees, same as every allocation). Live-verified (after 70
  submits, the first id → 404).

### Changed
- **Runners/barriers sleep instead of busy-spin** — `_batch_lane_acquire` and
  the completion barriers now `sleep_ms(1)` when waiting, so concurrent runners
  don't burn cores or starve the accept loop serving `GET /v1/batch/{id}`.
- **Toolchain: Cyrius 6.1.28** (pin). Clean `lib/` re-sync; no stdlib migration
  this bump; 414 tests green.

### Notes
- Concurrency is bounded by the 7-lane pool; submitting many batches at once is
  fine (they share lanes fairly), but per-batch throughput drops as lanes are
  split across active batches. See ADR-009.

## [2.3.2] — 2026-06-10

**Async batch inference** — `POST /v1/batch` with `{"async":true}` returns a
batch id immediately and runs the batch on a background thread; clients poll
`GET /v1/batch/{id}` for progress and `POST /v1/batch/{id}/cancel` to cancel.
Completes the `inference/batch.rs` port (submit → progress → cancel → registry).
The synchronous `POST /v1/batch` (2.3.1) is unchanged and remains the default.

### Added
- **`POST /v1/batch` async mode** (`{"async":true}`) → `{"id","status","total"}`.
  A background runner thread executes the items (waves of 7 crypto-bank workers,
  same engine as the sync path), updating a `BatchState`.
- **`GET /v1/batch/{id}`** → progress snapshot: `{id,status,total,completed,
  failed,results[]}`. `status` ∈ queued/running/completed/cancelled; a still-
  pending item shows `status:null,body:null`. Reads are lock-free (atomics +
  write-once result slots).
- **`POST /v1/batch/{id}/cancel`** → sets the cancel flag; in-flight items finish
  but no further waves launch (the runner checks the flag before each wave).
- **Batch registry** (`src/lib/batch.cyr`): `id → BatchState` map + atomic id
  counter, touched only by the single-threaded accept loop (no lock needed).
  Live-verified end-to-end: submit→poll→complete (6/6), cancel mid-flight (28-item
  batch → cancelled at 14/28), and **20/20 concurrent `/v1/chat/completions`
  served while a 28-item batch ran in the background** (the sync-pass guarantee).

### Changed
- **Sync pass: the accept-thread chat path now holds `_chat_lock`** around its
  shared-state phases (prep, assemble/commit; streaming audit appends too). An
  async batch runs workers concurrently with the accept loop, so a
  `/v1/chat/completions` served mid-batch would otherwise race batch workers on
  `_cache`/`_budget`/`_audit`. Uncontended (cheap) when no batch is in flight.
- **One batch executes at a time** via `_batch_exec_lock` (sync and async share
  it), so crypto-bank workers (lanes 1..7) never overlap across batches;
  additional submitted batches queue (`status:"queued"`).

### Notes
- Async batches run sequentially (one executes at a time); concurrent execution
  of multiple async batches, and completed-batch eviction from the registry, are
  future work. See ADR-009.

## [2.3.1] — 2026-06-10

**Concurrent batch inference** — `POST /v1/batch` runs an array of chat requests
on worker threads and returns all results in one response. Unblocked by
re-verifying (not assuming) that the cyrius 6.1.27 allocator is thread-safe: a
4-thread × 200k-alloc stress showed zero corruption (the v6.0.64 CAS spinlock
fixed the race the older roadmap audit flagged as a hard blocker). Async
job-id/progress-poll/cancel and a fully multi-threaded accept loop are deferred
(see roadmap); this ships the synchronous concurrent executor.

### Added
- **`POST /v1/batch`** — body `{"requests":[ <chat req>, … ]}`; runs the items
  concurrently (bounded to `BATCH_MAX_PARALLEL`=7 in flight, processed in waves;
  `BATCH_MAX_ITEMS`=64 cap) and returns `{"results":[{"index","status","body"},
  …]}`. Items are treated as non-streaming. Ports the concurrency core of
  `rust-old/src/inference/batch.rs`. Live-verified against a mock backend: 8
  forwards per wave overlap (confirmed by per-connection arrival timestamps),
  waves serialize, results are correct, and metrics stay exact under load.
- **`src/lib/batch.cyr`** — executor + string/nesting-aware request-array
  splitter (`_batch_split`, `_batch_extract_requests`). Worker completion is
  signalled through an **atomic counter barrier** rather than `thread_join`
  timing (empirically `thread_join` could return before a worker stored its
  result, which would let the next wave overlap and risk reading an unfilled
  slot). Unit-tested (split: nested arrays + bracket-bearing strings) + benched
  (`batch_split_4`).

### Changed
- **`handle_chat` refactored into a returns-body core** — `_chat_prep`
  (validation/routing/DLP/cache/rate/budget/messages → terminal descriptor or
  ready state), `_chat_forward` (the unlocked network call), and `_chat_assemble`
  (response build + cache/budget/audit writes). `handle_chat` is now a thin
  writer over these; batch workers call `_chat_produce` (prep + forward +
  assemble) and collect the body instead of writing a socket. Behavior-preserving
  (live-diffed byte-for-byte against the prior binary for stream + non-stream).
- **Coarse-lock for concurrent workers** — `_chat_produce` holds a global
  `_chat_lock` mutex around the shared-state phases (prep, assemble) and releases
  it for the network forward, so forwards overlap while cache/budget/audit
  bookkeeping stays serialized. `metrics_record` is now **atomic** (CAS adds) —
  it runs inside the unlocked forward. The single-threaded accept path is
  uncontended (the batch handler joins all workers before the accept loop
  resumes, so workers are the only concurrent actors).

### Fixed (pre-release, during 2.3.1 bring-up)
- **Worker-thread crypto crash** — the chat pipeline's sha256/HMAC/TLS (sigil)
  read a **thread-local crypto scratch bank** that is uninitialized on worker
  threads; the first threaded build corrupted the heap (batch returned fine, the
  *next* request segfaulted). Fixed per sigil's intended API: the handler calls
  `crypto_tls_main_init()` before fan-out and each worker calls
  `crypto_bank_set(1..7)` at entry (bank 0 = main). Caps `BATCH_MAX_PARALLEL` at
  7 (sigil's 8 banks − the main lane). See ADR-009 §5.

### Notes
- See **ADR-009** for the concurrency model, the allocator + sigil thread-safety
  findings, and why the barrier (not `thread_join`) is the correctness primitive.

## [2.3.0] — 2026-06-10

**MCP tool-server endpoints** — `GET /v1/tools/list` + `POST /v1/tools/call`,
backed by **bote 2.7.3**'s JSON-RPC 2.0 registry/dispatcher/codec. Closes the
last open item of the v2.2.x parity arc (was tracked as v2.2.5); the parity arc
is complete, so this ships as **2.3.0**. Also bumps the toolchain to the latest
Cyrius (**6.1.27**). Connection pooling and OpenTelemetry/event-bus observability
move to v2.3.1 and v2.3.2 respectively.

### Added
- **`GET /v1/tools/list`** — lists registered MCP tools in JSON-RPC form
  (`{"jsonrpc":"2.0","id":1,"result":{"tools":[…]}}`). hoosh synthesizes the
  `tools/list` request and returns bote's `ToolRegistry` listing verbatim.
- **`POST /v1/tools/call`** — invokes a tool by name. The request body is an MCP
  JSON-RPC request (`method:"tools/call"`, `params:{name,arguments}`); it is run
  through bote's codec + `Dispatcher`, so `initialize` and `tools/list` are also
  accepted here for full MCP-client compatibility. Unknown tools return a
  JSON-RPC error; an empty body returns 400.
- **`src/lib/mcp.cyr`** — the wiring module. `mcp_init` (called from
  `cmd_serve`) builds the registry + dispatcher and registers a built-in
  `bote_echo` smoke tool so both endpoints are live-verifiable end-to-end.
  szál's 58 tool implementations plug in here once they ship as a Cyrius
  distlib — register them alongside `bote_echo` in `mcp_init` and they appear in
  `tools/list` and dispatch through `tools/call` with no transport changes (the
  registry currently holds echo only).
  **Live-verified**: `tools/list` lists `bote_echo`; `tools/call` round-trips
  arguments as MCP text content; `initialize` negotiates the protocol version;
  unknown tool → JSON-RPC error.
- **Tests + benches** — `mcp_tools` test group (registry/dispatch/codec via the
  real bote-core: list/call/unknown-tool) and `mcp_tools_list` / `mcp_tools_call`
  benches (full JSON-RPC parse → dispatch → serialize: ~4 µs / ~9 µs).

### Changed
- **Toolchain: Cyrius 6.1.21 → 6.1.27** (pin in `cyrius.cyml`). 384 → 392 tests
  green.
- **Stdlib migration: `bigint` + `toml` + `json` → `bayan`.** 6.1.27 consolidated
  the standalone `bigint`/`toml`/`json` snapshot modules into a single `bayan`
  module (it provides `json_parse`/`json_get`, `toml_parse`, and the bigint/u256
  surface). `[deps].stdlib` now lists `bayan` (placed before `sigil`, which
  references its u256 surface) in place of the three; `tests/hoosh.tcyr`'s
  explicit `lib/toml.cyr` + `lib/json.cyr` includes collapse to `lib/bayan.cyr`.
  Without this, a clean checkout's `cyrius deps` fails — `cannot read
  ./lib/{bigint,toml,json}.cyr` — since those files no longer ship.

### Notes
- **bote is vendored, not a `[deps.bote]` block.** Unlike ai-hwaccel (no git
  sub-deps), bote's manifest declares `[deps.libro]` + `[deps.majra]`; `cyrius
  deps` resolves those transitively, pulling libro/majra → bayan/ganita/agnosys
  into the compile set, where the agnos superset collides with bote-core's
  `registry_new` and trips an agnosys slice-include error. bote's `[lib.core]`
  bundle (`dist/bote-core.cyr`) is fully self-contained (9 transport-free
  modules, no includes), so it is committed at `src/vendor/bote-core.cyr` and
  included directly. Living under `src/` keeps `cyrius vet` trust intact and the
  generated file out of the fmt/lint globs. Re-sync with
  `./scripts/sync-bote.sh <tag>`.

## [2.2.4] — 2026-06-10

**Tool calling** across all three remote families (OpenAI, Anthropic, Gemini) —
forward `tools`, surface unified OpenAI `tool_calls`, both non-streaming and
streaming — plus the stale tool-pair prune that completes the compression port.
(The MCP server endpoints `/v1/tools/list` + `/v1/tools/call` are deferred to
v2.2.5, pending szál as a Cyrius distlib.)

### Added
- **Streaming tool-call assembly — OpenAI, Anthropic, Gemini** — `stream:true`
  requests forward `tools` on all three families and convert each provider's
  streaming tool deltas into OpenAI `chat.completion.chunk` `tool_calls` deltas
  (`_sse_tool_chunk` in `_remote_stream_cb`):
  - **OpenAI-compat**: deltas pass straight through (client assembles id/name/
    arguments fragments).
  - **Anthropic**: `content_block_start` (tool_use) → id+name delta;
    `input_json_delta.partial_json` → incremental `arguments` deltas
    (`_emit_anthropic_tool_delta`).
  - **Gemini**: complete `functionCall` parts re-emitted as one tool_call delta
    (reusing `_gemini_tool_calls`).

  The buffered early-error fallback also emits any tool calls.
  **Live-verified** against all three (streamed `get_weather` produced the call
  id+name then incremental arguments for OpenAI/Anthropic, one complete delta for
  Gemini); plain streaming unchanged.
- **Compression: stale tool-pair pruning** (`compression.cyr` `prune_tool_pairs`)
  — completes the `context/compression.rs` port (the half deferred from 2.2.3,
  now that tool-call message structure exists). In a long agentic conversation,
  assistant messages carrying `tool_calls` (and their matching `role:"tool"`
  results, paired by id) that fall before the last 3 tool turns are dropped;
  ordinary turns are untouched. Applied after whitespace collapse when
  `[compression]` is enabled. Self-contained byte-level helpers (`_cmp_*`).
  Unit-tested (5-turn conversation → first 2 call+result pairs pruned, recent 3
  and user turns kept; ≤3 turns unchanged).
- **Tool calling — OpenAI, Anthropic, and Gemini** — the gateway forwards a
  request's `tools` to the provider (converting to each native format) and
  surfaces the model's tool calls back as OpenAI `tool_calls` +
  `finish_reason:"tool_calls"`. Ports `tools/convert.rs`.
  - **Request**: `_extract_tools` lifts the `tools` array (balanced-bracket scan);
    threaded through `retry_forward`/`provider_forward` into the body. OpenAI-compat
    passes them verbatim (`_build_chat_body_raw`); `_tools_convert` maps them to
    Anthropic `input_schema` and Gemini `functionDeclarations`.
  - **Response**: `_extract_openai_tool_calls` (OpenAI), `_anthropic_tool_calls`
    (`content[].tool_use` → `tool_calls`, `input` stringified), and
    `_gemini_tool_calls` (`functionCall` parts → `tool_calls`, synthesised ids).
    Tool-call argument objects are JSON-string-escaped (incl. control chars, so
    Gemini's pretty-printed `args` stay valid).
  - **Live-verified** against all three: `get_weather` called for Paris
    (Anthropic), Tokyo (Gemini), London (OpenAI). Plain (no-tools) requests
    unchanged. Unit-tested: `_extract_tools`, OpenAI→Anthropic/Gemini conversion
    (incl. the `"function"`-as-value-vs-key case), Anthropic `tool_use` parsing.

## [2.2.3] — 2026-06-10

The **Cost & cache intelligence** parity arc — semantic cache, cost optimizer,
prompt compression, cache warming, and the now-functional response cache — plus
the native-TLS-by-default toolchain flip (cyrius 6.1.21) and the `hoosh.toml` →
`hoosh.cyml` config rename.

### Added
- **Semantic cache** (`src/lib/semantic.cyr`, `[semantic_cache]` config) — ports
  `cache/semantic.rs`. On an exact-cache miss, the query is embedded via the
  configured `embedding_model` provider and compared (cosine similarity) to
  stored query embeddings; a match above `threshold` reuses that entry's cached
  response — so semantically-similar (differently-worded) requests hit.
  - **Core**: fixed-point cosine (`cosine_x1000` over ×10000 integer vectors,
    magnitudes via integer sqrt to stay within i64), embedding store
    (`semantic_insert`/`semantic_find`, nearest neighbour above threshold with a
    `max_search` cap), and an embeddings-response float-vector parser
    (`semantic_parse_embedding`, sign/decimal/exponent).
  - **Chat-path wiring** (`handle_chat`): on miss, `_embed_query_body` POSTs the
    (JSON-escaped) request to the embedding provider's `/v1/embeddings`
    (`/api/embeddings` for Ollama), parses the vector, `semantic_find`s a hit, or
    stores the embedding under the exact key after forwarding. Any embedding
    failure degrades silently to a normal forward. Streaming requests are not
    semantically cached.
  - Config: `enabled`, `threshold` (0–1 → ×1000), `embedding_model`, `max_search`.
  - **Unit-tested** (cosine, integer sqrt, store, float parsing incl. scientific
    notation) **and live-verified**: two paraphrased "capital of France" queries
    — the second (different exact-key) hit the semantic cache and returned the
    first's response without re-forwarding.
- **Cost optimizer — cheapest *capable* model recommendation** (`pricing.cyr` +
  `metadata.cyr`, ports `cost/{mod,optimizer}.rs` + the needed slice of
  `provider/metadata.rs`).
  - **Pricing** (`pricing.cyr`): 16-model per-token table (input/output $/M,
    carried ×1000 since no floats), exact → longest-prefix → provider-fallback
    lookup; cost in micro-USD. `POST /v1/cost/estimate` `{model,input_tokens,
    output_tokens}` → estimated cost.
  - **Capability filter** (`metadata.cyr`): per-model tier/context-window/
    vision/tools/system metadata + `classify_complexity` (request → min tier) +
    `meets_requirements`. `POST /v1/cost/recommend` `{input_tokens,
    max_output_tokens,uses_tools,has_vision,has_system_prompt}` classifies the
    request, keeps only configured exact-model routes whose metadata satisfies
    tier/modality/tool/system/context, and returns the **cheapest capable** one
    (with its `tier` + `required_tier`). Wildcard-only/unknown-metadata routes
    are skipped.
  - **Live-verified**: pricing (exact/prefix/fallback/local-free/cheapest-wins);
    capability (plain text → cheapest economy; `has_vision`/large-input → Standard
    tier picks `gpt-4o` over economy models; `uses_tools` excludes a no-tools
    model; over-context → 404). Note: per-token pricing is hardcoded, **not**
    `data/cloud_pricing.json` (which is cloud-GPU $/hour for hardware planning).
- **Prompt compression** (`src/lib/compression.cyr`, `[compression]` config) —
  opt-in whitespace collapse over JSON `content` *values*: runs of whitespace
  (incl. `\n`/`\t`/`\r` escapes) collapse to a single space with leading/trailing
  trim; keys, structure, and other escapes (`\"`, `\\`, `\uXXXX`) are preserved.
  Applied in `handle_chat` before compaction when enabled. Ports the
  whitespace-collapse half of `context/compression.rs`; the stale tool-pair
  prune is deferred to v2.2.4 (needs tool-call message structure). Distinct from
  compaction (which drops whole messages). Unit-tested (6 cases) + live-verified.
- **Cache warming** (`handlers.cyr` `warming_run`/`warming_add`/`_warming_body`,
  `[[warming]]` config) — pre-populates the response cache at startup with
  operator-configured `(model, prompt)` prompts so common requests are instant
  before traffic arrives. Synchronous (the single-threaded runtime has no
  background task); fires one inference per prompt, skips already-cached keys,
  logs `cache warming: N/M entries cached`. Each warmed entry is stored under the
  exact key a client hits by POSTing the canonical body
  `{"model","messages":[{"role":"user","content"}]}` — unit-tested that the
  warmed key equals the client-request key. **Live-verified**: startup warmed
  1/1, client request → cache hit returning the warmed response without
  forwarding.

### Fixed
- **Build output always lands in `build/`** — the `build/` directory is now
  tracked via `build/.gitkeep` so it always exists; the compiler no longer falls
  back to dropping the binary at the repo root when `build/` is absent.
- **Response cache was inert — now wired into `/v1/chat/completions`.** The
  exact-key LRU cache (`cache.cyr`) was configured and exposed via
  `/v1/cache/stats`, but `handle_chat` never read or wrote it, so every request
  hit the provider. Non-streaming requests now compute an exact key
  (`_cache_key` = sha256-hex over model + raw body), short-circuit on a hit
  before rate-limit/budget/forward, and store the response body on a miss.
  **Live-verified**: identical request twice → 1 miss + 1 hit (`/v1/cache/stats`
  `hits:1`). Foundation for cache warming (v2.2.3). Streaming responses are not
  cached.

### Changed
- **Runtime config renamed `hoosh.toml` → `hoosh.cyml`** — matches the project's
  `cyrius.cyml` manifest convention (TOML syntax in a `.cyml` file; still parsed
  by `toml_parse_file`). `load_config` + `/v1/admin/reload` + all docs/comments
  updated. Verified: server loads providers + serves inference from `hoosh.cyml`.
- **Cyrius pin 6.1.20 → 6.1.21**; stdlib re-synced, `cyrius.lock` refreshed. This
  ships sandhi's **native-TLS-by-default** flip.
- **Native TLS is now the default — the gateway builds with no TLS flag**
  (`cyrius build src/main.cyr build/hoosh`). The old opt-in `-D CYRIUS_TLS_NATIVE`
  is gone from CI/release/CLAUDE.md (kept as a deprecated, harmless no-op
  upstream). The crash-prone libssl fdlopen bridge is now the explicit **opt-out**
  via `-D CYRIUS_TLS_LIBSSL`, which hoosh never passes. `main()` still asserts
  native via `sandhi_tls_use_native()` and warns only if a libssl-only build
  disabled it. **Verified**: default build → native active (5 sequential remote
  HTTPS, clean exit, no warning); `-D CYRIUS_TLS_LIBSSL` build → startup warning
  as expected.

## [2.2.2] — 2026-06-10

Data Loss Prevention — the **v2.2.2 parity item**. Ports `dlp/scanner.rs`: the
Cyrius gateway previously had only `ERR_DLP_BLOCKED` + a test stub. Now requests
are scanned for PII/secrets, classified, and routed by privacy policy.

### Added
- **DLP scanner** (`src/lib/dlp.cyr`) — eight built-in PII/secret matchers,
  hand-rolled as byte-level scanners (the Cyrius port carries no regex engine,
  and adding one would be an unnecessary dependency). Patterns and levels mirror
  `BuiltinPatterns::all`: `email`/`ipv4` → Internal, `phone_us` → Confidential,
  `ssn`/`credit_card`/`api_key`/`aws_key`/`github_token` → Restricted. Each
  honours `\b` word boundaries; `dlp_scan_level` returns the highest level found
  and short-circuits on the first Restricted match.
- **Classification levels** (`DlpClass`: Public/Internal/Confidential/Restricted,
  ordered) + `dlp_class_name`/`dlp_class_from_str`.
- **Privacy-aware routing** (`handle_chat`) — when DLP is enabled: Restricted
  content is **blocked** (`403`); Confidential content is forced to a **local
  provider** via the new `router_select_local` (blocked `403` if no local route
  serves the model); Internal and Public pass through. **Live-verified
  end-to-end**: SSN → blocked, US phone on a remote-only model → local-required
  block, clean prompt → normal inference.
- **`[dlp]` config section** (`src/lib/config.cyr`) — `enabled` (default false)
  and `default_level`; documented (disabled) in `hoosh.toml`.

### Tests & benchmarks
- `+19` assertions (**317 pass**): every pattern at its level, highest-level-wins,
  clean/empty → default, disabled → default, and two false-positive guards
  (20-digit run is not a card; leading-dot domain is not an email).
- New bench `dlp_scan_clean_prompt` (~4µs for a typical prompt) — 12 benches total.

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
