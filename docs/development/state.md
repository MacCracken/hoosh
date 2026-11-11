# Hoosh — Current State

> Refreshed every release. CLAUDE.md is preferences/process (durable); this file
> is **state** (volatile). Per-release detail is canonical in
> [CHANGELOG.md](../../CHANGELOG.md); the forward plan is in [roadmap.md](roadmap.md);
> doc currency is tracked in [doc-health.md](../doc-health.md).

## Current state

| | |
|---|---|
| **Version** | **2.11.2** (the rest of [ADR 013](../decisions/013-request-memory.md): a request keeps nothing on the heap, and the response cache, the audit chain and the OTLP ring recycle what they drop through `keep.cyr`; tentib 1.5.1 and the serving-only load. 2.11.1: tentib 1.4.0, one cached decode step per token; 2.11.0: the `type = "tentib"` provider, [ADR 014](../decisions/014-in-process-ternary-inference.md)) |
| **Toolchain** | Cyrius pin **6.7.18** (`cyrius.cyml`); `ai-hwaccel` **2.5.6**; `tentib` **1.5.1** (since 2.11.2; 1.4.0 at 2.11.1, 1.3.0 at 2.11.0; brings `rosnet` 1.4.1 and `tyche` 1.2.2); vendored `bote-core` **3.8.15** (since 2.10.1), `majra` **2.13.0**; every one the latest tag on 2026-11-11 |
| **Binary** (x86_64 static ELF) | ~2.7 MB under `CYRIUS_DCE=1` (the release build; DCE compacts dead code out since cyrius 6.6.20), ~4.2 MB without |
| **Source** | ~18,600 lines / 48 files (`src/main.cyr` + 47 `src/lib/*.cyr`) + 2 vendored distlib bundles |
| **Tests** | 2263 assertions, 187 groups (`tests/hoosh.tcyr`); 41 of 47 `src/lib` modules are included for real (`scripts/coverage.sh` lists them); no copies of `src/` code remain; `scripts/wire-smoke.py` reads a real gateway as a strict client does; `scripts/memory-growth.py` measures what each path and keeper keeps per request |
| **Benchmarks** | 41 (`tests/hoosh.bcyr`, on the real `src/` code since 2.9.24; two live-catalog rows since 2.9.25, `cache_insert_full` since 2.10.0, the listings and `provider_err_set` since 2.10.2, the trace context, the request arena's builders, a stream chunk and a built event since 2.10.3, a tentib reply, load and unchanged reload since 2.11.0, a 22-token reply and a same-shape load since 2.11.1); CSV history + `benchmarks.md` (release gate) |
| **Fuzz** | 4 targets (`fuzz/*.fcyr`) — batch split, trace extract (on the real `trace.cyr` since 2.10.3), inference request, message content |
| **Coverage** | symbol coverage 58% (506/858; `scripts/coverage.sh`, CI floor 30%) |
| **Providers** | 18 (9 local incl. vLLM/TensorRT-LLM/ONNX + Whisper, a type no endpoint uses yet; 1 in-process: tentib since 2.11.0; 8 remote) |
| **ADRs** | 14 (`docs/decisions/`) |
| **Model catalog** | live per route (Anthropic / Gemini / OpenAI-compatible / Ollama lists), refreshed at startup, on reload, on `POST /v1/models/refresh` and every 6 h; compiled fallback table refreshed 2026-10-09 (107 metadata rows, 81 price rows) |
| **Concurrency** | unified 7-worker pool (sigil crypto lanes from a pool of 64 since 2.9.1; no per-thread banks); accept loop enqueues — [ADR 011](../decisions/011-multithreaded-accept-loop.md); on a target with no second thread (AGNOS) the accept loop runs each job itself ([ADR 011](../decisions/011-multithreaded-accept-loop.md)'s 2.10.1 addendum) |

## 2.6.5 — errors stop being laundered into successes

⭐ **The failure mode this release exists to remove: a well-formed, successful, EMPTY completion.**
Three ways hoosh could produce one, all now closed.

1. **An in-stream provider `error` event was silently dropped.** A provider that has already sent
   `200 OK` reports a mid-stream failure as an *event* — Anthropic names the SSE event `error`,
   OpenAI-compatible backends put an `error` object in a data frame. `_remote_stream_cb` looked only for
   text, thinking and tool deltas, so the frame extracted to empty everywhere, was skipped, and the
   stream ended normally. Nothing reached hoosh's log either. Measured live: a request the
   **non-streaming** path answered fine came back from the streaming path with no content, and thoth's
   operator spent a session believing the model had gone quiet. Recognition is **structural, never a
   substring search** — a model's prose routinely contains the word "error".
2. **The provider's error body was thrown away.** A non-2xx logged one fixed sentence and dropped the
   body, so the most useful fact in the failure reached neither the log nor the caller. Now captured
   (bounded, control bytes flattened so one error stays one log line) and surfaced both ways.
3. **"provider backend unreachable" was the wrong words.** A 400/404 means the provider *was* reached
   and refused; the old wording sent operators to look at a backend that had answered instantly.

New modules, both created so the surface could be TESTED: `src/lib/provider_err.cyr` (capture,
sanitising, renderings, the SSE error frame, the recognisers) and `src/lib/jsonlite.cyr` (a pure lift of
the five byte-level JSON readers out of `provider.cyr`). The reason is mechanical: **`provider.cyr`
reaches `_router`, a global defined in `main.cyr`, so it cannot be included by the test binary** — and
untrusted-byte parsing is exactly what must be tested against the real implementation, not a mirror.

✅ **Since closed:** the sandhi SSE event loss at read boundaries (what made hoosh forward a tool call whose
`id`/`name` frame never arrived) was fixed in **sandhi 1.9.15**, which the cyrius stdlib hoosh pins carries (1.13.7 at 6.7.18);
2.8.0 adds a test that splits a `tool_use` start at every byte, and drops any orphan argument fragment besides.

## Active cycle — v2.5.x arc: COMPLETE

The rust-old parity closeout arc (**2.5.1 – 2.5.11**) is done. It began with a
full behavioral diff of the archived Rust tree (1,007 behaviors catalogued; see
[rust-old-parity-review.md](rust-old-parity-review.md)) which found the port
matched rust-old's *surface area* but not its *request path*. 2.7.0 and 2.7.1 closed the remaining gaps
before the Rust tree's deletion ([rust-old-retirement.md](rust-old-retirement.md)).

Ten bands closed that gap and an eleventh hardened the result. The more useful
list is what the arc found that the review did not — the defects only visible
once the code was actually exercised:

| | |
|---|---|
| **2.5.1** | `[server] bind` was never read: the gateway listened on `0.0.0.0` while its own config said loopback, with auth optional |
| **2.5.2** | `max_tokens` was invisible to `json_get` when it followed `messages`, so the budget silently used its 2048 default on real client traffic |
| **2.5.3** | No socket timeouts at all — a wedged backend pinned a worker forever; 7 such calls hang the gateway |
| **2.5.4** | `clock_now_ms` measured at 1.351 µs, a syscall with no vDSO path — the finding that drove 2.5.11 |
| **2.5.6** | Local providers were billed at hosted-model prices (`llama-3.3-70b` on Ollama matched the Groq row) |
| **2.5.7** | Compaction dropped the system prompt for any client whose JSON encoder emits `"role": "system"` with a space — i.e. essentially all of them |
| **2.5.7** | The audit signing key was compiled into the binary; `audit_verify` could not detect a deleted record |
| **2.5.8** | `sys_exit` is thread-exit, so a clean shutdown left the process alive; `crypto_tls_main_init` was trapped inside `cmd_serve`, segfaulting every non-serve path that made an HTTP request |
| **2.5.9** | ai-hwaccel's threaded detector corrupted the registry (fixed upstream in 2.3.25; hoosh still runs the serial one, see 2.6.12) |
| **2.5.11** | RSS grew 64 KiB per request served — an OOM proportional to traffic handled |

Twice the self-contained test mirror was **correct** while `src/` had drifted
(2.5.6 pricing, 2.5.7 audit linkage) and the suite stayed green. That failure mode
is unguarded and is the top structural item in the roadmap.

## Open

See [roadmap.md](roadmap.md), banded by release (re-checked 2026-11-11):
- **2.12.x, next:** the command line through cmdit 1.5.1 (2.12.0), speech (`POST /v1/audio/speech`) through shabda and
  svara, served in-process, and szál 2.3.2's MCP tools behind an allow-list.
- **2.13.x – 2.15.x:** accounting and capacity (cache tokens, cost recommendations over the live catalog, a
  configurable worker pool), requests for current models (thinking on Gemini and Ollama, multimodal parts, the
  Responses API, sampling on a tentib route), observability (nested spans, OTLP/protobuf, an AGNOS CI lane).
- **Patches, on whichever minor is current (none gates a band):** what is left of ADR 013 (the semantic index; the
  hardware, MCP, token-pool and batch paths, not measured yet), the layout-dependent string costs waiting on cyrius
  and ai-hwaccel, and remote TLS connection pooling, measured at −33 ms per request and waiting on two sandhi
  preconditions.
- **2.x.x:** tentib's next releases (a model's own vocabulary, the 2-bit store, model handles) and demand-gated items.
- **3.0.0 backlog:** the compatibility forms a major would drop (`queued`/`pending`, `rate_limit_rpm`, the bare
  `serve` port).
- **Memory:** ~170–190 KB per new TLS connection remains, in sandhi, which opens TLS on the global heap even when given
  an arena (upstream-gated).

> **Handoff (2026-07-23):** 2.5.11 closes the arc. Two hot-path wins worth
> knowing: a coarse clock ticker replaced `clock_now_ms` in the cache and rate
> limiter (`cache_get_hit` 1487→113 ns, `rate_limit_check` 1409→8 ns), and the
> per-connection 64 KiB allocation became a per-thread reused buffer. Buffer reuse
> is only safe because `batch_submit` deep-copies async batch items
> (`src/lib/batch.cyr`) — **verify that copy still exists** before touching
> `_handle_conn`'s buffer lifetime. Also: hoosh's rate limiter is now
> `hoosh_ratelimit_*` because the vendored majra exports the same names with
> different arities and resolution was by include order.
