# Hoosh — Current State

> Refreshed every release. CLAUDE.md is preferences/process (durable); this file
> is **state** (volatile). Per-release detail is canonical in
> [CHANGELOG.md](../../CHANGELOG.md); the forward plan is in [roadmap.md](roadmap.md);
> doc currency is tracked in [doc-health.md](../doc-health.md).

## Current state

| | |
|---|---|
| **Version** | **2.5.11** (security & hardening sweep — closes the rust-old parity arc) |
| **Toolchain** | Cyrius pin **6.4.62** (`cyrius.cyml`); `ai-hwaccel` **2.3.14** |
| **Binary** (x86_64 static ELF) | ~15 MB default build; smaller under `CYRIUS_DCE=1` |
| **Source** | ~11,070 lines / 32 files (`src/main.cyr` + 31 `src/lib/*.cyr`) + 2 vendored distlib bundles |
| **Tests** | 663 assertions · 141 groups (`tests/hoosh.tcyr`) |
| **Benchmarks** | 25 (`tests/hoosh.bcyr`); CSV history + `benchmarks.md` (release gate) |
| **Fuzz** | 4 targets (`fuzz/*.fcyr`) — batch split, trace extract, inference request, message content |
| **Coverage** | symbol coverage 32% (`scripts/coverage.sh`, CI floor 30%) |
| **Providers** | 17 (9 local incl. vLLM/TensorRT-LLM/ONNX + Whisper-STT→svara, 8 remote) |
| **ADRs** | 11 (`docs/decisions/`) |
| **Concurrency** | unified 7-worker pool (banks 1..7); accept loop enqueues — [ADR 011](../decisions/011-multithreaded-accept-loop.md) |

## Active cycle — v2.5.x arc: COMPLETE

The rust-old parity closeout arc (**2.5.1 – 2.5.11**) is done. It began with a
full behavioral diff of the archived Rust tree (1,007 behaviors catalogued; see
[rust-old-parity-review.md](rust-old-parity-review.md)) which found the port
matched rust-old's *surface area* but not its *request path*.

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
| **2.5.9** | ai-hwaccel's threaded detector corrupts the registry (filed upstream; hoosh uses the serial one) |
| **2.5.11** | RSS grew 64 KiB per request served — an OOM proportional to traffic handled |

Twice the self-contained test mirror was **correct** while `src/` had drifted
(2.5.6 pricing, 2.5.7 audit linkage) and the suite stayed green. That failure mode
is unguarded and is the top structural item in the roadmap.

## Open

See [roadmap.md](roadmap.md). Headline: the **per-request arena** — 2.5.11 removed
the 64 KiB-per-request leak (~128 MB → 2.0 MB over 2000 requests) but ~1 KiB per
request still accumulates in response building, and the allocator never frees.

> **Handoff (2026-07-23):** 2.5.11 closes the arc. Two hot-path wins worth
> knowing: a coarse clock ticker replaced `clock_now_ms` in the cache and rate
> limiter (`cache_get_hit` 1487→113 ns, `rate_limit_check` 1409→8 ns), and the
> per-connection 64 KiB allocation became a per-thread reused buffer. Buffer reuse
> is only safe because `batch_submit` deep-copies async batch items
> (`src/lib/batch.cyr`) — **verify that copy still exists** before touching
> `_handle_conn`'s buffer lifetime. Also: hoosh's rate limiter is now
> `hoosh_ratelimit_*` because the vendored majra exports the same names with
> different arities and resolution was by include order.
