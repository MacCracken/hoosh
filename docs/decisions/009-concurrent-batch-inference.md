# ADR-009: Concurrent batch inference

**Status:** Accepted
**Date:** 2026-06-10 (2.3.1)

## Context

`POST /v1/batch` accepts an array of chat requests and must run them
concurrently for throughput (the gateway is otherwise one-request-at-a-time).
The reference is `rust-old/src/inference/batch.rs` — a semaphore-bounded
concurrent executor with per-item results.

The roadmap (race audit, 2026-06-04) listed the multi-threaded path as **BLOCKED**
on the cyrius global allocator not being thread-safe ("~5000 corruptions across
4 threads"). That note predated the toolchain in use.

## Decision

### 1. Re-verify the blocker before believing it

The bump to cyrius 6.1.27 carried `alloc.cyr`'s **v6.0.64 global allocation
lock** — a process-wide CAS spinlock serializing `alloc()`/`alloc_reset()`
across threads. We verified empirically rather than trusting either the comment
or the stale audit: a 4-thread × 200,000-allocation stress, each thread stamping
and re-reading its blocks, reported **0 mismatches**. The allocator is
thread-safe; the blocker is gone. (Lesson: verify claims against the current code
and a live test, not against docs.)

### 2. In-process worker pool, synchronous (2.3.1 scope)

`POST /v1/batch` spawns worker threads that each run the request through
`_chat_produce` and store a response descriptor in a per-item slot. The handler
runs on the accept thread and **joins all workers before returning**, so the
workers are the only concurrent actors — the single-threaded request path never
runs concurrently with them. A fully multi-threaded accept loop (all traffic
concurrent) is deferred to ~2.4.0; async job-id/progress-poll/cancel to a later
2.3.x point release.

### 3. Coarse lock around bookkeeping, forward runs unlocked

The throughput win is overlapping the slow network forwards. `_chat_produce`
holds a global `_chat_lock` mutex around the shared-state phases — `_chat_prep`
(cache/rate/budget reads + reservation) and `_chat_assemble` (budget commit,
audit append, response/semantic cache insert) — and **releases it for
`_chat_forward`** (the provider call). So forwards run concurrently while the
mutations of `_cache`/`_budget`/`_audit_chain` stay serialized. `metrics_record`
is the one shared write inside the unlocked forward, so it was made **atomic**
(CAS adds). The allocator's own spinlock covers `alloc` underneath all of this.

`handle_chat` was refactored to make this possible: a returns-body core
(`_chat_prep` → `_chat_forward` → `_chat_assemble`) that produces a response
descriptor, with `handle_chat` reduced to a thin socket writer over it. The
refactor is behavior-preserving (byte-for-byte live diff, stream + non-stream).

### 4. Atomic counter barrier, not `thread_join` timing

Workers atomically increment `_batch_done` after storing their result; the
handler waits on that counter as the barrier between waves. We do **not** rely on
`thread_join` for the completion guarantee: empirically `thread_join` could
return before the worker had stored its slot (two waves overlapped, effective
concurrency = 2× the cap), which would risk result assembly reading an unfilled
slot. `thread_join` is still called for thread cleanup; the counter is the
correctness primitive. Verified: with the barrier, per-connection arrival
timestamps at the backend show exactly `BATCH_MAX_PARALLEL` forwards per wave and
waves serialize; metrics totals are exact (no lost updates) under a
5×16-unique-prompt stress.

### 5. Per-worker sigil crypto scratch bank (the crash that almost shipped)

The first threaded build crashed — but **not** in the batch: the batch returned
correct results, then the *next* `/v1/chat/completions` segfaulted. Bisection
ruled out concurrency (serial workers crashed too), stack size (8 MB didn't
help), `thread_join`/munmap, `json_parse`, and logging — and pinned it to
`_cache_key`'s **sha256 (sigil)**. Root cause: sigil's crypto primitives
(sha256, HMAC, TLS) read a **thread-local scratch "bank"** index
(`_SIGIL_CBANK_SLOT`); on a worker thread that slot was never initialized, so the
crypto scrawled scratch through a garbage pointer and corrupted adjacent heap
(which the next request then walked into).

sigil is *designed* for this: `SIGIL_CRYPTO_BANKS` (8) lanes, bank 0 for the
main/serial path, banks 1..7 for workers. The fix is exactly what sigil's API
intends — the handler calls `crypto_tls_main_init()` before fan-out, and **each
worker calls `crypto_bank_set(bank)` as its first action** (bank = its index
1..7 within the wave; reused across waves since the barrier drains each wave).
One `crypto_bank_set` covers *all* sigil use on that thread — `_cache_key`, the
audit-chain HMAC, and remote-provider TLS. This caps `BATCH_MAX_PARALLEL` at 7.

Lesson (again): the chat pipeline calls into stdlib crypto with thread-local
state — running *any* such pipeline on a worker thread requires per-thread crypto
init, not just a thread-safe allocator.

## Consequences

- Real concurrency for batches (8 forwards in flight per wave by default),
  bounded to avoid overwhelming a backend; `BATCH_MAX_ITEMS`=64 cap.
- The inter-wave barrier is a brief busy-spin on the accept thread (no
  `sched_yield` wrapper exists in the stdlib). The accept loop is dedicated to
  the batch while it runs, so this is acceptable; a blocking primitive can
  replace it later.
- Unlocks the rest of the concurrency roadmap (multi-threaded accept loop →
  2.4.0) now that allocator thread-safety is established and verified.

## Alternatives considered

- **Trust the roadmap's BLOCKED note** — would have wrongly deferred the whole
  feature; the blocker was already fixed upstream.
- **One global lock around the entire `_chat_produce`** — correct but serializes
  the forwards too, erasing the throughput win.
- **Rely on `thread_join` for the barrier** — unsafe here (returns early); see §4.
- **Multi-threaded accept loop now** — bigger blast radius (every handler +
  all shared state must be thread-safe); deferred to 2.4.0.

## Update (2.3.2): async batch

2.3.1 shipped the *synchronous* executor (the handler blocks until all items
finish). 2.3.2 adds the **async** surface: `POST /v1/batch {"async":true}`
returns a batch id immediately, a background runner thread executes the batch,
and clients poll `GET /v1/batch/{id}` / `POST /v1/batch/{id}/cancel`.

**What changed and why:**

1. **The accept loop now runs concurrently with batch workers.** In 2.3.1 the
   accept thread was *parked* (joining) during a sync batch, so workers were the
   only concurrent actors and only they needed `_chat_lock`. With a background
   runner, the accept loop keeps serving requests — so a `/v1/chat/completions`
   served mid-batch races workers on `_cache`/`_budget`/`_audit`. The fix is the
   **sync pass**: `handle_chat` (and the streaming audit append) now take
   `_chat_lock` around their shared-state phases, exactly like a worker. This is
   a focused slice of the broader sync work scoped for 2.4.0.

2. **`BatchState` + registry, lock-free reads.** Per-batch state (total,
   completed, failed, status, cancel flag, per-item result slots) lives in a
   `BatchState`; `completed`/`failed`/`status`/`cancel` are accessed atomically
   and result slots are write-once, so `GET` needs no lock. The `id → BatchState`
   registry is touched only by the single-threaded accept loop, so it needs no
   lock either; the runner holds its `BatchState` pointer directly.

3. **Request bodies are copied at submit.** The runner outlives the POST handler,
   but the accept loop reuses its receive buffer for the next request — so
   `batch_submit` copies each item's body into heap-owned memory.

4. **One batch executes at a time** (`_batch_exec_lock`, shared by sync + async).
   This keeps the crypto-bank lanes (1..7) exclusive to the running batch — bank
   0 is the accept thread — avoiding the cross-batch lane collision that 8 banks
   can't otherwise cover. Extra submitted batches queue (`status:"queued"`).
   Cancellation is checked before each wave (in-flight items finish).

**Deferred:** concurrent execution of multiple async batches (needs a global lane
pool / semaphore across batches), and eviction of completed batches from the
registry (currently retained for the process lifetime).
