# ADR-011: Multi-threaded accept loop — unified worker pool

**Status:** Accepted
**Date:** 2026-06-10 (2.4.0)

## Context

Through 2.3.x the accept loop was single-threaded: `accept → recv → route →
respond → close`, one request at a time. A slow inference forward (the dominant
cost — backend latency) blocked the whole gateway; only `/v1/batch` ran work
concurrently (on its own worker threads). 2.3.x had already done the groundwork:
the allocator is thread-safe (cyrius 6.0.64 CAS spinlock, re-verified by a
4-thread × 200k-alloc stress, 0 corruption), the chat path is structured as
prep(lock) → forward(unlock) → assemble(lock) under `_chat_lock`, and metrics are
atomic.

The hard constraint is **sigil crypto banks**: `SIGIL_CRYPTO_BANKS = 8`, bank 0 =
main thread, banks 1..7 = workers. Any thread doing sha256/HMAC/TLS (the chat
path's `_cache_key`, the audit HMAC, remote TLS) needs its own bank or it
corrupts the heap ([ADR 009 §5](009-concurrent-batch-inference.md)). So **at most
7 concurrent crypto-using worker threads** exist, ever — shared between
interactive requests and batch.

## Decision

### Unified worker pool (7 workers, banks 1..7)

`src/lib/pool.cyr`: a fixed pool of `WORKER_COUNT = 7` threads, each calling
`crypto_bank_set(i)` once at startup (bank scratch readied by
`crypto_tls_main_init()` on main first). The accept loop on the main thread only
**accepts + enqueues**; the workers drain a queue and run the full request chain.

Batch items are the **same kind of job** as connections, so HTTP traffic and
batch work share the one pool and the full bank budget — there is no longer a
separate batch "lane pool", and no static interactive/batch split. Total
concurrent crypto threads = 7, allocated dynamically to whatever mix is present.

Alternatives rejected:
- **Disjoint split** (e.g. 5 interactive + 2 batch banks): simple, but caps each
  pool statically and regresses batch from 7→2.
- **Thread-per-connection**: unbounded threads (DoS) and a lane-hold deadlock when
  a sync batch nests under a lane-holding connection thread.

### Work queue = a bounded ring with non-blocking pop

A mutex-guarded ring of job pointers (`wq_push` / `wq_pop`). `wq_pop` is
**non-blocking** (returns 0 when empty) — this is what makes **work-stealing**
possible: a synchronous `/v1/batch` runs on a worker, enqueues its items, and
then processes queued jobs itself while waiting on its (per-batch, local)
completion counter. So a sync batch can't stall the pool even if many run at once
— there's always a worker making progress. Async batch spawns a **bankless
coordinator thread** that enqueues + waits on `BS_COMPLETED`; the pool workers do
the crypto. The old lane acquire/release and per-worker `crypto_bank_set` in batch
are gone.

### Synchronization pass

With 7 workers running every handler concurrently, shared mutable state is
guarded:
- **Chat** — already correct via `_chat_lock` (prep/assemble locked, forward
  unlocked → forwards overlap, the throughput win).
- **Batch registry** (map + id list) — a new `_batch_reg_lock` around submit /
  get / cancel / evict (map+vec mutation can't race).
- **Token handlers** (`/v1/tokens/*`) — `_chat_lock` around the budget ops (plain
  counters, shared with chat's reserve/commit); values captured under the lock,
  response built unlocked.
- **Health map** — `_chat_lock` around the per-route get/set.
- **Metrics / events / OTLP** — already atomic / own locks.

### Lock-free reads via never-frees

The bump allocator never frees, which makes some hot reads lock-free:
- **Config hot-reload** rebuilds `_router` and assigns the global pointer (an
  aligned 64-bit store is atomic). Lock-free readers (pure forwarders) load
  old-or-new — both point at complete, never-reclaimed routers. The in-place
  `_budget` pool append is serialized under `_chat_lock` (shared with chat/token
  readers). Reload itself takes `_chat_lock`.
- Routing reads in the chat path are already under `_chat_lock`.

### Worker stacks

Handlers ran on the 8 MB main stack before; workers default to 64 KB
(`THREAD_STACK_SIZE`). Batch workers already proved the chat path fits 64 KB, but
`pool_start` bumps `THREAD_STACK_SIZE` to 1 MB before spawning for headroom
(7 × 1 MB is negligible).

## Consequences

- Interactive throughput scales to 7 concurrent requests instead of 1.
  Live-verified: 14 concurrent 50 ms chats in **115 ms** (vs ~700 ms serialized),
  50 in 417 ms; 200 mixed requests + reload-under-load survived; sync + async +
  concurrent batches all correct; no crash/corruption (the delayed-corruption
  bank failure mode did not surface — permanent per-worker banks hold).
- Concurrency ceiling is 7 (the bank budget). A per-thread-arena allocator + more
  banks would lift it; out of scope here.
- The accept loop no longer blocks on slow forwards.
- Unblocks threaded hardware detection (deferred on the single-threaded runtime).

## Alternatives considered

See "Decision" — disjoint bank split and thread-per-connection were both rejected
for the unified pool. Per-handler fine-grained locking (vs the coarse `_chat_lock`
+ batch registry lock) was deferred: the fast handlers serialize cheaply under
`_chat_lock` while the slow forwards already run unlocked, so the coarse lock
captures the throughput win without the audit surface of fine-grained locking.

## Update (2.4.5): sync-pass audit follow-ups

A hardening audit found two read paths the original sync pass missed — both
iterate a shared map structurally concurrent with a writer on the pool workers:
`GET /v1/cache/stats` (`map_count` vs `cache_insert`/evict) and
`GET /v1/tokens/pools` (`map_keys(_budget)` vs reload's `budget_add_pool` +
reserve/commit). A map rehash mid-iteration is a crash, not just a torn read, so
both now take `_chat_lock` for the snapshot. Lesson: the sync pass must cover
**readers that iterate** shared maps, not only writers — a `map_count`/`map_keys`
during another worker's `map_set`/`map_delete` is unsafe. Also un-deaded the
lowest-latency strategy, whose per-request EMA update uses a dedicated `_lat_lock`
(not `_chat_lock`) so latency recording doesn't serialize with chat bookkeeping.

## Update (2.9.1): the crypto-bank constraint is gone

The hard constraint in *Context* no longer holds. sigil 3.13.14 (folded into cyrius 6.7.18, which 2.9.1 pins) replaced
per-thread banks with a pool of `SIGIL_CRYPTO_BANKS = 64` lanes. Every sigil entry point that reaches module-global
crypto state borrows a lane for that one operation and returns it, so any thread may hash, sign or do TLS, and
`crypto_bank_set` is a no-op (2.9.1 stops calling it). Through 3.13.13 the 64th crypto-touching thread wrapped the lane
counter and shared a lane with another thread; sigil fixed that as CYRIUS-2026-0040.

The pool itself is unchanged: still `WORKER_COUNT = 7`, with HTTPS work (OTLP export, the catalog refresh) still routed to
a worker rather than done on the helper threads. Those are now choices, not requirements. A larger pool, or a TLS health
probe on the prober thread, is bounded by 64 concurrent crypto operations rather than 8 banks.

## Addendum (2.10.1): a target with no second thread

This decision assumed `thread_create` starts a thread. On AGNOS it does not: `lib/thread_agnos.cyr` runs the body inline and says so with `THREADS_CONCURRENT = 0`. A body that loops forever never returned, so `coarse_clock_start` held `main`, and the accept loop never ran. hoosh logged that it was serving and read no request, and nothing had ever run it there to notice. thoth's first real turn on AGNOS found it.

On such a target the accept loop is the one thread. No forever-thread starts, and the loop runs each job itself (`wq_serial_turn`), yielding with `pause`#14 while `accept`#57 has nothing, because `sleep_ms`#41 holds the CPU there. The choice is keyed on `THREADS_CONCURRENT`, the capability, and not on the target's name.

What it costs: one request at a time, so a long stream holds a second client in the listen queue. A batch still drains through its own work-stealing barrier, on that one thread. Health probes after the first sweep, catalog refresh, hardware re-detection and OTLP export do not run. `pool_start` says all of that at startup. With real threads nothing changes: `wq_serial_turn` returns at its first line, and `sock_accept` blocks as it did.
