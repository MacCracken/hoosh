# ADR-008: Optional Durable Persistence via patra

**Status:** Accepted
**Date:** 2026-06-04

## Context

The HMAC audit chain (tamper-evident request log) and token-budget usage were
in-memory only — both reset on every restart. For the audit chain this defeats
the point (history and the linked-hash continuity are lost); for budgets it means
usage accounting can't survive a restart. Cyrius 6.0.x ships `patra` (an embedded
SQL database) in the stdlib, making durable persistence available without a new
external dependency.

## Decision

Add an **opt-in** persistence layer (`src/lib/storage.cyr`) backed by patra,
enabled only when `hoosh.cyml` sets `[[storage]] path = "..."`. When unset,
`_storage == 0` and every storage entry point is a no-op — hoosh runs fully
in-memory, byte-for-byte as before (backward compatible).

### Schema (two tables, created on open; `PATRA_ERR_EXISTS` on reopen is ignored)
- `audit (id INT, ts INT, event STR, level STR, message TEXT, provider STR,
  model STR, signature STR, prev_hash STR, entry_hash STR)` — `message` is
  `TEXT` (variable length); everything else fits `STR` (256 B). Inserts use the
  **typed `patra_insert_row`**, never SQL-string building, so audit messages with
  quotes/commas can't break or inject SQL.
- `budgets (name STR, capacity INT, used INT)` — one row per pool, written with
  delete-then-typed-insert set-semantics on every `pool_commit`. Pool names come
  from trusted config.

### Write-through + restore
- `audit_record` (audit.cyr) and `pool_commit` (budget.cyr) call the storage
  hooks after updating in-memory state — no-ops when disabled.
- On startup (after `load_config`): `storage_restore_audit` rebuilds the chain's
  linked list in `id` order, taking hashes from disk and setting `last_hash` +
  `next_id` so new entries continue the existing chain; `storage_restore_budgets`
  restores each pool's `used`.
- patra requires `fl_init()` + `patra_init()` before use — called in `main()`
  before `storage_open`.

## Consequences

- **Positive:** the audit chain and budgets survive restarts; no new external
  dependency (patra is stdlib); typed inserts make the audit write path
  injection-safe; opt-in keeps the default zero-cost and behavior-identical.
- **Negative / constraints:**
  - **patra is single-threaded** — all DB access must be serialized once the
    threaded accept loop lands (roadmap concurrency item); the storage layer will
    need a mutex or a dedicated DB-owner there.
  - `STR` columns are capped at 256 B (fine for hoosh's event/level/provider/
    model/hash fields); long content uses `TEXT`.
  - Budget persistence rewrites a pool's row per commit (set-semantics) rather
    than an in-place UPDATE — simpler and idempotent, slightly more writes.
- **Verification:** budget persistence is covered end-to-end (commit via
  `/v1/tokens/report`, restart, `used` restored). The audit table's typed
  multi-column + `TEXT` insert and ordered restore were validated against patra
  directly (arbitrary message content round-trips); the audit write path can't be
  driven from an endpoint without a live provider.
