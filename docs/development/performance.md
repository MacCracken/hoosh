# Performance Testing & Benchmarks

> How hoosh's benchmark system works and how to run it. **Canonical current
> numbers live in [benchmarks.md](../../benchmarks.md)** (auto-generated) — this
> guide does not duplicate them (they drift every run).

---

## The bench system

Cyrius-native, three pieces:

| Piece | Role |
|-------|------|
| `tests/hoosh.bcyr` | The microbenchmark suite, on the real `src/` code (since 2.9.24; it was copies before) — run with `cyrius bench` |
| `scripts/bench-history.sh` | Runs the suite, appends a row per benchmark to `bench-history.csv`, regenerates `benchmarks.md` |
| `bench-history.csv` | Full append-only history (timestamp, commit, branch, suite, benchmark, avg/min/max ns, iters) |
| `benchmarks.md` | Canonical current view — baseline / previous / current columns with regression markers |

**Benchmarks are a mandatory release gate** (CLAUDE.md). CI runs
`bench-history.sh` and fails the build if the suite doesn't run; a release ships
without a fresh run only with an explicit maintainer waiver
(`CYRIUS_SKIP_BENCH=1`). Never claim a performance change without a bench run.

---

## Running

```bash
# Run the suite directly
cyrius bench tests/hoosh.bcyr

# Run + record history + regenerate benchmarks.md (what CI does)
./scripts/bench-history.sh bench-history.csv benchmarks.md
```

---

## What's benched

45 CPU microbenchmarks over the per-request hot paths, each calling the gateway's own
function (no live backend — inference latency is backend-bound and not meaningful to
micro-measure here). The clock-reading rows run on the production coarse clock, which
the bench sets as the gateway's ticker would:

- **Routing** — `route_select_20_providers`, `route_round_robin_10`,
  `route_matches_model`
- **Budget** — `pool_reserve_commit`, `pool_available`
- **Cache** — `cache_get_hit`, `cache_get_miss`, `cache_insert` (one key, replaced),
  `cache_insert_full` (a full 10000-entry cache taking new keys, 2.10.0)
- **Tokens / DLP** — `estimate_tokens_per_provider`, `dlp_scan_clean_prompt`
- **MCP** — `mcp_tools_list`, `mcp_tools_call`
- **Batch / Observability** — `batch_split_4`, `latency_bucket_find`,
  `work_queue_push_pop`, `event_publish_ring`
- **Security / accounting** (the work every request pays for) —
  `auth_verify_token`, `auth_verify_wrong_late`, `auth_verify_wrong_early`,
  `rate_limit_check`, `cost_record_known_model`, `cost_record_local_free`,
  `audit_record_sign`
- **With a live catalog** (three routes listing 100 models each, as a gateway with
  `catalog = "live"` routes has) — `cost_record_live_catalog`,
  `metadata_lookup_live_catalog` (2.9.25)
- **The model listings at 200 entries** (2.10.2) — `catalog_body_build_200`, what every
  `GET /v1/models/catalog` did through 2.10.1 and one per change does now; `catalog_body_held_200`,
  every other GET; `models_list_200_arena`, `GET /v1/models`' body in the thread's arena
- **A provider's refusal** (2.10.2) — `provider_err_set_utf8_511b`, the clip over three-byte UTF-8,
  the slow branch of the error path
- **The request path in the request arena** (2.10.3) — `trace_begin_child`, a request's trace context under an
  inbound `traceparent`; `cache_key_arena`, `chat_body_arena` and `provider_body_arena`, what a buffered chat builds,
  each released by the next job's start; `sse_chunk_send`, one streamed token from the reused chunk buffer to
  `/dev/null`; `event_completed_ring`, a completion's event built on the stack and copied into its recycled ring slot.
  Since 2.10.3 `event_publish_ring` includes that copy (it stored a pointer before), so it reads ~55 ns higher
- **What a request keeps** (2.11.2) is not a bench row: the suite asserts it (`alloc_used()` deltas of 0 for each path and
  keeper), and `scripts/memory-growth.py` measures it on a real gateway against a `-D HOOSH_ALLOC_PROBE` build
  (`--cache-max`, `--hits`, `--audit`, `--otlp`, `--provider`). `cache_insert` times a replace of one key, which since
  2.11.2 takes the new value in place and allocates nothing; `audit_record_sign` signs on the stack
- **Speech** (2.12.2) — `speech_plan_sentence`, shabda's conversion of one 67-byte sentence and the layout of its
  reply (102 µs); `speech_render_sentence`, that reply's 2.4 s of 24 kHz audio rendered phoneme by phoneme,
  crossfaded and written as PCM to `/dev/null` (50.6 ms, about 48× real time). `scripts/memory-growth.py --speech`
  measures what a speech request keeps

Add a benchmark alongside any new hot-path code (dev-loop step 3) and re-run
`bench-history.sh` so the CSV + `benchmarks.md` pick it up. Call the real function, never a
copy: through 2.9.23 the suite timed copies of `src/`, and five rows were timing far less
work than the gateway does (CHANGELOG 2.9.24). Time heap strings (`_b_s`), never a bare
literal in a timed loop (CHANGELOG 2.9.11).

### Reading the auth benchmarks

The three `auth_verify_*` cases use **equal-length** tokens on purpose. A
constant-time compare short-circuits on a length mismatch — that is unavoidable
and the length is not secret — so benching a 25-byte token against a 24-byte one
measures the early exit, not the property that matters. What must not vary is a
same-length token differing **early** versus **late**: if a first-byte mismatch is
cheaper than a last-byte one, the compare leaks a prefix oracle and a token can be
recovered byte by byte. All three currently land at 100–101 ns.

### Live / end-to-end

`scripts/bench-live.sh` (opt-in, not in CI) measures a running gateway end to end.
It needs a live hoosh and backend, so it measures a machine and a model rather
than the code — and each iteration forks a `curl`, so its ~5 ms floor is the
harness. Use it for relative comparisons only.

### Nanosecond-scale noise

`estimate_tokens_per_provider` and `pool_available` sit at 4–9 ns, where 1 ns of
timer quantization reads as a >10% swing. Check a flagged result across repeated
runs before treating it as a regression.

---

## Test environment

Numbers in `benchmarks.md` are from the development machine and are **relative,
not absolute** — the value is the baseline→current trend (regression markers),
not the wall-clock figure. The CSV records the commit for each run so regressions
are bisectable.
