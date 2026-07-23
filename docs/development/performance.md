# Performance Testing & Benchmarks

> How hoosh's benchmark system works and how to run it. **Canonical current
> numbers live in [benchmarks.md](../../benchmarks.md)** (auto-generated) — this
> guide does not duplicate them (they drift every run).

---

## The bench system

Cyrius-native, three pieces:

| Piece | Role |
|-------|------|
| `tests/hoosh.bcyr` | The microbenchmark suite (self-contained, mirrors src hot paths) — run with `cyrius bench` |
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

25 CPU microbenchmarks over the per-request hot paths (no live backend — inference
latency is backend-bound and not meaningful to micro-measure here):

- **Routing** — `route_select_20_providers`, `route_round_robin_10`,
  `route_matches_model`
- **Budget** — `pool_reserve_commit`, `pool_available`
- **Cache** — `cache_get_hit`, `cache_get_miss`, `cache_insert`
- **Queue** — `queue_enqueue_dequeue`, `queue_5tier_sort`
- **Tokens / DLP** — `estimate_tokens_per_provider`, `dlp_scan_clean_prompt`
- **MCP** — `mcp_tools_list`, `mcp_tools_call`
- **Batch / Observability** — `batch_split_4`, `latency_bucket_find`,
  `work_queue_push_pop`, `event_publish_ring`
- **Security / accounting** (the work every request pays for) —
  `auth_verify_token`, `auth_verify_wrong_late`, `auth_verify_wrong_early`,
  `rate_limit_check`, `cost_record_known_model`, `cost_record_local_free`,
  `audit_record_sign`

Add a benchmark alongside any new hot-path code (dev-loop step 3) and re-run
`bench-history.sh` so the CSV + `benchmarks.md` pick it up.

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
