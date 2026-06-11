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

16 CPU microbenchmarks over the per-request hot paths (no live backend — inference
latency is backend-bound and not meaningful to micro-measure here):

- **Routing** — `route_select_20_providers`, `route_round_robin_10`,
  `route_matches_model`
- **Budget** — `pool_reserve_commit`, `pool_available`
- **Cache** — `cache_get_hit`, `cache_get_miss`, `cache_insert`
- **Queue** — `queue_enqueue_dequeue`, `queue_5tier_sort`
- **Tokens / DLP** — `estimate_tokens_per_provider`, `dlp_scan_clean_prompt`
- **MCP** — `mcp_tools_list`, `mcp_tools_call`
- **Batch / Observability** — `batch_split_4`, `latency_bucket_find`

Add a benchmark alongside any new hot-path code (dev-loop step 3) and re-run
`bench-history.sh` so the CSV + `benchmarks.md` pick it up.

---

## Test environment

Numbers in `benchmarks.md` are from the development machine and are **relative,
not absolute** — the value is the baseline→current trend (regression markers),
not the wall-clock figure. The CSV records the commit for each run so regressions
are bisectable.
